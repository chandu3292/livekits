import os
import time
import pickle
import json
import asyncio
import uuid
import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from google import genai
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from livekit import api
from outbound_calls import make_outbound_call
from agent_personas import list_personas, get_persona, DEFAULT_PERSONA_ID
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import logging
import datetime
from datetime import datetime, timedelta, timezone
from calendar_integration import get_appointment_manager
import warnings

# Suppress the legacy cryptography warning from pypdf/other libs
warnings.filterwarnings("ignore", category=UserWarning, module='cryptography')
warnings.filterwarnings("ignore", message=".*ARC4.*")

# Configure Loggers
logger = logging.getLogger("server")
rag_logger = logging.getLogger("rag")

# 1. Setup FastAPI wrapper
app = FastAPI()
mcp = FastMCP("Vector RAG 🧠")

# Load environment variables from .env
load_dotenv()

# Configuration
# Indic SBERT Model
indicator_model = SentenceTransformer("l3cube-pune/indic-sentence-similarity-sbert")
EMBEDDING_DIMENSIONS = indicator_model.get_sentence_embedding_dimension()
CHUNK_SIZE = 1000 # Slightly smaller for better precision
CHUNK_OVERLAP = 200 # Larger overlap for better context
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
google_client = None
if os.getenv("GOOGLE_API_KEY"):
    google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class KnowledgeBase:
    # ── Valkey / RediSearch shared state ──────────────────────────────────────
    VALKEY_INDEX      = "idx:livekits:vectors:global"
    VALKEY_KEY_PREFIX = "lk:store:"
    _valkey_index_verified: bool = False
    _valkey_client = None

    def __init__(self):
        self.backend = os.getenv("VECTOR_STORE", "faiss").lower()
        if self.backend not in ("faiss", "valkey"):
            rag_logger.warning(f"Unknown VECTOR_STORE='{self.backend}', defaulting to faiss")
            self.backend = "faiss"

        if self.backend == "faiss":
            self.metadata = []  # list of {text, source, doc_id}
            self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
            rag_logger.info("KnowledgeBase initialised (backend=faiss)")
        else:
            self._connect_valkey()
            try:
                self._ensure_valkey_index()
                rag_logger.info("KnowledgeBase initialised (backend=valkey)")
            except Exception as e:
                rag_logger.error(f"[VALKEY] Index setup failed at startup: {e}")
                rag_logger.warning("[VALKEY] Server will start but Valkey ops will retry on first use")

    def _connect_valkey(self):
        if KnowledgeBase._valkey_client is not None:
            return
        import redis as _redis
        host     = os.getenv("VALKEY_HOST", "localhost")
        port     = int(os.getenv("VALKEY_PORT", 6379))
        password = os.getenv("VALKEY_PASSWORD") or None
        ssl      = os.getenv("VALKEY_SSL", "false").lower() == "true"
        KnowledgeBase._valkey_client = _redis.Redis(
            host=host, port=port, password=password,
            ssl=ssl, ssl_cert_reqs=None,
            decode_responses=False,
            socket_timeout=5, socket_connect_timeout=5
        )
        rag_logger.info(f"[VALKEY] Connected to {host}:{port} (ssl={ssl})")

    def _ensure_valkey_index(self):
        if KnowledgeBase._valkey_index_verified:
            return
        import redis as _redis
        from redis.commands.search.field import TagField, NumericField, VectorField
        from redis.commands.search.index_definition import IndexDefinition, IndexType
        r = KnowledgeBase._valkey_client
        try:
            r.ft(self.VALKEY_INDEX).info()
            rag_logger.info(f"[VALKEY] Reusing index: {self.VALKEY_INDEX}")
        except _redis.exceptions.ResponseError as e:
            if "not found" in str(e).lower():
                schema = (
                    TagField("doc_id"),
                    TagField("source_name"),
                    NumericField("chunk_index"),
                    VectorField("embedding", "HNSW", {
                        "TYPE": "FLOAT32",
                        "DIM": EMBEDDING_DIMENSIONS,
                        "DISTANCE_METRIC": "COSINE"
                    })
                )
                definition = IndexDefinition(prefix=[self.VALKEY_KEY_PREFIX], index_type=IndexType.HASH)
                r.ft(self.VALKEY_INDEX).create_index(schema, definition=definition)
                rag_logger.info(f"[VALKEY] Created index: {self.VALKEY_INDEX}")
            else:
                raise
        KnowledgeBase._valkey_index_verified = True

    @staticmethod
    def _escape_tag(value: str) -> str:
        special = set(r',.<>{}[]"\':;!@#$%^&*()+=~|/ ')
        return "".join(("\\" + ch if ch in special else ch) for ch in value)

    def _clean_text(self, text: str) -> str:
        if not text: return ""
        import re
        return re.sub(r'\s+', ' ', text).strip()

    def _get_chunks(self, text: str):
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + CHUNK_SIZE, text_len)
            if end < text_len:
                last_space = text.rfind(' ', start, end)
                if last_space != -1 and last_space > start + (CHUNK_SIZE // 2):
                    end = last_space
            chunk = text[start:end].strip()
            if len(chunk) > 30:
                chunks.append(chunk)
            start = end - CHUNK_OVERLAP
            if start < 0: start = 0
            if end >= text_len: break
        return chunks

    def build_index(self, text: str, source_name: str, doc_id: str = None) -> str:
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        self.clear(doc_id=doc_id)
        clean_text = self._clean_text(text)
        chunks = self._get_chunks(clean_text)
        self.add_documents(chunks, source_name=source_name, doc_id=doc_id)
        return doc_id

    def add_documents(self, chunks, source_name: str, doc_id: str = None):
        if not chunks:
            rag_logger.warning(f"No chunks to add for source: {source_name}")
            return
        if doc_id is None:
            doc_id = str(uuid.uuid4())

        rag_logger.info(f"[{self.backend.upper()}] Adding {len(chunks)} chunks — doc_id={doc_id} source='{source_name}'")
        add_start = time.time()

        vectors = indicator_model.encode(chunks).astype('float32')
        faiss.normalize_L2(vectors)
        rag_logger.info(f"Embeddings generated in {(time.time()-add_start)*1000:.2f}ms")

        if self.backend == "faiss":
            self.index.add(vectors)
            for chunk in chunks:
                self.metadata.append({"text": chunk, "source": source_name, "doc_id": doc_id})
            rag_logger.info(f"FAISS index size: {self.index.ntotal} vectors")
        else:
            r = KnowledgeBase._valkey_client
            pipeline = r.pipeline()
            for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                key = f"{self.VALKEY_KEY_PREFIX}{doc_id}:{i}"
                pipeline.hset(key, mapping={
                    "doc_id":      doc_id,
                    "source_name": source_name,
                    "chunk_index": i,
                    "text_chunk":  chunk,
                    "embedding":   np.array(vec, dtype=np.float32).tobytes()
                })
            pipeline.execute()
            rag_logger.info(f"[VALKEY] Stored {len(chunks)} hashes under lk:store:{doc_id}:*")

        rag_logger.info(f"add_documents done in {(time.time()-add_start)*1000:.2f}ms")

    def clear(self, doc_id: str = None):
        if self.backend == "faiss":
            if doc_id:
                self.metadata = [m for m in self.metadata if m.get("doc_id") != doc_id]
                self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
                if self.metadata:
                    vecs = indicator_model.encode([m["text"] for m in self.metadata]).astype('float32')
                    faiss.normalize_L2(vecs)
                    self.index.add(vecs)
                rag_logger.info(f"FAISS: removed doc_id={doc_id}, {len(self.metadata)} chunks remain")
            else:
                self.metadata = []
                self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
                rag_logger.info("FAISS index cleared")
        else:
            r = KnowledgeBase._valkey_client
            pattern = f"{self.VALKEY_KEY_PREFIX}{doc_id}:*" if doc_id else f"{self.VALKEY_KEY_PREFIX}*"
            cursor, deleted = 0, 0
            while True:
                cursor, keys = r.scan(cursor=cursor, match=pattern, count=500)
                if keys:
                    r.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            rag_logger.info(f"[VALKEY] Deleted {deleted} keys (pattern={pattern})")

    def list_documents(self):
        """Returns list of {source_name, chunk_count} grouped by category."""
        if self.backend == "faiss":
            docs = {}
            for m in self.metadata:
                sn = m.get("source", "unknown")
                if sn not in docs:
                    docs[sn] = {"source_name": sn, "chunk_count": 0}
                docs[sn]["chunk_count"] += 1
            return list(docs.values())
        else:
            r = KnowledgeBase._valkey_client
            prefix = self.VALKEY_KEY_PREFIX
            docs = {}
            for key in r.scan_iter(match=f"{prefix}*", count=500):
                source_raw = r.hget(key, "source_name")
                sn = source_raw.decode() if source_raw else "unknown"
                if sn not in docs:
                    docs[sn] = {"source_name": sn, "chunk_count": 0}
                docs[sn]["chunk_count"] += 1
            return list(docs.values())

    def search_rag(self, query: str, k: int = 5, source_name: str = None) -> str:
        rag_logger.info(f"[{self.backend.upper()}] RAG search: '{query[:100]}' source_name={source_name}")
        start_time = time.time()
        THRESHOLD = 0.25

        query_vector = indicator_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_vector)
        llm_results = []

        if self.backend == "faiss":
            if self.index.ntotal == 0:
                rag_logger.warning("FAISS index is empty")
                return "NO_INFORMATION_IN_KNOWLEDGE_BASE"
            D, I = self.index.search(query_vector, k)
            rag_logger.info(f"--- [RAG FAISS] TOP {k} MATCHES ---")
            for i, idx in enumerate(I[0]):
                if idx == -1 or idx >= len(self.metadata): continue
                meta = self.metadata[idx]
                if source_name and meta.get("source") != source_name: continue
                score = float(D[0][i])
                status = "✅ ABOVE" if score > THRESHOLD else "❌ BELOW"
                rag_logger.info(f"  Match {i+1:02d} | Score: {score:.4f} | {status} | {meta['text'][:80]}")
                if score > THRESHOLD and len(llm_results) < 3:
                    llm_results.append(f"[{meta['source']}]: {meta['text']}")
            rag_logger.info(f"--- [RAG FAISS] END ---")

        else:
            r = KnowledgeBase._valkey_client
            from redis.commands.search.query import Query as RQuery
            total_count = sum(1 for _ in r.scan_iter(match=f"{self.VALKEY_KEY_PREFIX}*", count=500))
            rag_logger.info(f"[VALKEY] {total_count} total chunk(s) in store")
            if total_count == 0:
                return "NO_INFORMATION_IN_KNOWLEDGE_BASE"
            vec_bytes = query_vector.astype('float32').tobytes()
            if source_name:
                q_str = f"(@source_name:{{{self._escape_tag(source_name)}}})=>[KNN {k} @embedding $vec AS score]"
            else:
                q_str = f"*=>[KNN {k} @embedding $vec AS score]"
            search_q = RQuery(q_str).return_fields("text_chunk", "source_name", "score").dialect(2)
            results = r.ft(self.VALKEY_INDEX).search(search_q, query_params={"vec": vec_bytes})
            rag_logger.info(f"--- [RAG VALKEY] TOP {k} MATCHES ---")
            for i, doc in enumerate(results.docs):
                score = float(doc.score) if hasattr(doc, 'score') else 0.0
                text  = doc.text_chunk  if hasattr(doc, 'text_chunk') else ""
                src   = doc.source_name if hasattr(doc, 'source_name') else ""
                status = "✅ ABOVE" if score > THRESHOLD else "❌ BELOW"
                rag_logger.info(f"  Match {i+1:02d} | Score: {score:.4f} | {status} | {text[:80]}")
                if score > THRESHOLD and len(llm_results) < 3:
                    llm_results.append(f"[{src}]: {text}")
            rag_logger.info(f"--- [RAG VALKEY] END ---")

        total_time = (time.time() - start_time) * 1000
        result_text = "\n\n---\n\n".join(llm_results) if llm_results else "No specific information found."
        rag_logger.info(f"[RAG] 🔎 DONE. Latency: {total_time:.2f}ms | Sent to LLM: {len(llm_results)}")
        return result_text


# Initialize global RAG instance
rag = KnowledgeBase()
rag_logger.info(f"RAG backend: {rag.backend}")

# Active category for RAG queries (set by UI dropdown)
active_source_name: str = None

# Active agent persona (set by UI selector)
active_persona_id: str = DEFAULT_PERSONA_ID

def analyze_transcript_content(transcript_text: str):
    """Performs LLM analysis on a transcript for handoff/summary."""
    prompt = f"""
    Analyze the following phone conversation transcript for a handoff to a human agent.
    
    TRANSCRIPT:
    {transcript_text}
    
    QUESTIONS:
    1. User Details: Fetch User details from transcript (Name, contact if mentioned).
    2. User Sentiment: Is the user angry or peeved or disappointed? If so, why?
    3. Actionable Requests: Does the transcript have any actionable that the user wants the agent to perform?
    
    Format the response using these exact headings with a '#' prefix (e.g., # User Details). 
    Use markdown for the content under each heading. Be concise but thorough.
    
    """
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in LLM analysis: {e}")
        return f"Analysis failed: {e}"

@app.post("/api/analyze-transcript")
async def manual_analyze(data: dict):
    """Allows manual analysis of a transcript via API."""
    transcript_text = data.get("transcript")
    if not transcript_text:
        return {"error": "No transcript provided"}
    
    analysis = analyze_transcript_content(transcript_text)
    return {"analysis": analysis}

HANDOFFS_FILE = "handoffs.json"

def save_handoff(handoff_data):
    handoffs = []
    if os.path.exists(HANDOFFS_FILE):
        try:
            with open(HANDOFFS_FILE, "r") as f:
                handoffs = json.load(f)
        except Exception as e:
            logger.error(f"Error loading handoffs: {e}")
            handoffs = []
    
    handoffs.append(handoff_data)
    try:
        with open(HANDOFFS_FILE, "w") as f:
            json.dump(handoffs, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving handoffs: {e}")

@app.get("/api/handoffs")
async def list_handoffs():
    """Returns a list of all handoffs with LLM analysis."""
    if not os.path.exists(HANDOFFS_FILE):
        return []
    try:
        with open(HANDOFFS_FILE, "r") as f:
            return json.load(f)
    except:
        return []


# --- API Endpoints ---

@app.get("/")
async def read_index():
    return FileResponse('frontend/dist/index.html')

@app.get("/token")
def get_token():
    at = api.AccessToken(
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
    ).with_identity("web-user").with_grants(
        api.VideoGrants(room_join=True, room="default")
    )

    return {
        "token": at.to_jwt(),
        "url": os.environ["TOKEN_URL"],
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), category: str = Form("general")):
    """Receives a file and category, indexes it under that category for scoped queries."""
    try:
        file_extension = os.path.splitext(file.filename)[1].lower()
        text_content = ""

        if file_extension == ".pdf":
            pdf_reader = PdfReader(file.file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_content += text + "\n"
        else:
            content = await file.read()
            text_content = content.decode("utf-8", errors="ignore")

        logger.info(f"🚀 Indexing '{file.filename}' under category='{category}' ({rag.backend} backend)...")
        doc_id = rag.build_index(text_content, source_name=category)
        logger.info(f"✅ Indexed '{file.filename}' under category='{category}' — doc_id={doc_id}")
        return {"status": "success", "filename": file.filename, "category": category, "doc_id": doc_id}

    except Exception as e:
        logger.error(f"Error handling upload: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/documents")
async def list_documents():
    """Returns all indexed categories with chunk counts and active selection."""
    try:
        docs = rag.list_documents()
        return {"documents": docs, "active_source_name": active_source_name}
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return {"documents": [], "active_source_name": None}

class ActiveDocRequest(BaseModel):
    source_name: str | None = None

@app.post("/api/set-active-doc")
async def set_active_doc(req: ActiveDocRequest):
    """Sets the active category used by the agent for RAG queries."""
    global active_source_name
    active_source_name = req.source_name
    logger.info(f"Active source_name set to: {active_source_name}")
    return {"status": "ok", "active_source_name": active_source_name}

@app.get("/api/personas")
async def get_personas():
    """Returns available agent personas and the currently active one."""
    return {"personas": list_personas(), "active_persona_id": active_persona_id}

class SetPersonaRequest(BaseModel):
    persona_id: str

@app.post("/api/set-persona")
async def set_persona(req: SetPersonaRequest):
    """Sets the active agent persona."""
    global active_persona_id
    persona = get_persona(req.persona_id)
    if not persona:
        return {"status": "error", "message": f"Unknown persona: {req.persona_id}"}
    active_persona_id = req.persona_id
    logger.info(f"Active persona set to: {active_persona_id}")
    return {"status": "ok", "active_persona_id": active_persona_id}

class OutboundCallRequest(BaseModel):
    phone_number: str

@app.post("/api/outbound-call")
async def outbound_call(req: OutboundCallRequest):
    """Initiate an outbound call to a phone number via Vobiz SIP."""
    phone = req.phone_number.strip()
    if not phone:
        return {"status": "error", "message": "Phone number is required"}
    # Ensure it starts with + for international format
    if not phone.startswith("+"):
        phone = "+91" + phone  # Default to India
    logger.info(f"Outbound call requested to: {phone}")
    result = await make_outbound_call(phone)
    return result

@app.post("/api/clear-db")
async def clear_db():
    """Clears all indexed documents from the vector store."""
    rag.clear()
    global active_source_name
    active_source_name = None
    logger.info("Vector store cleared")
    return {"status": "ok"}

# --- Session Browser Endpoints ---

@app.get("/api/sessions")
async def list_sessions():
    """Returns a list of all recorded sessions."""
    sessions_dir = "sessions"
    if not os.path.exists(sessions_dir):
        return []
    
    files = os.listdir(sessions_dir)
    wav_files = sorted([f for f in files if f.endswith(".wav")], reverse=True)
    
    results = []
    for wav in wav_files:
        base = wav[:-4]
        txt = base + ".txt"
        
        # Extract metadata from filename
        # session_20260224_130529_sip_devkey
        parts = wav.split('_')
        date_str = parts[1] if len(parts) > 1 else ""
        time_str = parts[2] if len(parts) > 2 else ""
        user_id = "_".join(parts[3:]) if len(parts) > 3 else "unknown"
        user_id = user_id.replace(".wav", "")

        # Format display time (IST)
        formatted_time = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str[:2]}:{time_str[2:4]}"

        results.append({
            "id": base,
            "filename": wav,
            "transcript_exists": os.path.exists(os.path.join(sessions_dir, txt)),
            "time": formatted_time,
            "user": user_id
        })
    
    return results

@app.get("/api/sessions/{session_id}/transcript")
async def get_transcript(session_id: str):
    """Parses and returns the transcript file content."""
    txt_path = os.path.join("sessions", f"{session_id}.txt")
    if not os.path.exists(txt_path):
        return {"error": "Transcript not found"}
    
    lines = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            # Format: [18:09:45] 👤 User   : What are your office hours?
            try:
                time_part = line[line.find("[")+1:line.find("]")]
                content = line[line.find("]")+2:].strip()
                role_icon = content[:1]
                role = "user" if "👤" in role_icon else "assistant"
                text = content[content.find(":")+1:].strip()
                
                lines.append({
                    "time": time_part,
                    "role": role,
                    "text": text
                })
            except:
                lines.append({"raw": line.strip()})
                
    return lines

@mcp.tool()
def query_knowledge_base(question: str, source_name: str = None) -> str:
    """Queries the vector database (RAG) to find an answer.
    Pass source_name to restrict search to a specific category (e.g. 'general', 'finance', 'policy').
    Falls back to the active category set by the UI."""
    start = time.perf_counter()
    effective_source = source_name or active_source_name
    result = rag.search_rag(question, source_name=effective_source)
    end = time.perf_counter()
    logger.info(f"[MCP] Tool Execution: {(end - start)*1000:.2f} ms | source_name={effective_source}")
    return f"Relevant Context:\n{result}"

@mcp.tool()
def get_appointment_info() -> str:
    """Get current appointment configuration like duration and break time."""
    try:
        from calendar_integration.constants import TASK_TYPES, DEFAULT_BUFFER_MINUTES
        duration = TASK_TYPES.get("appointment", {}).get("duration_minutes", 120)
        buffer = DEFAULT_BUFFER_MINUTES
    except (ImportError, ModuleNotFoundError):
        duration = 120
        buffer = 60
    
    info = f"Appointments are {duration // 60} hours long."
    if buffer > 0:
        info += f" There is a {buffer // 60}-hour break between appointments."
    info += " All appointments are stored in Google Calendar and managed in India Standard Time (IST)."
    return info

@mcp.tool()
def check_and_book_appointment(date_text: str) -> str:
    """
    Check availability for a given date or day string (e.g., 'tomorrow', 'next Monday', 'March 15th').
    If only a time is mentioned without a date, it defaults to tomorrow.
    Returns available time slots or suggests the next available date if the requested date is full.
    """
    manager = get_appointment_manager()
    
    # Parse the date
    dt = manager.parse_date_time(date_text)
    if not dt:
        return f"I couldn't understand the date '{date_text}'. Could you please specify it more clearly?"
    
    # Check if slots are available
    slots = manager.get_available_slots(dt, "appointment")
    
    if not slots:
        # Suggest next available
        next_slot = manager.find_next_available_slot("appointment", from_date=dt)
        if next_slot and next_slot.get("found"):
            resp = f"I'm sorry, but there are no available slots on {dt.strftime('%A, %B %d')}.\n"
            resp += f"The next available date is {next_slot['date_formatted']}.\n"
            resp += "Available slots for that day are:\n"
            for slot in next_slot['all_slots']:
                resp += f"- {slot['formatted']} (Start ISO: {slot['start']})\n"
            return resp
        else:
            return f"I'm sorry, I couldn't find any available slots starting from {dt.strftime('%B %d')}."

    # Return slots
    resp = f"Here are the available slots for {dt.strftime('%A, %B %d')}:\n"
    for slot in slots:
        resp += f"- {slot['formatted']} (Start ISO: {slot['start'].isoformat()})\n"
    return resp

@mcp.tool()
def schedule_appointment(start_time_iso: str, user_name: str, user_email: str, user_phone: str = None, notes: str = None) -> str:
    """
    Schedules an appointment at the specified start time (use the exact Start ISO string from check_and_book_appointment).
    Requires user's name and email. Phone and notes are optional.
    """
    manager = get_appointment_manager()
    
    # Convert start_time string to datetime
    try:
        if 'T' in start_time_iso:
            start_dt = datetime.fromisoformat(start_time_iso.replace('Z', '+00:00'))
        else:
            # Fallback if it's just a partial string
            start_dt = manager.parse_date_time(start_time_iso)
    except Exception as e:
        return f"Error: Invalid start time format. Please provide the exact ISO timestamp. {e}"
        
    if not start_dt:
         return f"Error: Could not parse start time '{start_time_iso}'."

    # Create the appointment
    result = manager.create_appointment(
        user_id=1, # Default user ID
        user_name=user_name,
        user_email=user_email,
        task_type="appointment",
        start_time=start_dt,
        notes=notes,
        phone=user_phone
    )
    
    if result.get("success"):
        return manager.format_appointment_confirmation(result["appointment"])
    else:
        error = result.get("error", "Unknown error")
        if "already passed" in error.lower() or "past time" in error.lower():
            # Suggest next available
            next_available = manager.find_next_available_slot("appointment", from_date=start_dt + timedelta(days=1))
            if next_available and next_available.get("found"):
                return f"Scheduling failed: {error}. The next available slot is on {next_available['date_formatted']} at {next_available['first_slot']['formatted']}."
        return f"Failed to schedule appointment: {error}"

@mcp.tool()
async def transfer_to_human(participant_identity: str, room_name: str, transcript_path: str = None) -> str:
    """
    Transfers the current SIP call to a human agent (+919390694802).
    Requires the participant_identity and room_name to identify the call.
    """
    logger.info(f"🚀 [TRANSFER] Requesting SIP transfer for {participant_identity} in room {room_name} to +919390694802")
    url = "http://localhost:7880"
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    
    lkapi = api.LiveKitAPI(url, api_key, api_secret)
    transfer_status = "unknown"
    try:
        await lkapi.sip.transfer_sip_participant(
            api.TransferSIPParticipantRequest(
                participant_identity=participant_identity,
                room_name=room_name,
                transfer_to="tel:+919390694802",
                play_dialtone=True
            )
        )
        logger.info(f"✅ [TRANSFER] SIP transfer initiated for {participant_identity}")
        transfer_status = "success"
    except Exception as e:
        # If it's a timeout/deadline error, it often means the referral was sent but confirmation was late
        if "deadline exceeded" in str(e).lower() or "408" in str(e):
            logger.warning(f"⚠️ [TRANSFER] SIP transfer request timed out, but proceeding with analysis: {e}")
            transfer_status = "potential_success_timeout"
        else:
            logger.error(f"❌ [TRANSFER] SIP transfer failed: {e}")
            transfer_status = f"failed: {e}"
    finally:
        await lkapi.aclose()
        
    # Proceed with LLM Analysis regardless of transfer API result (since user says it often works anyway)
    if transcript_path and os.path.exists(transcript_path):
        try:
            logger.info(f"🧠 [ANALYSIS] Reading transcript from {transcript_path}...")
            # Give the agent a second to flush any final messages to the file
            await asyncio.sleep(1) 
            with open(transcript_path, "r") as f:
                transcript_text = f.read()
            
            analysis = analyze_transcript_content(transcript_text)
            
            # Save to handoffs
            handoff_info = {
                "id": str(time.time()),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "from_number": participant_identity.replace("sip_", ""),
                "to_number": "+919390694802",
                "analysis": analysis,
                "transcript_path": transcript_path,
                "transfer_api_status": transfer_status
            }
            save_handoff(handoff_info)
            logger.info("✅ [ANALYSIS] Handoff analysis completed and saved to dashboard.")
            
        except Exception as ae:
            logger.error(f"❌ [ANALYSIS] Failed to process transcript file: {ae}")

    if "success" in transfer_status:
        return "Transfer initiated successfully."
    else:
        return f"Transfer triggered, but API returned: {transfer_status}"

# Mount MCP on FastAPI
mcp_sse = mcp.sse_app()
app.mount("/mcp", mcp_sse)

# Mount sessions directory to serve WAV files
if not os.path.exists("sessions"):
    os.makedirs("sessions")
app.mount("/sessions", StaticFiles(directory="sessions"), name="sessions")

# Mount static files (at the end to not catch everything)
app.mount("/", StaticFiles(directory="frontend/dist"), name="static")

if __name__ == "__main__":
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    # Run on specified port
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)
    # Actually, uvicorn's log_config handles its internal logs. 
    # Let's just point uvicorn to our log file if we want it there too.
    # For now, let's just make sure app logs go to app.log.

