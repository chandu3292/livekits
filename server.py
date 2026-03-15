import os
import time
import json
import uuid
import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from google import genai
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from livekit import api
from agent_personas import list_personas, get_persona, DEFAULT_PERSONA_ID
from sentence_transformers import SentenceTransformer
from ocr import process_file as ocr_process_file
import logging
from datetime import datetime, timedelta, timezone
from calendar_integration import get_appointment_manager
import warnings

# Suppress the legacy cryptography warning from pypdf/other libs
warnings.filterwarnings("ignore", category=UserWarning, module='cryptography')
warnings.filterwarnings("ignore", message=".*ARC4.*")

# Configure Loggers
logger = logging.getLogger("server")
rag_logger = logging.getLogger("rag")

# Setup FastAPI + MCP
app = FastAPI()
mcp = FastMCP("Vector RAG")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables from .env
load_dotenv()

# Configuration
# Indic SBERT Model
indicator_model = SentenceTransformer("l3cube-pune/indic-sentence-similarity-sbert")
EMBEDDING_DIMENSIONS = indicator_model.get_sentence_embedding_dimension()
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
CHAT_MODEL = os.getenv("CHAT_MODEL", GEMINI_MODEL)

google_client = None
if os.getenv("GOOGLE_API_KEY"):
    google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# OCR supported extensions
OCR_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".docx", ".doc"}
ALL_UPLOAD_EXTENSIONS = {".pdf", ".txt", ".md", ".py", ".json"} | OCR_EXTENSIONS


class KnowledgeBase:
    """FAISS-based vector knowledge base for RAG."""

    def __init__(self):
        self.metadata = []  # list of {text, source, doc_id}
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
        rag_logger.info("KnowledgeBase initialised (backend=faiss)")

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

        rag_logger.info(f"[FAISS] Adding {len(chunks)} chunks — doc_id={doc_id} source='{source_name}'")
        add_start = time.time()

        vectors = indicator_model.encode(chunks).astype('float32')
        faiss.normalize_L2(vectors)
        rag_logger.info(f"Embeddings generated in {(time.time()-add_start)*1000:.2f}ms")

        self.index.add(vectors)
        for chunk in chunks:
            self.metadata.append({"text": chunk, "source": source_name, "doc_id": doc_id})
        rag_logger.info(f"FAISS index size: {self.index.ntotal} vectors")
        rag_logger.info(f"add_documents done in {(time.time()-add_start)*1000:.2f}ms")

    def clear(self, doc_id: str = None):
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

    def list_documents(self):
        """Returns list of {source_name, chunk_count} grouped by category."""
        docs = {}
        for m in self.metadata:
            sn = m.get("source", "unknown")
            if sn not in docs:
                docs[sn] = {"source_name": sn, "chunk_count": 0}
            docs[sn]["chunk_count"] += 1
        return list(docs.values())

    def search_rag(self, query: str, k: int = 5, source_name: str = None) -> str:
        rag_logger.info(f"[FAISS] RAG search: '{query[:100]}' source_name={source_name}")
        start_time = time.time()
        THRESHOLD = 0.25

        if self.index.ntotal == 0:
            rag_logger.warning("FAISS index is empty")
            return "NO_INFORMATION_IN_KNOWLEDGE_BASE"

        query_vector = indicator_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_vector)
        llm_results = []

        D, I = self.index.search(query_vector, k)
        for i, idx in enumerate(I[0]):
            if idx == -1 or idx >= len(self.metadata): continue
            meta = self.metadata[idx]
            if source_name and meta.get("source") != source_name: continue
            score = float(D[0][i])
            if score > THRESHOLD and len(llm_results) < 3:
                llm_results.append(f"[{meta['source']}]: {meta['text']}")

        total_time = (time.time() - start_time) * 1000
        result_text = "\n\n---\n\n".join(llm_results) if llm_results else "No specific information found."
        rag_logger.info(f"[RAG] DONE. Latency: {total_time:.2f}ms | Sent to LLM: {len(llm_results)}")
        return result_text


# Initialize global RAG instance
rag = KnowledgeBase()

# Active category for RAG queries (set by UI dropdown)
active_source_name: str = None

# Active agent persona (set by UI selector)
active_persona_id: str = DEFAULT_PERSONA_ID

# Chat conversation history (per-session, in-memory)
chat_histories: dict = {}


# --- API Endpoints ---

@app.get("/")
async def read_index():
    return FileResponse('frontend/dist/index.html')

@app.get("/token")
def get_token(persona: str = None):
    import uuid
    pid = persona or active_persona_id
    p = get_persona(pid)
    room_name = f"room-{uuid.uuid4().hex[:8]}"
    metadata = json.dumps({"persona_id": pid, "language": p["language"] if p else "en"})
    at = api.AccessToken(
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
    ).with_identity("web-user").with_grants(
        api.VideoGrants(room_join=True, room=room_name)
    ).with_metadata(metadata)
    return {
        "token": at.to_jwt(),
        "url": os.getenv("LIVEKIT_URL", "ws://localhost:7880"),
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), category: str = Form("general")):
    """Receives a file and category, indexes it under that category.
    Uses OCR (pytesseract) for images and scanned PDFs.
    Supports: PDF, DOCX, MD, TXT, PNG, JPG, TIFF, BMP, WEBP
    """
    try:
        file_extension = os.path.splitext(file.filename)[1].lower()
        file_bytes = await file.read()
        text_content = ""
        ocr_metadata = {}

        if file_extension in OCR_EXTENSIONS or file_extension == ".pdf":
            result = ocr_process_file(file_bytes, file.filename)
            text_content = result.get("text", "")
            ocr_metadata = {
                "tables_found": len(result.get("tables", [])),
                "key_value_pairs": len(result.get("key_value_pairs", {})),
                "file_type": result.get("file_type", ""),
                "pages": result.get("pages", 0),
            }
            logger.info(f"OCR processed '{file.filename}': {len(text_content)} chars, "
                        f"{ocr_metadata['tables_found']} tables, "
                        f"{ocr_metadata['key_value_pairs']} KV pairs")
        else:
            text_content = file_bytes.decode("utf-8", errors="ignore")

        if not text_content.strip():
            return {"status": "error", "message": "No text could be extracted from the file"}

        logger.info(f"Indexing '{file.filename}' under category='{category}'...")
        doc_id = rag.build_index(text_content, source_name=category)
        logger.info(f"Indexed '{file.filename}' under category='{category}' - doc_id={doc_id}")

        response = {
            "status": "success",
            "filename": file.filename,
            "category": category,
            "doc_id": doc_id,
            "chars_extracted": len(text_content),
        }
        if ocr_metadata:
            response["ocr"] = ocr_metadata
        return response

    except Exception as e:
        logger.error(f"Error handling upload: {e}")
        return {"status": "error", "message": str(e)}


@app.post("/api/ocr")
async def ocr_extract(file: UploadFile = File(...)):
    """Standalone OCR endpoint - extracts text and structured data without indexing."""
    try:
        file_bytes = await file.read()
        result = ocr_process_file(file_bytes, file.filename)
        return {
            "status": "success",
            "filename": file.filename,
            "text": result.get("text", ""),
            "tables": result.get("tables", []),
            "key_value_pairs": result.get("key_value_pairs", {}),
            "file_type": result.get("file_type", ""),
            "pages": result.get("pages", 0),
        }
    except Exception as e:
        logger.error(f"OCR extraction error: {e}")
        return {"status": "error", "message": str(e)}


# --- Chat Endpoint (Gemini-powered) ---

class ChatMessage(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/api/chat")
async def chat(req: ChatMessage):
    """Chat endpoint using Gemini model with RAG context."""
    if not google_client:
        return {"status": "error", "message": "Gemini API key not configured"}

    user_message = req.message.strip()
    if not user_message:
        return {"status": "error", "message": "Empty message"}

    # Get or create conversation history
    if req.session_id not in chat_histories:
        chat_histories[req.session_id] = []
    history = chat_histories[req.session_id]

    # Search RAG for context
    rag_context = ""
    effective_source = active_source_name
    try:
        rag_result = rag.search_rag(user_message, source_name=effective_source)
        if rag_result and rag_result != "NO_INFORMATION_IN_KNOWLEDGE_BASE" and rag_result != "No specific information found.":
            rag_context = rag_result
    except Exception as e:
        logger.warning(f"RAG search failed during chat: {e}")

    # Build the prompt with system context
    system_prompt = (
        "You are a helpful AI assistant. You work at DocQuery. "
        "Be concise, friendly, and helpful. Use the provided context to answer questions accurately. "
        "If the context doesn't contain relevant information, use your general knowledge but mention that. "
        "IMPORTANT: Detect the language of the user's message and ALWAYS respond in the SAME language. "
        "For example, if the user writes in Telugu, respond entirely in Telugu. "
        "If the user writes in Hindi, respond in Hindi. If in English, respond in English. "
        "Never mix languages — reply fully in the user's language."
    )

    if rag_context:
        system_prompt += f"\n\nRelevant context from knowledge base:\n{rag_context}"

    # Build messages for Gemini
    contents = [{"role": "user", "parts": [{"text": system_prompt + "\n\nRespond to the following conversation."}]}]
    contents.append({"role": "model", "parts": [{"text": "Understood. I'm ready to help."}]})

    # Add conversation history (last 20 messages max)
    for msg in history[-20:]:
        contents.append({"role": msg["role"], "parts": [{"text": msg["text"]}]})

    # Add current user message
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    try:
        response = google_client.models.generate_content(
            model=CHAT_MODEL,
            contents=contents,
        )
        assistant_text = response.text

        # Save to history
        history.append({"role": "user", "text": user_message})
        history.append({"role": "model", "text": assistant_text})

        # Trim history to prevent memory bloat
        if len(history) > 100:
            chat_histories[req.session_id] = history[-60:]

        return {
            "status": "success",
            "response": assistant_text,
            "has_context": bool(rag_context),
        }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/api/chat/clear")
async def clear_chat(data: dict = {}):
    """Clear chat history for a session."""
    session_id = data.get("session_id", "default")
    chat_histories.pop(session_id, None)
    return {"status": "ok"}


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
        parts = wav.split('_')
        date_str = parts[1] if len(parts) > 1 else ""
        time_str = parts[2] if len(parts) > 2 else ""
        user_id = "_".join(parts[3:]) if len(parts) > 3 else "unknown"
        user_id = user_id.replace(".wav", "")
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
            try:
                time_part = line[line.find("[")+1:line.find("]")]
                content = line[line.find("]")+2:].strip()
                role = "user" if "User" in content[:10] else "assistant"
                text = content[content.find(":")+1:].strip()
                lines.append({"time": time_part, "role": role, "text": text})
            except:
                lines.append({"raw": line.strip()})

    return lines


# --- MCP Tools (used by voice agent) ---

@mcp.tool()
def query_knowledge_base(question: str, source_name: str = None) -> str:
    """Queries the vector database (RAG) to find an answer."""
    effective_source = source_name or active_source_name
    result = rag.search_rag(question, source_name=effective_source)
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
    """Check availability for a given date or day string."""
    manager = get_appointment_manager()
    dt = manager.parse_date_time(date_text)
    if not dt:
        return f"I couldn't understand the date '{date_text}'. Could you please specify it more clearly?"

    slots = manager.get_available_slots(dt, "appointment")
    if not slots:
        next_slot = manager.find_next_available_slot("appointment", from_date=dt)
        if next_slot and next_slot.get("found"):
            resp = f"No available slots on {dt.strftime('%A, %B %d')}.\n"
            resp += f"Next available: {next_slot['date_formatted']}.\n"
            for slot in next_slot['all_slots']:
                resp += f"- {slot['formatted']} (Start ISO: {slot['start']})\n"
            return resp
        return f"No available slots starting from {dt.strftime('%B %d')}."

    resp = f"Available slots for {dt.strftime('%A, %B %d')}:\n"
    for slot in slots:
        resp += f"- {slot['formatted']} (Start ISO: {slot['start'].isoformat()})\n"
    return resp

@mcp.tool()
def schedule_appointment(start_time_iso: str, user_name: str, user_email: str, user_phone: str = None, notes: str = None) -> str:
    """Schedules an appointment at the specified start time."""
    manager = get_appointment_manager()
    try:
        if 'T' in start_time_iso:
            start_dt = datetime.fromisoformat(start_time_iso.replace('Z', '+00:00'))
        else:
            start_dt = manager.parse_date_time(start_time_iso)
    except Exception as e:
        return f"Error: Invalid start time format. {e}"

    if not start_dt:
        return f"Error: Could not parse start time '{start_time_iso}'."

    result = manager.create_appointment(
        user_id=1, user_name=user_name, user_email=user_email,
        task_type="appointment", start_time=start_dt, notes=notes, phone=user_phone
    )

    if result.get("success"):
        return manager.format_appointment_confirmation(result["appointment"])
    else:
        error = result.get("error", "Unknown error")
        if "already passed" in error.lower() or "past time" in error.lower():
            next_available = manager.find_next_available_slot("appointment", from_date=start_dt + timedelta(days=1))
            if next_available and next_available.get("found"):
                return f"Scheduling failed: {error}. Next available: {next_available['date_formatted']} at {next_available['first_slot']['formatted']}."
        return f"Failed to schedule appointment: {error}"


# Mount MCP on FastAPI
mcp_sse = mcp.sse_app()
app.mount("/mcp", mcp_sse)

# Mount sessions directory
if not os.path.exists("sessions"):
    os.makedirs("sessions")
app.mount("/sessions", StaticFiles(directory="sessions"), name="sessions")

# Mount static files (last, to not catch API routes)
app.mount("/", StaticFiles(directory="frontend/dist"), name="static")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_config=None)
