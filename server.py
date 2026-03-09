import os
import time
import pickle
import json
import asyncio
import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from google import genai
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from livekit import api
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
    def __init__(self, embedding_model_type='openai'):
        self.embedding_model_type = embedding_model_type
        self.metadata = []
        # Use IndexFlatIP for cosine similarity
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
        self.model = None # Placeholder for sentence_transformer if needed
        rag_logger.info(f"KnowledgeBase initialized (Type: {embedding_model_type})")

    def _clean_text(self, text: str) -> str:
        """Removes excessive whitespace and standardizes text for better embedding."""
        if not text: return ""
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _get_chunks(self, text: str):
        """Standard recursive-style chunking."""
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

    def clear(self):
        """Reset the knowledge base."""
        self.metadata = []
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
        rag_logger.info("KnowledgeBase cleared")

    def add_documents(self, chunks, source_name):
        """
        Add documents to the knowledge base by creating embeddings and indexing them.
        """
        if not chunks:
            rag_logger.warning(f"No chunks to add for source: {source_name}")
            return
        
        rag_logger.info(f"Starting document addition for '{source_name}' with {len(chunks)} chunks")
        add_start = time.time()
        
        # Using SentenceTransformer for Indic languages
        vectors = indicator_model.encode(chunks).astype('float32')
        
        # Switch to Cosine Similarity by normalizing vectors
        faiss.normalize_L2(vectors)
        
        embed_time = (time.time() - add_start) * 1000
        rag_logger.info(f"Embedding generation completed in {embed_time:.2f}ms for {len(chunks)} chunks")
        
        index_start = time.time()
        self.index.add(vectors)
        index_time = (time.time() - index_start) * 1000
        rag_logger.info(f"Index update completed in {index_time:.2f}ms")
        
        for chunk in chunks:
            self.metadata.append({"text": chunk, "source": source_name})
        
        total_add_time = (time.time() - add_start) * 1000
        rag_logger.info(f"Document addition completed in {total_add_time:.2f}ms - Total index size: {self.index.ntotal} vectors")

    def build_index(self, text, source_name):
        """High-level method to update the single document knowledge base."""
        self.clear()
        clean_text = self._clean_text(text)
        chunks = self._get_chunks(clean_text)
        self.add_documents(chunks, source_name)

    def search_rag(self, query, k=5):
        """
        Search the knowledge base using RAG (Retrieval-Augmented Generation).
        Logs top 30 matches and returns top 3 for LLM.
        """
        rag_logger.info(f"Starting RAG search for query: {query[:100]}..." if len(query) > 100 else f"Starting RAG search for query: {query}")
        start_time = time.time()
        
        # Check if index is empty
        if self.index is None or self.index.ntotal == 0:
            rag_logger.warning(f"Knowledge base is empty, returning NO_INFORMATION response")
            return "NO_INFORMATION_IN_KNOWLEDGE_BASE"
        
        # Step 1: Encode query
        encode_start = time.time()
        query_vector = indicator_model.encode([query]).astype('float32')
        # Normalize query vector for Cosine Similarity
        faiss.normalize_L2(query_vector)
        
        encode_time = (time.time() - encode_start) * 1000
        
        # Step 2: Search FAISS index for top 30
        search_start = time.time()
        D, I = self.index.search(query_vector, k)
        search_time = (time.time() - search_start) * 1000
        rag_logger.info(f"FAISS search (k={k}) completed in {search_time:.2f}ms")

        # Step 3: Log top 30 and pick top 3 for LLM
        THRESHOLD = 0.25
        llm_results = []
        
        rag_logger.info(f"--- [RAG] TOP {k} MATCHES FOR DEBUGGING ---")
        for i, idx in enumerate(I[0]):
            if idx == -1 or idx >= len(self.metadata):
                continue
                
            score = float(D[0][i])
            content = self.metadata[idx]['text']
            status = "✅ ABOVE THRESHOLD" if score > THRESHOLD else "❌ BELOW THRESHOLD"
            
            # Log ALL matches
            rag_logger.info(f"Match {i+1:02d} | Score: {score:.4f} | {status} | {content}")

            # Collect top 3 for LLM
            if score > THRESHOLD and len(llm_results) < 3:
                llm_results.append(f"[{self.metadata[idx]['source']}]: {self.metadata[idx]['text']}")
        
        rag_logger.info(f"--- [RAG] END TOP {k} ---")
        
        total_time = (time.time() - start_time) * 1000
        result_text = "\n\n---\n\n".join(llm_results) if llm_results else "No specific information found."
        
        rag_logger.info(f"[RAG] 🔎 DONE. Latency: {total_time:.2f} ms | Sent to LLM: {len(llm_results)}")
        
        return result_text


# Initialize global RAG instance
rag = KnowledgeBase()

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
async def upload_file(file: UploadFile = File(...)):
    """Receives a file and updates the singleton RAG index."""
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
        
        logger.info(f"🚀 Updating RAG Index for: {file.filename}...")
        rag.build_index(text_content, file.filename)

        return {"status": "success", "filename": file.filename}

    except Exception as e:
        logger.error(f"Error handling upload: {e}")
        return {"status": "error", "message": str(e)}

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
def query_knowledge_base(question: str) -> str:
    """Queries the vector database (RAG) to find an answer."""
    start = time.perf_counter()
    result = rag.search_rag(question)
    end = time.perf_counter()
    logger.info(f"[MCP] Tool Execution: {(end - start)*1000:.2f} ms")
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

