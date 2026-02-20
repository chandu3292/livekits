import os
import time
import pickle
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
import logging
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
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 512
CHUNK_SIZE = 1000  # Slightly smaller for better precision
CHUNK_OVERLAP = 100 # Larger overlap for better context
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
        
        # Using the global openai_client instead of 'client' from user snippet
        resp = openai_client.embeddings.create(input=chunks, model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS)
        vectors = np.array([item.embedding for item in resp.data]).astype('float32')
        
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

    def search_rag(self, query, k=116):
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
        if self.embedding_model_type == 'sentence_transformer' and self.model:
            query_vector = self.model.encode([query], normalize_embeddings=True).astype("float32")
        else:
            resp = openai_client.embeddings.create(input=[query], model=EMBEDDING_MODEL, dimensions=EMBEDDING_DIMENSIONS)
            query_vector = np.array([resp.data[0].embedding]).astype('float32')
            # Normalize query vector for Cosine Similarity
            faiss.normalize_L2(query_vector)
        
        encode_time = (time.time() - encode_start) * 1000
        
        # Step 2: Search FAISS index for top 30
        search_start = time.time()
        D, I = self.index.search(query_vector, k)
        search_time = (time.time() - search_start) * 1000
        rag_logger.info(f"FAISS search (k={k}) completed in {search_time:.2f}ms")

        # Step 3: Log top 30 and pick top 3 for LLM
        THRESHOLD = 0.2
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
        "url": os.environ["LIVEKIT_URL"],
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

@mcp.tool()
def query_knowledge_base(question: str) -> str:
    """Queries the vector database (RAG) to find an answer."""
    start = time.perf_counter()
    result = rag.search_rag(question)
    end = time.perf_counter()
    logger.info(f"[MCP] Tool Execution: {(end - start)*1000:.2f} ms")
    return f"Relevant Context:\n{result}"

# Mount MCP on FastAPI
mcp_sse = mcp.sse_app()
app.mount("/mcp", mcp_sse)

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

