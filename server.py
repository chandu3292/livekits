import os
import time
import pickle
import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

# 1. Setup FastAPI wrapper
app = FastAPI()
mcp = FastMCP("Vector RAG 🧠")

# Load environment variables from .env
load_dotenv()

# Configuration
KNOWLEDGE_FILE = "shared_knowledge.txt"
INDEX_FILE = "vector_store.index"
META_FILE = "vector_store.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

def generate_answer(question: str, context: str) -> str:
    if not context or context.strip() == "Knowledge base is empty. Please upload a document.":
        return "Knowledge base is empty. Please upload a document."

    system_prompt = (
        "You are a helpful assistant. Use ONLY the provided context to answer. "
        "If the answer is not in the context, say you don't know. "
        "Keep the answer concise."
    )

    user_prompt = f"Question: {question}\n\nContext:\n{context}"

    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.output_text.strip() if response.output_text else "No answer returned."

class FaissRAG:
    def __init__(self, filepath):
        self.filepath = filepath
        self.chunks = []
        self.index = None
        
        print("[RAG] Loading Embedding Model...")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Load existing index on startup if available
        if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
            self.load_index()
        else:
            print("[RAG] No existing index found. Waiting for upload.")

    def load_index(self):
        """Quickly loads the index from disk."""
        try:
            print("[RAG] 💾 Loading index from disk...")
            self.index = faiss.read_index(INDEX_FILE)
            with open(META_FILE, "rb") as f:
                self.chunks = pickle.load(f)
            print(f"[RAG] ✅ Loaded {len(self.chunks)} chunks.")
        except Exception as e:
            print(f"[RAG] ⚠️ Load failed, rebuilding: {e}")
            self.build_index()

    def build_index(self):
        """Heavy operation: Reads file -> Embeds -> Saves."""
        print(f"[RAG] 🔄 Starting Index Rebuild...")
        start = time.perf_counter()

        if not os.path.exists(self.filepath):
            print("[RAG] ❌ No knowledge file found at", self.filepath)
            return

        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # 1. Chunking
        new_chunks = []
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[i:i + CHUNK_SIZE].strip()
            if len(chunk) > 20:
                new_chunks.append(chunk)
        
        if not new_chunks:
            print("[RAG] File was empty or too short.")
            return

        # 2. Embedding
        print(f"[RAG] 🧠 Embedding {len(new_chunks)} chunks...")
        t_embed_start = time.perf_counter()
        embeddings = self.model.encode(new_chunks, normalize_embeddings=True)
        print(f"[RAG] ⚡ Embeddings done in {(time.perf_counter() - t_embed_start):.2f}s")

        # 3. Indexing (HNSW)
        dimension = embeddings.shape[1]
        new_index = faiss.IndexHNSWFlat(dimension, 32) # 32 links per node
        new_index.hnsw.efConstruction = 40
        new_index.add(np.array(embeddings).astype('float32'))

        # 4. Atomic Swap
        self.index = new_index
        self.chunks = new_chunks

        # 5. Persistence
        faiss.write_index(self.index, INDEX_FILE)
        with open(META_FILE, "wb") as f:
            pickle.dump(self.chunks, f)

        print(f"[RAG] ✅ Index Updated & Saved in {time.perf_counter() - start:.2f}s")

    def search(self, query, top_k=3):
        if self.index is None or self.index.ntotal == 0:
            return "Knowledge base is empty. Please upload a document."
        
        t0 = time.perf_counter()
        query_vector = self.model.encode([query], normalize_embeddings=True)
        
        # Increase search depth for accuracy
        self.index.hnsw.efSearch = 64
        
        D, I = self.index.search(np.array(query_vector).astype('float32'), k=top_k)
        
        results = []
        for idx in I[0]:
            if idx != -1 and idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        latency = (time.perf_counter() - t0) * 1000
        print(f"[RAG] 🔎 Search Latency: {latency:.2f} ms")
        return "\n---\n".join(results)

# Initialize RAG Logic
rag = FaissRAG(KNOWLEDGE_FILE)

# Print top 3 chunks
if rag.chunks:
    print("\n" + "="*80)
    print("TOP 3 CHUNKS FROM VECTOR STORE")
    print("="*80)
    for i, chunk in enumerate(rag.chunks[:3], 1):
        print(f"\n--- CHUNK {i} ---\n{chunk}\n")
    print("="*80 + "\n")

# --- API Endpoints ---

@app.post("/trigger-update")
async def trigger_update():
    """Called by web_server.py after upload to force a rebuild."""
    rag.build_index()
    return {"status": "success", "chunks": len(rag.chunks)}

@app.post("/query")
async def query_rag(payload: QueryRequest):
    """HTTP endpoint for text-based querying."""
    context = rag.search(payload.question, top_k=payload.top_k)
    answer = generate_answer(payload.question, context)
    return {"answer": answer, "context": context}

@mcp.tool()
def query_knowledge_base(question: str) -> str:
    """Queries the vector database (RAG) to find an answer."""
    start = time.perf_counter()
    result = rag.search(question)
    end = time.perf_counter()
    print(f"[MCP] ⏱️ Tool Execution: {(end - start)*1000:.2f} ms")
    return f"Relevant Context:\n{result}"

# Mount MCP on FastAPI
mcp_sse = mcp.sse_app()
app.mount("/mcp", mcp_sse)

if __name__ == "__main__":
    # Run on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)