import time
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

# Initialize MCP
mcp = FastMCP("Vector RAG 🧠")

# Configuration
KNOWLEDGE_FILE = "shared_knowledge.txt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Small, fast, accurate
CHUNK_SIZE = 500 # Characters per chunk
CHUNK_OVERLAP = 50

class FaissRAG:
    def __init__(self, filepath):
        self.filepath = filepath
        self.last_mtime = 0
        self.chunks = []
        self.index = None
        
        print("[RAG] Loading Embedding Model...")
        # Load once at startup
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print("[RAG] Model Loaded.")

    def _read_and_chunk(self):
        """Reads file and splits it into overlapping chunks."""
        if not os.path.exists(self.filepath):
            return []
            
        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # Simple sliding window chunking
        chunks = []
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk = text[i:i + CHUNK_SIZE].strip()
            if len(chunk) > 20: # Filter tiny chunks
                chunks.append(chunk)
        return chunks

    def _refresh_index_if_needed(self):
        """Checks if file changed. If so, re-embeds and rebuilds FAISS index."""
        if not os.path.exists(self.filepath):
            return

        current_mtime = os.path.getmtime(self.filepath)
        
        # If file hasn't changed and we have an index, do nothing
        if self.index is not None and current_mtime == self.last_mtime:
            return

        print(f"[RAG] 🔄 File changed. Rebuilding vector index...")
        start_time = time.time()

        # 1. Read & Chunk
        self.chunks = self._read_and_chunk()
        if not self.chunks:
            print("[RAG] File is empty.")
            return

        # 2. Embed (Batch processing is faster)
        # normalize_embeddings=True is important for cosine similarity (dot product)
        embeddings = self.model.encode(self.chunks, normalize_embeddings=True)

        # 3. Build FAISS Index
        # We use IndexFlatIP (Inner Product) which is identical to Cosine Similarity for normalized vectors
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

        # Update cache trackers
        self.last_mtime = current_mtime
        print(f"[RAG] ✅ Index built with {len(self.chunks)} chunks in {time.time() - start_time:.2f}s")

    def search(self, query, top_k=3):
        # Ensure index is up to date
        self._refresh_index_if_needed()

        if self.index is None or self.index.ntotal == 0:
            return "Knowledge base is empty. Please upload a document."

        # 1. Embed Query
        query_vector = self.model.encode([query], normalize_embeddings=True)

        # 2. Search FAISS
        # D = distances (scores), I = indices of closest chunks
        D, I = self.index.search(np.array(query_vector).astype('float32'), k=top_k)

        # 3. Retrieve Text
        results = []
        for idx in I[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        return "\n---\n".join(results)

# Initialize RAG engine
rag = FaissRAG(KNOWLEDGE_FILE)

@mcp.tool()
def query_knowledge_base(question: str) -> str:
    """
    Queries the vector database (RAG) to find an answer from the uploaded document.
    """
    start = time.perf_counter()
    
    try:
        context = rag.search(question)
        result = f"Relevant Context from Document:\n{context}"
    except Exception as e:
        result = f"Error searching vector DB: {str(e)}"

    end = time.perf_counter()
    # This latency is now mostly just the query embedding time (typically <50ms)
    print(f"[MCP] Vector search latency: {(end - start)*1000:.1f} ms")

    return result

if __name__ == "__main__":
    mcp.run(transport="sse")