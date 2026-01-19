# LiveKit Voice Agent with RAG

A voice-based AI agent system built with LiveKit that combines real-time voice communication with retrieval-augmented generation (RAG) capabilities.

## Features

- **Voice Communication**: Real-time voice interaction using LiveKit
- **RAG System**: Vector-based document retrieval using FAISS and sentence transformers
- **Document Upload**: Support for PDF and text file uploads
- **MCP Integration**: Model Context Protocol server for tool execution
- **Multiple Agent Implementations**: Both standard and MCP-enhanced agent variants

## Project Structure

```
├── agent.py              # Standard LiveKit voice agent with weather tool
├── mcp-agent.py          # Agent with MCP integration and RAG capabilities
├── server.py             # FAISS-based RAG server with MCP endpoints
├── web_server.py         # FastAPI server for token generation and file uploads
├── token_server.py       # Separate token generation service
├── shared_knowledge.txt  # Knowledge base file (populated via uploads)
├── index.html            # Frontend for web interface
├── requirements.txt      # Python dependencies
└── README.md            # This file
```


```

2. Create a Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your credentials:
```env
LIVEKIT_URL=<your-livekit-url>
LIVEKIT_API_KEY=<your-api-key>
LIVEKIT_API_SECRET=<your-api-secret>
OPENAI_API_KEY=<your-openai-key>
```

## Configuration

### Environment Variables

- `LIVEKIT_URL`: URL of your LiveKit server
- `LIVEKIT_API_KEY`: LiveKit API key
- `LIVEKIT_API_SECRET`: LiveKit API secret
- `OPENAI_API_KEY`: OpenAI API key for LLM inference

### RAG Configuration (server.py)

- `EMBEDDING_MODEL`: Sentence transformer model (default: "all-MiniLM-L6-v2")
- `CHUNK_SIZE`: Characters per knowledge chunk (default: 500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)

## Usage

### Running the Web Server

```bash
python web_server.py
```

Starts FastAPI server on `http://localhost:8000`

Endpoints:
- `GET /token` - Get LiveKit access token
- `POST /upload` - Upload PDF or text document

### Running the MCP Agent

```bash
python mcp-agent.py dev
```

Starts the voice agent with RAG capabilities via MCP.



### Running the RAG Server

```bash
python server.py
```

Starts the MCP server with FAISS vector index and document retrieval.

## How It Works

1. **Document Upload**: Users upload PDF or text files via the web interface
2. **Text Extraction**: PDFs are parsed and text is extracted
3. **Chunking**: Documents are split into overlapping chunks for better retrieval
4. **Embedding**: Chunks are converted to vector embeddings using sentence transformers
5. **Indexing**: Vectors are indexed in FAISS for fast similarity search
6. **Query**: When the agent receives a user query, it searches the knowledge base and passes relevant context to the LLM

## Technologies Used

- **LiveKit**: Real-time voice/video communication
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embedding models
- **FastAPI**: Web server framework
- **OpenAI GPT-4**: Language model
- **Deepgram Nova-3**: Speech-to-text
- **Cartesia Sonic-3**: Text-to-speech
- **MCP (Model Context Protocol)**: Tool execution framework

## Dependencies

See `requirements.txt` for complete list of Python packages.

## License

[Add your license here]

## Support

For issues and questions, please refer to the [LiveKit documentation](https://docs.livekit.io/).
