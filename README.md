# DocQuery

A multimodal AI agent platform that combines real-time voice conversation, intelligent document processing, and calendar scheduling into a single integrated system. Users interact with a context-aware AI assistant via text chat or voice, asking questions about uploaded documents, and booking appointments through natural language.

---

## Key Features

- **Real-time voice AI** using WebRTC (LiveKit), Deepgram STT, Cartesia TTS, and Google Gemini 2.5 Flash
- **Retrieval-Augmented Generation (RAG)** with FAISS + BGE-M3 hybrid search (dense + sparse embeddings)
- **Multiformat OCR pipeline** supporting PDF, DOCX, PNG, JPG, TIFF, BMP, WEBP, Markdown, and plain text
- **Google Calendar scheduling** via natural language ("book me a slot tomorrow afternoon")
- **Five voice personas** with multilingual support across English, Hindi, and Telugu
- **Session recording** with synchronized WAV audio and timestamped transcripts
- **Model Context Protocol (MCP)** integration connecting the voice agent to backend tools over HTTP/SSE

---

## Architecture

```
Browser (React 19 + LiveKit WebRTC)
        |
        v
FastAPI Backend (port 8005)
  |-- /api/chat         RAG-powered text conversation
  |-- /upload           Document ingestion and indexing
  |-- /api/ocr          Standalone OCR extraction
  |-- /token            LiveKit access token
  |-- /mcp/sse          MCP tool server (for voice agent)
  |-- /api/sessions     Session history and playback
        |
        v
Core Modules
  |-- KnowledgeBase     FAISS index + BGE-M3 hybrid search
  |-- OCR Pipeline      Tesseract + PIL preprocessing
  |-- Calendar Module   Google Calendar API + availability checker
        |
        v
LiveKit Voice Agent (mcp-agent.py)
  |-- STT: Deepgram Nova-3
  |-- LLM: Gemini 2.5 Flash-lite
  |-- TTS: Cartesia Sonic-3
  |-- VAD: Silero
  |-- MCP: Connects to FastAPI tool server
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Voice Agent | LiveKit Agents, Google Gemini 2.5 Flash |
| STT | Deepgram Nova-3 |
| TTS | Cartesia Sonic-3 |
| Embeddings | BGE-M3 (FlagEmbedding, 1024-dim) |
| Vector Search | FAISS IndexFlatIP |
| OCR | Tesseract + Pillow |
| Document Parsing | PyPDF, python-docx |
| Backend API | FastAPI + Uvicorn |
| Frontend | React 19, TypeScript, Vite 7 |
| Calendar | Google Calendar API v3 |
| Protocol | MCP (Model Context Protocol) over SSE |

---

## RAG Pipeline

Documents are chunked (1000 chars, 200-char overlap) and embedded using BGE-M3, a multilingual model producing 1024-dimensional vectors. Retrieval uses a weighted combination:

- **Dense similarity** (cosine via FAISS IndexFlatIP): 60%
- **Sparse lexical matching**: 40%

Top-3 retrieved chunks are injected as context into the Gemini prompt before generating a response.

---

## OCR Pipeline

```
Input File
    |
file_handlers.py  (routes by MIME type / extension)
    |
extractor.py      (grayscale + contrast + binarization + pytesseract)
    |
table_extractor.py (structured table and key-value pair extraction)
```

For PDFs, each page is analyzed for text density. Pages with fewer than 50 characters fall back to image-based OCR at 200 DPI.

---

## Calendar Scheduling

The `calendar_integration/` module exposes four MCP tools to the voice agent:

| Tool | Purpose |
|---|---|
| `query_knowledge_base` | Retrieve context from indexed documents |
| `check_and_book_appointment` | Check availability for a natural language date |
| `schedule_appointment` | Create a Google Calendar event |
| `get_appointment_info` | Return slot duration and configuration |

Business hours default to Monday through Friday, 9 AM to 5 PM IST, in 30-minute slots.

---

## Voice Personas

| Persona | Language | Voice |
|---|---|---|
| Sophia | English | Female, professional |
| Alex | English | Male, casual |
| Maya | English | Female, Indian accent |
| Priya | Telugu | Female |
| Arjun | Hindi | Male |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Tesseract OCR installed and on PATH
- LiveKit server binary
- API keys: Google Gemini, Deepgram, Cartesia, Google Calendar

### Setup

```bash
git clone https://github.com/chandu3292/livekits.git
cd livekits
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

cd frontend
npm install
npm run build
cd ..

cp .env.example .env
# Fill in API keys in .env
```

### Run

```bash
start.bat restart   # Start all services (LiveKit + FastAPI + Voice Agent)
start.bat stop      # Stop all services
start.bat status    # Check service status
```

Open `http://localhost:8005` in your browser.

---

## Environment Variables

```env
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=your_secret

DEEPGRAM_API_KEY=your_key
CARTESIA_API_KEY=your_key
GOOGLE_API_KEY=your_key

GEMINI_MODEL=gemini-2.5-flash-lite
CHAT_MODEL=gemini-2.5-flash

GOOGLE_CALENDAR_ID=your_calendar_id
DEFAULT_TIMEZONE_OFFSET=330
PORT=8005
```

---

## Project Structure

```
livekits/
├── server.py               FastAPI server, RAG, MCP tool endpoints
├── mcp-agent.py            LiveKit voice agent
├── agent_personas.py       Persona definitions and voice mappings
├── ocr/
│   ├── file_handlers.py    Format routing (PDF, DOCX, images, text)
│   ├── extractor.py        Tesseract + PIL preprocessing
│   └── table_extractor.py  Table and key-value extraction
├── calendar_integration/
│   ├── google_calendar.py  Google Calendar API wrapper
│   ├── availability_checker.py
│   └── appointment_manager.py
├── frontend/               React 19 + Vite 7 UI
│   └── src/App.tsx         Chat, Speech, and History tabs
├── sessions/               Recorded WAV files and transcripts
└── logs/                   Per-service log files
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/token` | LiveKit access token with persona metadata |
| POST | `/upload` | Upload and index a document |
| POST | `/api/chat` | Text conversation with RAG context |
| POST | `/api/ocr` | Standalone OCR extraction |
| GET | `/api/documents` | List indexed documents |
| GET | `/api/personas` | List available personas |
| POST | `/api/set-persona` | Switch active persona |
| GET | `/api/sessions` | List recorded voice sessions |
| GET | `/api/sessions/{id}/transcript` | Get session transcript |

---

## License

MIT
