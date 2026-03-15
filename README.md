# DocQuery - Multimodal AI Agent Platform

A multimodal AI agent platform with browser-based voice conversation, text chat, OCR document processing, Vector RAG knowledge base, and Google Calendar scheduling.

---

## Quick Start

### Start / Restart All Services
```
start.bat restart
```
Launches: LiveKit Server, FastAPI Server (`server.py`), and AI Voice Agent (`mcp-agent.py`).

### Stop All Services
```
start.bat stop
```

### Check Status
```
start.bat status
```

### Logs
```
logs\livekit.log   - LiveKit server
logs\server.log    - FastAPI backend
logs\agent.log     - Voice agent
```

---

## System Architecture

```
livekits/
├── server.py                  # FastAPI backend (RAG, chat, OCR, MCP tools)
├── mcp-agent.py               # LiveKit AI voice agent (Gemini + Deepgram STT + Cartesia TTS)
├── agent_personas.py          # Voice persona definitions
├── start.bat                  # Windows service orchestration
├── ocr/                       # OCR module (pytesseract)
│   ├── extractor.py           # Image preprocessing & text extraction
│   ├── table_extractor.py     # Structured table & key-value extraction
│   └── file_handlers.py       # PDF, DOCX, image, text file routing
├── calendar_integration/      # Google Calendar appointment scheduling
├── credentials/               # Service account keys
├── frontend/                  # React/Vite UI
│   └── src/App.tsx
├── sessions/                  # Recorded .wav audio & .txt transcripts
└── livekit.yaml               # LiveKit server config
```

### Core Services

| Service | File | Description |
|---------|------|-------------|
| FastAPI Server | `server.py` | Backend — FAISS vector store, file upload, OCR, text chat (Gemini), MCP tool server |
| Voice Agent | `mcp-agent.py` | LiveKit voice agent — Deepgram STT, Gemini LLM, Cartesia TTS |
| Frontend | `frontend/` | React UI with Chat, Speech, and History tabs |

---

## Features

### Text Chat
Chat with the AI agent via the browser. Uses Gemini with RAG context from uploaded documents.

### Voice Conversation (Speech Tab)
Connect via browser microphone for real-time voice conversation with the AI agent powered by LiveKit.

### OCR Document Processing
Upload PDFs, DOCX, images (PNG, JPG, TIFF, BMP) with:
- pytesseract OCR with image preprocessing
- Structured table extraction
- Key-value pair detection

### Vector RAG (Knowledge Base)
Upload documents to give the agent domain knowledge. Uses FAISS with `l3cube-pune/indic-sentence-similarity-sbert` for multilingual embeddings.

### Google Calendar Scheduling
The agent can check availability and book appointments through voice or text conversation.

### Session Recording
All voice sessions are recorded as 48kHz mono WAV files with timestamped transcripts in `sessions/`.

---

## UI

Access at `http://localhost:8005`

- **Chat** — Text conversation with the AI agent
- **Speech** — Real-time voice conversation via browser microphone
- **History** — Browse past sessions with audio playback and transcripts

---

## Configuration

### Environment Variables (`.env`)

```bash
# LiveKit
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=<your-secret>

# STT / TTS
DEEPGRAM_API_KEY=<key>
CARTESIA_API_KEY=<key>

# LLM (Gemini only)
LLM_PROVIDER=gemini
GOOGLE_API_KEY=<key>
GEMINI_MODEL=gemini-2.5-flash-lite
CHAT_MODEL=gemini-2.5-flash

# Google Calendar
GOOGLE_CALENDAR_CREDENTIALS=credentials/calendar.json
GOOGLE_CALENDAR_ID=<calendar-id>
DEFAULT_TIMEZONE_OFFSET=330
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/token` | Get LiveKit access token |
| POST | `/upload` | Upload document for RAG indexing (with OCR) |
| POST | `/api/chat` | Text chat with AI agent |
| POST | `/api/chat/clear` | Clear chat history |
| POST | `/api/ocr` | Standalone OCR extraction |
| GET | `/api/sessions` | List recorded sessions |
| GET | `/api/sessions/{id}/transcript` | Get session transcript |
| GET | `/api/personas` | Get active persona |

### MCP Tools (used by the voice agent)

| Tool | Description |
|------|-------------|
| `query_knowledge_base` | Search RAG vector store |
| `check_and_book_appointment` | Check calendar availability |
| `schedule_appointment` | Book a Google Calendar appointment |
| `get_appointment_info` | Get appointment configuration |

---

## Maintenance

### Update Frontend
```bash
cd frontend && npm install && npm run build
```

### Update Python Backend
```bash
pip install -r requirements.txt
```
