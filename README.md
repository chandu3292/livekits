# LiveKit AI Agent with SIP & Handoff Analysis

A multimodal AI voice agent platform with SIP telephony (inbound + outbound), Vector RAG knowledge base, Google Calendar scheduling, and a Handoff Analysis dashboard for human agents.

---

## Quick Start

### Start / Restart All Services
```bash
bash run_all.sh
```
This launches: Redis, LiveKit Server, SIP Gateway, RAG Server (`server.py`), and AI Agent (`mcp-agent.py`).
The script kills any previous instances before starting.

### Stop All Services
```bash
pkill -f livekit && pkill -f python3
```

### Monitoring Logs
```bash
tail -f logs/agent.log          # AI agent thoughts
tail -f logs/server.log         # RAG server & API
tail -f logs/livekit-sip.log    # SIP gateway
tail -f logs/livekit-server.log # LiveKit core
```

---

## System Architecture

```
livekits/
├── server.py                  # FastAPI backend (RAG, API endpoints, MCP tools)
├── mcp-agent.py               # LiveKit AI agent (voice, SIP, tool calls)
├── provision_sip.py           # SIP trunk & dispatch rule provisioning
├── run_all.sh                 # Service orchestration script
├── outbound_calls/            # Outbound call module (Vobiz SIP)
│   ├── __init__.py
│   ├── trunk.py               # Outbound trunk management & caching
│   └── dialer.py              # SIP participant dialer
├── calendar_integration/      # Google Calendar appointment scheduling
│   ├── google_calendar.py     # Calendar API integration
│   ├── availability_checker.py# Business hours & slot logic
│   └── appointment_manager.py # Appointment coordination
├── frontend/                  # React/Vite UI
│   └── src/App.tsx
├── sip/                       # LiveKit SIP binary & config
├── sessions/                  # Recorded .wav audio & .txt transcripts
├── livekit.yaml               # LiveKit server config
├── sip-config.yaml            # SIP gateway config
└── handoffs.json              # AI-to-human transfer records
```

### Core Services

| Service | File | Description |
|---------|------|-------------|
| RAG Server | `server.py` | FastAPI backend — vector embeddings (FAISS/Valkey), file upload, transcript parsing, handoff analysis, MCP tool server |
| AI Agent | `mcp-agent.py` | LiveKit voice agent — listens to speech, queries knowledge base, schedules appointments, transfers calls |
| SIP Gateway | `sip/` | Bridges SIP/telephony traffic into LiveKit rooms |
| Frontend | `frontend/` | React dashboard for live calls, document upload, session history, handoffs, and outbound dialing |

---

## Features

### Inbound Calls (SIP)
Callers dial in via SIP trunk. Extension-based language routing:
- **100** — English
- **101** — Tamil
- **102** — Telugu

Provisioned via `provision_sip.py` which creates inbound trunks, dispatch rules, and the outbound trunk.

### Outbound Calls (Vobiz SIP)
Dial out to any phone number from the UI. The AI agent greets with *"Hello, I'm calling from Coastal Seven Consulting"* and then operates identically to inbound calls.

**Module:** `outbound_calls/`
- `trunk.py` — Creates/caches a Vobiz outbound SIP trunk
- `dialer.py` — Creates a SIP participant that dials out and joins a LiveKit room

**API:** `POST /api/outbound-call` with `{"phone_number": "+91XXXXXXXXXX"}`

### Vector RAG (Knowledge Base)
Upload PDF, TXT, or MD files via the UI to give the agent domain knowledge. Supports two backends:
- **FAISS** — In-memory, local (default)
- **Valkey/Redis** — Shared, persistent (set `VECTOR_STORE=valkey` in `.env`)

Uses `l3cube-pune/indic-sentence-similarity-sbert` for multilingual embeddings (English, Tamil, Telugu).

### Google Calendar Scheduling
The agent can check availability and book appointments through voice conversation. See `calendar_integration/README.md` for setup details.

### Smart Handoff (Transfer to Human)
When a caller requests a human agent, the AI:
1. Initiates a SIP transfer to the configured number
2. Analyzes the call transcript (sentiment, user details, actionable requests)
3. Saves the analysis to the Handoffs dashboard

### Session Recording
All calls are recorded as 48kHz mono WAV files with timestamped transcripts in `sessions/`.

---

## Dashboard

Access at `http://<server-ip>:8000`

- **Live Call** — Connect via browser microphone to test the agent
- **Document Upload** — Drag-and-drop files with category tagging for scoped RAG
- **Outbound Call** — Enter a phone number and click Call to dial out
- **History** — Browse past sessions with audio playback and transcripts
- **Handoffs** — Review AI-to-human transfers with sentiment analysis

---

## Configuration

### Environment Variables (`.env`)

```bash
# LiveKit
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=<your-secret>
TOKEN_URL=ws://localhost:7880

# LLM
OPENAI_API_KEY=<key>
OPENAI_MODEL=gpt-4o-mini
GOOGLE_API_KEY=<key>
GEMINI_MODEL=gemini-2.5-flash-lite

# Vector Store Backend — options: faiss, valkey
VECTOR_STORE=valkey

# Valkey / Redis (when VECTOR_STORE=valkey)
VALKEY_HOST=<host>
VALKEY_PORT=6379
VALKEY_PASSWORD=
VALKEY_SSL=true

# Vobiz SIP (Outbound Calls)
VOBIZ_SIP_HOST=<sip-host>
VOBIZ_SIP_USERNAME=<username>
VOBIZ_SIP_PASSWORD=<password>

# Google Calendar
GOOGLE_CALENDAR_CREDENTIALS=/path/to/service-account-key.json
GOOGLE_CALENDAR_ID=primary
```

### SIP Provisioning
After updating `.env`, re-provision SIP trunks and dispatch rules:
```bash
./venv/bin/python3 provision_sip.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/token` | Get LiveKit access token |
| POST | `/upload` | Upload document for RAG indexing |
| GET | `/api/documents` | List indexed document categories |
| POST | `/api/set-active-doc` | Set active RAG category |
| POST | `/api/outbound-call` | Initiate outbound SIP call |
| POST | `/api/clear-db` | Clear all vector store data |
| GET | `/api/sessions` | List recorded sessions |
| GET | `/api/sessions/{id}/transcript` | Get session transcript |
| POST | `/api/analyze-transcript` | Run LLM analysis on transcript text |
| GET | `/api/handoffs` | List AI-to-human transfer records |

### MCP Tools (used by the AI agent)

| Tool | Description |
|------|-------------|
| `query_knowledge_base` | Search RAG vector store |
| `check_and_book_appointment` | Check calendar availability |
| `schedule_appointment` | Book a Google Calendar appointment |
| `get_appointment_info` | Get appointment configuration |
| `transfer_to_human` | SIP transfer + transcript analysis |

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
