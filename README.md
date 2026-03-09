# 🎙️ LiveKit AI Agent with SIP & Handoff Analysis

A sophisticated multimodal AI agent platform featuring SIP (Telephony) integration, Vector RAG (Knowledge Base), Google Calendar scheduling, and a premium "Handoff Analysis" dashboard for human agents.

---

## 🚀 Quick Start (Running & Restarting)

The entire system is orchestrated via a background management script.

### 1. Start / Restart All Services
To launch the complete stack (Redis, LiveKit, SIP Gateway, RAG Server, and AI Agent):
```bash
bash run_all.sh
```
*Note: This script automatically kills any previous instances before starting new ones.*

### 2. Stop All Services
To immediately terminate all background processes:
```bash
pkill -f livekit && pkill -f python3
```

### 3. Monitoring Logs
Individual service logs are stored in the `./logs/` directory:
- **AI Thoughts**: `tail -f logs/agent.log`
- **RAG Server**: `tail -f logs/server.log`
- **SIP Status**: `tail -f logs/livekit-sip.log`
- **LiveKit Core**: `tail -f logs/livekit-server.log`

---

## 🛠️ System Architecture

### 1. Core Services
- **RAG Server (`server.py`)**: A FastAPI backend managing FAISS vector embeddings, transcript parsing, and the Handoff Analysis API.
- **AI Agent (`mcp-agent.py`)**: The LiveKit logic that listens to voice, queries the knowledge base, and performs tool calls (scheduling, transfers).
- **SIP Gateway**: Connects the AI agent to the telephony network.

### 2. Intelligence Features
- **Vector RAG**: Upload documents (PDF, TXT, MD) via the UI to give the agent immediate domain knowledge.
- **Smart Transfer**: When a user asks for a human, the agent initiates a SIP transfer and concurrently generates a "Handoff Analysis".
- **Handoff Analysis**: Automatically extracts **User Details**, **Sentiment**, and **Actionable Requests** from the call transcript to brief the receiving human agent.

---

## 🖥️ Dashboard Features

Access the dashboard at `http://localhost:8000` (or your configured port).

- **Live Call Interface**: Connect directly via the browser to test the agent's voice response.
- **Context Injection**: Drag-and-drop documents to update the agent's knowledge base in real-time.
- **History**: Review past recorded sessions, listen to audio playbacks, and read structured transcripts.
- **Handoffs**: A premium workspace showing transfers to human agents, complete with sentiment analysis and raw Q&A logs.

---

## ⚙️ Configuration & Setup

### Environment Variables (`.env`)
Ensure your `.env` contains:
- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- `OPENAI_API_KEY` (for RAG and Analysis)
- `GOOGLE_API_KEY` (if using Gemini)
- `TOKEN_URL` (usually your server's access token endpoint)

### File Structure
- `/frontend`: React/Vite source code.
- `/sessions`: Storage for `.wav` (audio) and `.txt` (transcripts).
- `/sip`: SIP configuration and logs.
- `handoffs.json`: Persistent record of AI-to-Human transfers.

---

## 📦 Maintenance

### Update Frontend
If you modify anything in the `frontend/` directory, you must rebuild:
```bash
cd frontend
npm install
npm run build
```

### Update Python Backend
```bash
pip install -r requirements.txt
```

---

