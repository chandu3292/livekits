# LiveKit Multilingual MCP Agent

This project implements a sophisticated voice-to-voice AI agent using **LiveKit**, **Model Context Protocol (MCP)**, and **RAG (Retrieval-Augmented Generation)**. It supports English, Tamil (à®¤à®®à®¿à®´à¯ ), and Telugu (à°¤à±†à°²à± °—à± ) and is capable of answering questions based on uploaded documents.

## 🚀 Features

- **Multilingual Support**: Real-time speech-to-speech interaction in English, Tamil, and Telugu.
- **RAG Capability**: In-memory FAISS vector database for querying PDF and text documents.
- **MCP Integration**: Uses the Model Context Protocol to decouple the knowledge base tool from the agent logic.
- **Low Latency**: Optimized pipeline using Deepgram (STT), OpenAI/Gemini (LLM), and Cartesia (TTS).
- **Telephony Support**: Integrated SIP provisioning for phone-based interactions.
- **Modern Frontend**: A Vite-based React frontend for real-time interaction and document management.

---

## 🏗️ Project Structure

- `server.py`: FastAPI server that manages the RAG index, serves the frontend, and provides the MCP endpoint.
- `mcp-agent.py`: The core LiveKit agent logic using the Agent Framework.
- `frontend/`: React application for connecting to the room and uploading knowledge base files.
- `provision_sip.py`: Script to configure SIP dispatch rules and trunks in LiveKit.
- `livekit.yaml`: Configuration for the local LiveKit server.
- `sip-config.yaml`: Configuration for the SIP bridge.

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.9 or higher
- Node.js & npm (for frontend)
- Redis server (required for LiveKit)
- **livekit-server** binary (included in root as `livekit-server.exe`)

### 2. Environment Variables
Create a `.env` file in the root directory with the following keys:

```env
# LiveKit Configuration
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=devsecret_1234567890_abcdefghijklmnopqrstuvwxyz

# AI Provider Keys
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
CARTESIA_API_KEY=your_cartesia_api_key

# Optional Configuration
LLM_PROVIDER=openai # or 'gemini'
PORT=8000
```

### 3. Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Frontend dependencies
cd frontend
npm install
npm run build
cd ..
```

---

## 🏃 How to Run

Follow these steps in separate terminal windows:

### Step 1: Start Redis
Ensure Redis is running on `localhost:6379`.

### Step 2: Start LiveKit Server
```bash
./livekit-server.exe --config livekit.yaml
```


### Step 3: Start the Backend Server
This serves the frontend and the MCP/RAG service.
```bash
python server.py
```

### Step 4: Start the AI Agent
```bash
python mcp-agent.py dev
```

### Step 5: Provision SIP (Telephony)
If you want to enable phone calls to your agent:
1. Ensure the LiveKit SIP bridge is running (or the local server has SIP enabled).
2. Run the provisioning script to create a trunk and dispatch rule:
```bash
python provision_sip.py
```
*Note: This script maps incoming calls to the `phone_call_room` by default.*

### Step 6: Access the Application
Open `http://localhost:8000` in your browser. You can:
1. Upload a PDF or Text file to the knowledge base.
2. Connect to the LiveKit room.
3. Start speaking in English, Tamil, or Telugu.

---

### Running in Background (Linux/EC2)
For production or long-running sessions on a server, you can run the components in the background using `nohup`:

```bash
# Create logs directory if it doesn't exist
mkdir -p logs

# 1. Start LiveKit Server
nohup ./livekit-server --config livekit.yaml > logs/livekit-server.log 2>&1 &

# 2. Start LiveKit SIP Bridge
nohup livekit-sip --config sip-config.yaml > logs/livekit-sip.log 2>&1 &

# 3. Provision SIP (Run once to setup trunk/rules)
python3 provision_sip.py

# 4. Start the Backend Server (MCP/RAG)
nohup python3 server.py > logs/server.log 2>&1 &

# 5. Start the AI Agent (in dev mode)
nohup python3 mcp-agent.py dev > logs/agent.log 2>&1 &
```

---

## ⚙️ How it Works

1. **Knowledge Ingestion**: When you upload a file via the frontend, `server.py` extracts the text, chunks it, and generates embeddings using the `l3cube-pune/indic-sentence-similarity-sbert` model (optimized for Indic languages). These are stored in an in-memory FAISS index.
2. **MCP Tooling**: The `server.py` exposes a tool called `query_knowledge_base` via an MCP SSE endpoint. 
3. **Agent Interaction**: 
    - The agent listens to the user via **LiveKit**.
    - **Deepgram** converts speech to text.
    - The **LLM** (GPT-4o / Gemini) decides if it needs information from the knowledge base.
    - If needed, it calls the MCP tool, which performs a similarity search in the FAISS index.
    - The **LLM** generates a response in the user's language.
    - **Cartesia** converts the text back to high-quality speech.
4. **SIP Support**: For telephony, `provision_sip.py` can be used to map incoming SIP calls to specific LiveKit rooms, allowing users to call the agent from a regular phone.

---

## 📝 Configuration

- **STT**: Uses Deepgram `nova-3` with multi-language detection.
- **TTS**: Uses Cartesia `sonic-3` for ultra-low latency high-fidelity voice.
- **RAG**: Uses a threshold-based retrieval system to ensure the AI only answers when it has relevant context.
