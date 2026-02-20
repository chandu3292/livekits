# LiveKit Voice Explorer 🧠🎙️

A production-grade RAG-powered voice assistant with support for Web and Telephony (Twilio SIP).

## 🚀 Quick Start (EC2 / Linux)

### 1. Prerequisites
Ensure you have the following installed on your server:
- **Python 3.10+**
- **Node.js & npm**
- **Redis Server** (`sudo apt install redis-server -y`)
- **LiveKit Server Binary** (`curl -sSL https://get.livekit.io | bash`)

### 2. Infrastructure Setup
Run these commands once to optimize your server and set up the telephony bridge:

```bash
# Optimize network buffers for high-quality audio
sudo sysctl -w net.core.rmem_max=2500000

# Install Python dependencies
pip install -r requirements.txt

# Build the Frontend (Serves the Web Dashboard)
cd frontend
npm install
npm run build
cd ..

# Provision SIP rules (Connects Twilio to LiveKit)
# Ensure LiveKit server is running (Step 3) before running this
python3 provision_sip.py
```

### 3. Launching the System
You need to run these **three commands** in separate terminal sessions (or using `pm2`/`tmux`).

**Terminal 1: LiveKit Media Server**
```bash
livekit-server --config livekit.yaml
```

**Terminal 2: Backend API & Dashboard**
```bash
python3 server.py
```

**Terminal 3: AI Voice Agent**
```bash
python3 mcp-agent.py dev
```

---

## 📞 Telephony (Twilio) Integration

To enable phone calls to your AI:
1. **Twilio SIP Trunk**: Point your Termination/Origination SIP URI to `sip:<YOUR_EC2_IP>:5060`.
2. **Security Groups**: Open these ports in your AWS Console:
   - `8005` (Web Dashboard)
   - `7880-7882` (LiveKit WebRTC)
   - `5060` (SIP Signaling)
   - `10000-60000 (UDP)` (Telephony Audio - CRITICAL)

call this number to test the agent: +19378802983

## 🛠️ Key Components
- **`server.py`**: FastAPI backend with RAG implementation (OpenAI Embeddings + FAISS).
- **`mcp-agent.py`**: The "brain" of the agent using Deepgram (STT), Gemini/OpenAI (LLM), and Cartesia (TTS).
- **`frontend/`**: Premium React dashboard for uploading docs and talking to the agent via browser.
- **`livekit.yaml`**: Server configuration including Redis & SIP settings.

## 📝 Configuration
Update the `.env` file with your API keys:
- `LIVEKIT_URL`: Set to `ws://<YOUR_EC2_IP>:7880`
- `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, `GOOGLE_API_KEY`
