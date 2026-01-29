import logging
import time
from dotenv import load_dotenv
import os

# Disable default gateways for local development
os.environ["LIVEKIT_DISABLE_GATEWAYS"] = "true"
os.environ["LIVEKIT_DISABLE_AGENT_GATEWAY"] = "true"

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    mcp,
)
from livekit.plugins import silero, openai
from livekit.plugins.deepgram import STT as DeepgramSTT
from livekit.plugins.cartesia import TTS as CartesiaTTS
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("mcp-agent")

load_dotenv()

class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a voice assistant powered by a specific knowledge base. "
                "You have access to a tool called 'query_knowledge_base'. "
                "For specific questions, you MUST use the tool. "
                "For casual greetings (e.g., 'hello', 'hi'), answer directly WITHOUT using the tool. "
                "Keep answers concise."
            )
        )

    async def on_enter(self):
        logger.info("✅ Agent entered session")

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Initialize Plugins
    stt = DeepgramSTT(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        language="multi"
    )

    llm = openai.LLM(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"]
    )

    tts = CartesiaTTS(
        api_key=os.environ["CARTESIA_API_KEY"]
    )

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=stt,
        llm=llm,
        tts=tts,
        turn_detection=MultilingualModel(),
        mcp_servers=[
            mcp.MCPServerHTTP(url="http://localhost:8000/mcp/sse"),
        ],
        preemptive_generation=True,
    )

    # --- LATENCY TRACKING ---
    timings = {}

    # 1. USER STOPS SPEAKING
    @session.on("user_state_changed")
    def on_user_state_changed(event):
        if str(event.new_state) == "listening":
            timings["speech_end"] = time.perf_counter()

    # 2. STT DONE (Text is ready)
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(event):
        if event.is_final:
            now = time.perf_counter()
            timings["stt_done"] = now
            if "speech_end" in timings:
                latency = (now - timings["speech_end"]) * 1000
                logger.info(f"⏱️ STT Latency: {latency:.0f}ms")

    # 3. LLM DONE / TTS START (Text sent to TTS)
    @session.on("speech_created")
    def on_speech_created(event):
        now = time.perf_counter()
        timings["tts_start"] = now
        
        # Calculate LLM + Tool Latency
        if "stt_done" in timings:
            llm_latency = (now - timings["stt_done"]) * 1000
            logger.info(f"🧠 LLM/Tool Latency: {llm_latency:.0f}ms")

    # 4. AUDIO PLAYING (First byte of audio sent)
    @session.on("agent_speech_committed")
    def on_agent_speech_committed(msg):
        now = time.perf_counter()
        
        # Calculate TTS Latency (Generation + Network)
        if "tts_start" in timings:
            tts_latency = (now - timings["tts_start"]) * 1000
            logger.info(f"🗣️ TTS Latency: {tts_latency:.0f}ms")

        # Calculate TOTAL Latency
        if "speech_end" in timings:
            total_latency = (now - timings["speech_end"]) * 1000
            logger.info(f"⚡ TOTAL Voice-to-Voice Latency: {total_latency:.0f}ms")
            
            # Reset for next turn
            timings.pop("speech_end", None)
            timings.pop("stt_done", None)
            timings.pop("tts_start", None)

    await session.start(agent=MyAgent(), room=ctx.room)

if __name__ == "__main__":
    os.environ.pop("LIVEKIT_AGENT_GATEWAY_URL", None)
    cli.run_app(server)