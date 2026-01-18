import logging
import time

from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    inference,
    mcp,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# --------------------
# Logging setup
# --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("mcp-agent")

load_dotenv()


# --------------------
# Agent definition
# --------------------
class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a voice assistant. "
                "If the user asks about weather, you MUST call MCP tools. "
                "Do not guess. Speak short, clear English."
            )
        )

    async def on_enter(self):
        logger.info("✅ Agent entered session")
        self.session.generate_reply()


# --------------------
# LiveKit server
# --------------------
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
        mcp_servers=[
            mcp.MCPServerHTTP(url="http://localhost:8000/sse"),
        ],
    )

    timings = {}

    # 🎙️ user finished speaking (STT committed)
    # ADD *args to accept the incoming message/transcript
    @session.on("user_speech_committed")
    def _on_user_speech(*args):
        timings["speech_end"] = time.perf_counter()
        logger.info("🎙️ user_speech_committed")

    # 🧠 LLM begins generation
    # ADD *args to accept LLM metadata
    @session.on("llm_generation_started")
    def _on_llm_start(*args):
        timings["llm_start"] = time.perf_counter()
        
        # Check if speech_end exists (it might not if triggered by on_enter)
        if "speech_end" in timings:
            delta = timings["llm_start"] - timings["speech_end"]
            logger.info(f"🧠 LLM started after {delta*1000:.1f} ms")
        else:
            logger.info("🧠 LLM started (initial greeting)")

    # 🔊 first audio frame from agent
    # ADD *args to accept audio frame data
    @session.on("agent_audio_committed")
    def _on_first_audio(*args):
        now = time.perf_counter()
        
        if "llm_start" in timings:
            logger.info(
                f"🔊 first audio after LLM start: {(now - timings['llm_start'])*1000:.1f} ms"
            )
        
        if "speech_end" in timings:
            logger.info(
                f"⏱️ TOTAL latency (speech → audio): {(now - timings['speech_end'])*1000:.1f} ms"
            )

    await session.start(agent=MyAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)
