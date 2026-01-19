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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("mcp-agent")

load_dotenv()

class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a smart voice assistant equipped with a Retrieval-Augmented Generation (RAG) system. "
                "The user has uploaded a document which represents your primary knowledge base. "
                "If the user asks ANY question that might be answered by the document (technical details, stories, facts), "
                "you MUST use the `query_knowledge_base` tool to find the answer. "
                "Do not hallucinate. If the tool returns no info, say you don't know based on the document. "
                "Keep responses concise and conversational."
            )
        )

    async def on_enter(self):
        logger.info("✅ Agent entered session")
        self.session.generate_reply()

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
            # Ensure this matches the port your server.py is running on
            mcp.MCPServerHTTP(url="http://localhost:8000/sse"),
        ],
    )

    # ... (Timing logic remains the same as your original file) ...
    # Re-adding brevity for the snippet
    timings = {}
    @session.on("user_speech_committed")
    def _on_user_speech(*args):
        timings["speech_end"] = time.perf_counter()

    @session.on("llm_generation_started")
    def _on_llm_start(*args):
        timings["llm_start"] = time.perf_counter()

    @session.on("agent_audio_committed")
    def _on_first_audio(*args):
        now = time.perf_counter()
        if "speech_end" in timings:
             logger.info(f"⏱️ Response Latency: {(now - timings['speech_end'])*1000:.1f} ms")

    await session.start(agent=MyAgent(), room=ctx.room)

if __name__ == "__main__":
    cli.run_app(server)