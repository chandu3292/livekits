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
                "You are a voice assistant powered by a specific knowledge base. "
                "You have access to a tool called 'query_knowledge_base'. "
                "You MUST call 'query_knowledge_base' for every user question to check for information first. "
                "Do not answer from your own knowledge unless the tool returns no results. "
                "Keep answers concise and conversational."
            )
        )

    async def on_enter(self):
        logger.info("✅ Agent entered session")
        # Optional: A greeting
        # self.session.generate_reply("Hello! Upload a document and I can answer questions about it.")

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS("cartesia/sonic-3"),
        turn_detection=MultilingualModel(),
        mcp_servers=[
            # Connects to our FastAPI wrapper in server.py
            mcp.MCPServerHTTP(url="http://localhost:8000/mcp/sse"),
        ],
    )
    
    # Latency logging
    timings = {}
    @session.on("user_speech_committed")
    def _on_user_speech(*args):
        timings["speech_end"] = time.perf_counter()

    @session.on("agent_audio_committed")
    def _on_first_audio(*args):
        if "speech_end" in timings:
            latency = (time.perf_counter() - timings["speech_end"]) * 1000
            logger.info(f"⏱️  Voice Response Latency: {latency:.1f} ms")

    await session.start(agent=MyAgent(), room=ctx.room)

if __name__ == "__main__":
    cli.run_app(server)