import asyncio
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
    AutoSubscribe,
    cli,
    mcp,
)
from livekit.plugins import silero, openai, google
from livekit.plugins.deepgram import STT as DeepgramSTT
from livekit.plugins.cartesia import TTS as CartesiaTTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("mcp-agent")

load_dotenv()

# IMPORTANT: On EC2, the agent should connect to LiveKit via localhost 
# even though the browser needs the public IP.
if os.getenv("LIVEKIT_URL"):
    # We keep the public IP for the frontend but force the agent to use localhost
    os.environ["LIVEKIT_URL"] = "ws://localhost:7880"
    logger.info(f"?? Agent forcing internal connection: {os.environ['LIVEKIT_URL']}")

class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "You are a versatile voice assistant specialized in English, Tamil (?????), and Telugu (??????). "
                "Always respond in the SAME LANGUAGE as the user's spoken input. "
                "Note: You might be speaking to someone on a phone call. Keep answers extremely concise, "
                "natural, and avoid long lists. For casual greetings, answer directly. "
                "For any knowledge questions, you MUST use the tool 'query_knowledge_base'."
            )
        )

    async def on_enter(self):
        logger.info("? Agent entered session")

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    logger.info(f"??? Starting agent session for room: {ctx.room.name}")
    
    # Connect with AUDIO_ONLY to reduce metadata processing
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    # Initialize Plugins
    stt = DeepgramSTT(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        model="nova-3",
        language="en" # Let Deepgram detect the language for better accuracy
    )

    # Initialize LLM based on provider setting
    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if llm_provider == "gemini":
        logger.info("?? Using Google Gemini LLM")
        llm = google.LLM(
            model="gemini-2.5-flash-lite", # or "gemini-1.5-flash"
            api_key=os.environ["GOOGLE_API_KEY"]
        )
    else:
        logger.info("?? Using OpenAI LLM")
        llm = openai.LLM(
            model="gpt-4o-mini",
            api_key=os.environ["OPENAI_API_KEY"]
        )

    tts = CartesiaTTS(
        api_key=os.environ["CARTESIA_API_KEY"],
        model="sonic-3",
        voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"
    )
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=stt,
        llm=llm,
        tts=tts,
        mcp_servers=[
            mcp.MCPServerHTTP(url=f"http://localhost:{os.getenv('PORT', '8000')}/mcp/sse"),
        ],
        preemptive_generation=True,
    )

    # Greet the user when they join
    async def say_greeting():
        await asyncio.sleep(1.0)
        await session.say("Welcome to Coastal Seven Consulting, how can I help you?", allow_interruptions=True)

    asyncio.create_task(say_greeting())

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
                logger.info(f"?? STT Latency: {latency:.0f}ms")

    # 3. LLM DONE / TTS START (Text sent to TTS)
    @session.on("speech_created")
    def on_speech_created(event):
        now = time.perf_counter()
        timings["tts_start"] = now
        
        # Calculate LLM + Tool Latency
        if "stt_done" in timings:
            llm_latency = (now - timings["stt_done"]) * 1000
            logger.info(f"?? LLM/Tool Latency: {llm_latency:.0f}ms")

    # 4. AUDIO PLAYING (First byte of audio sent)
    @session.on("agent_speech_committed")
    def on_agent_speech_committed(msg):
        now = time.perf_counter()
        
        # Calculate TTS Latency (Generation + Network)
        if "tts_start" in timings:
            tts_latency = (now - timings["tts_start"]) * 1000
            logger.info(f"??? TTS Latency: {tts_latency:.0f}ms")

        # Calculate TOTAL Latency
        if "speech_end" in timings:
            total_latency = (now - timings["speech_end"]) * 1000
            logger.info(f"? TOTAL Voice-to-Voice Latency: {total_latency:.0f}ms")
            
            # Reset for next turn
            timings.pop("speech_end", None)
            timings.pop("stt_done", None)
            timings.pop("tts_start", None)

    try:
        await session.start(agent=MyAgent(), room=ctx.room)
    finally:
        logger.info("?? Disconnecting room immediately...")
        ctx.room.disconnect()

if __name__ == "__main__":
    os.environ.pop("LIVEKIT_AGENT_GATEWAY_URL", None)
    cli.run_app(server)