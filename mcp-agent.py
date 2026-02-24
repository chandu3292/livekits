import asyncio
import logging
import os
from dotenv import load_dotenv

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-agent")

load_dotenv()

os.environ["LIVEKIT_DISABLE_GATEWAYS"] = "true"
os.environ["LIVEKIT_DISABLE_AGENT_GATEWAY"] = "true"

server = AgentServer()


class MyAgent(Agent):
    def __init__(self, forced_language):
        lang_names = {"en": "English", "ta": "Tamil (தமிழ்)", "te": "Telugu (తెలుగు)"}
        target_lang = lang_names.get(forced_language, "English")
        
        base_instruction = (
            f"You are a versatile voice assistant specialized in English, Tamil, and Telugu. "
            f"STRICTLY respond ONLY in {target_lang}. "
            "Keep responses extremely concise and natural. "
            "For ANY question, you MUST use the tool 'query_knowledge_base'. "
            "Explain the answer using the information found in the tool context. "
            "give respose as you are talking in a phone call conversation"
            "If the context contains relevant details, synthesize a helpful response from them. "
            "Only say you don't know if the context is completely unrelated to the question. "
            f"STRICTLY call the tool in {target_lang}."
        )

        super().__init__(instructions=base_instruction)
        self.forced_language = forced_language
        self.voice_session = None
        self.iteration_count = 0

    async def on_user_turn_completed(self, turn_ctx, new_message):
        self.iteration_count += 1
        logger.info(
            f"\n------------------***********************************************"
            f"({self.iteration_count})"
            f"------------------***********************************************"
        )

    async def on_enter(self):
        logger.info(f"Agent on_enter called. Language: {self.forced_language}")
        if not self.voice_session:
            return

        if self.forced_language == "en":
            greeting = "Welcome to Coastal Seven Consulting. How can I help you today?"
        elif self.forced_language == "ta":
            greeting = "கோஸ்டல் செவன் கன்சல்டிங்கிற்கு உங்களை வரவேற்கிறோம். இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?" # Tamil
        elif self.forced_language == "te":
            greeting = "కోస్టల్ సెవెన్ కన్సల్టింగ్ కు స్వాగతం. ఈ రోజు నేను మీకు ఎలా సహాయం చేయగలను?" # Telugu
        else:
            greeting = "Welcome to Coastal Seven Consulting."

        # Add a 1s delay to ensure audio is stable before greeting
        await asyncio.sleep(1)

        await self.voice_session.say(
            greeting,
            allow_interruptions=False,
            add_to_chat_ctx=False
        )


@server.rtc_session()
async def entrypoint(ctx: JobContext):

    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)

    # ?? Wait for participant to join
    participant = await ctx.wait_for_participant()
    attrs = participant.attributes
    logger.info(f"Participant joined: {participant.identity} with attributes: {attrs}")
    
    # Try multiple common keys for the dialed extension
    extension = (
        attrs.get("sip.trunkPhoneNumber") or 
        attrs.get("sip.to_user") or 
        attrs.get("sip.called_number") or 
        ""
    )
    extension = extension.strip()

    print(f"Dialed Extension: '{extension}' from {participant.identity}")

    if extension == "100":
        forced_language = "en"
    elif extension == "101":
        forced_language = "ta"
    elif extension == "102":
        forced_language = "te"
    else:
        logger.warning(f"Unknown extension '{extension}', defaulting to English")
        forced_language = "en"

    # STT locked to selected language
    stt = DeepgramSTT(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        model="nova-3",
        language=forced_language
    )

    llm_provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if llm_provider == "gemini":
        llm = google.LLM(
            model="gemini-2.5-flash-lite",
            api_key=os.environ["GOOGLE_API_KEY"]
        )
    else:
        llm = openai.LLM(
            model="gpt-4o",
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
            mcp.MCPServerHTTP(
                url=f"http://localhost:{os.getenv('PORT', '8000')}/mcp/sse"
            )
        ],
        preemptive_generation=True,
    )

    agent = MyAgent(forced_language)
    agent.voice_session = session

    try:
        await session.start(agent=agent, room=ctx.room)
        
        # Keep the session alive until it's closed
        close_future = asyncio.Future()
        session.on("close", lambda _: close_future.set_result(None) if not close_future.done() else None)
        await close_future
    finally:
        await ctx.room.disconnect()


if __name__ == "__main__":
    cli.run_app(server)