import logging
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RunContext,
    cli,
    inference,
    metrics,
    room_io,
)
from livekit.agents.llm import function_tool
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()
logger = logging.getLogger("v2v-agent")
logging.basicConfig(level=logging.INFO)


class MyAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=(
                "Your name is Kelly. You speak with users via voice. "
                "Keep responses short, natural, and conversational. "
                "No emojis, no markdown. Speak only English."
            )
        )

    async def on_enter(self):
        self.session.generate_reply(allow_interruptions=False)

    @function_tool
    async def lookup_weather(
        self,
        context: RunContext,
        location: str,
        latitude: str,
        longitude: str,
    ):
        logger.info(f"Weather requested for {location}")
        return "It is sunny and about 70 degrees."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("openai/gpt-4.1-mini"),
        tts=inference.TTS(
            "cartesia/sonic-3",
            voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        resume_false_interruption=True,
        false_interruption_timeout=1.0,
    )

    usage = metrics.UsageCollector()

    @session.on("metrics_collected")
    def on_metrics(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage.collect(ev.metrics)

    async def on_shutdown():
        logger.info(f"Usage summary: {usage.get_summary()}")

    ctx.add_shutdown_callback(on_shutdown)

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions()
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
