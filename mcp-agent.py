import asyncio
import logging
import os
import datetime
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
from livekit import rtc
from livekit.agents.voice import ConversationItemAddedEvent
import wave
import struct
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
            "For ANY question (EXCEPT for order bookings or providing order details), you MUST use the tool 'query_knowledge_base'. "
            "Explain the answer using the information found in the tool context. "
            "give respose as you are talking in a phone call conversation"
            "If the context contains relevant details, synthesize a helpful response from them. "
            "Only say you don't know if the context is completely unrelated to the question. "
            f"STRICTLY call the tool in {target_lang}."
            "\n\nAPPOINTMENT SCHEDULING:\n"
            "When user asks about appointment details (like duration or break time), call 'get_appointment_info' to get current configuration.\n"
            "All appointments are stored in Google Calendar.\n"
            "Current timezone: India Standard Time (IST).\n"
            "IMPORTANT: If the break/buffer time is 0, do NOT mention anything about break or buffer time in the conversation.\n"
            "When a user mentions a date or asks to book an appointment:\n"
            "1. As soon as they mention a date/day (e.g., 'tomorrow', 'next Monday', 'March 15th'), say 'Let me check the availability for you' and immediately call 'check_and_book_appointment'\n"
            "2. If user mentions only a TIME without a DATE, default to tomorrow's date (not today)\n"
            "3. The tool will automatically check Google Calendar availability and suggest next available date if needed\n"
            "4. Present ONLY the starting times of the available time slots to the user (e.g., 'I have slots at 9:00 AM and 10:00 AM')\n"
            "5. Once they confirm a specific time, collect: name, email, and optionally phone\n"
            "6. Confirm ALL details with user (date, ONLY the starting time, contact info)\n"
            "7. Before calling schedule_appointment say 'One moment, I'm scheduling your appointment' then call 'schedule_appointment' to create the calendar event\n"
            "8. If scheduling fails due to 'past time', suggest the next available date with the same time\n"
            "9. Inform user that the appointment has been added to their calendar and they'll receive a calendar invitation\n\n"
            "\n\nORDER HANDLING:\n"
            "If the user mentions wanting to book an order, place an order, or provides any order-related details:\n"
            "1. Listen carefully and acknowledge the details naturally.\n"
            "2. Do NOT use 'query_knowledge_base' for recording or acknowledging order details provided by the user.\n"
            "3. If the user says they want to book or place an order but HAS NOT provided details yet, YOU MUST ASK them for the specific details (such as items, quantity, or specific requirements).\n"
            "4. Only once they have provided the details, inform the user that their order details have been noted. and ask if final or need to add anything\n\n"
            "IMPORTANT: Always call 'check_and_book_appointment' when user mentions ANY date. No need to ask about appointment type - there's only one type. Always say a brief acknowledgment before calling any appointment tool."
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
        
        # Manually log the greeting to transcript since it's not in chat context
        if hasattr(self, "log_callback") and self.log_callback:
            self.log_callback("assistant", greeting)


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

    # Voice Selection based on language
    if forced_language in ["ta", "te"]:
        voice_id = "f8f5f1b2-f02d-4d8e-a40d-fd850a487b3d" # Indic voice
    else:
        voice_id = "f786b574-daa5-4673-aa0c-cbe3e8534c02" # English voice

    tts = CartesiaTTS(
        api_key=os.environ["CARTESIA_API_KEY"],
        model="sonic-3",
        voice=voice_id,
        sample_rate=48000 # Make the agent talk faster and more naturally
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

    # ── Audio & Transcript capture ───────────────────────────────────────────
    # Use IST (UTC+5:30) for filenames and logging
    ist_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    session_id = ist_now.strftime("%Y%m%d_%H%M%S")
    os.makedirs("sessions", exist_ok=True)
    base_name = f"sessions/session_{session_id}_{participant.identity}"
    transcript_file = f"{base_name}.txt"
    audio_file = f"{base_name}.wav"
    
    conversation_log: list[dict] = []
    
    # Audio Capture Logic (48kHz Mono - High-Fidelity Production Standard)
    wav_out = wave.open(audio_file, "wb")
    wav_out.setnchannels(1)
    wav_out.setsampwidth(2)
    wav_out.setframerate(48000)
    
    audio_tasks = []
    write_lock = asyncio.Lock()
    total_frames_written = 0

    async def record_track(track: rtc.Track):
        nonlocal total_frames_written
        logger.info(f"Started recording track: {track.sid or 'local'} - Normalizing to 48kHz")
        audio_stream = rtc.AudioStream(track, sample_rate=48000, num_channels=1)
        try:
            async for event in audio_stream:
                async with write_lock:
                    # Write raw PCM data
                    wav_out.writeframes(event.frame.data.tobytes())
                    total_frames_written += 1
                    if total_frames_written % 100 == 0:
                        logger.info(f"Audio progressing: {total_frames_written//50} seconds saved...")
        except Exception as e:
            logger.error(f"Error recording track {track.sid}: {e}")
        finally:
            logger.info(f"Stopped recording track: {track.sid or 'local'}")

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.Participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_tasks.append(asyncio.create_task(record_track(track)))

    # Start recording any ALREADY subscribed tracks
    for p in ctx.room.remote_participants.values():
        for pub in p.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                audio_tasks.append(asyncio.create_task(record_track(pub.track)))

    # Also record the AGENT'S own audio (local participant)
    async def record_local_audio():
        # Wait for agent to publish track
        while not any(pub.track is not None and pub.track.kind == rtc.TrackKind.KIND_AUDIO 
                      for pub in ctx.room.local_participant.track_publications.values()):
            await asyncio.sleep(0.1)
        
        for pub in ctx.room.local_participant.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info("Found local audio track, adding to mixer")
                audio_tasks.append(asyncio.create_task(record_track(pub.track)))
                break

    audio_tasks.append(asyncio.create_task(record_local_audio()))

    def log_turn(role: str, text: str):
        """Helper to write to log and transcript file."""
        if not text:
            return
        label = "👤 User   " if role == "user" else "🤖 Agent  "
        # Use IST for per-message timing
        ist_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
        ts    = ist_now.strftime("%H:%M:%S")
        line  = f"[{ts}] {label}: {text}"

        conversation_log.append({"role": role, "text": text, "time": ts})
        logger.info(line)

        # Append to per-session transcript file
        with open(transcript_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # Hook the logger into the agent for manual greeting logging
    agent.log_callback = log_turn

    def on_conversation_item_added(event: ConversationItemAddedEvent) -> None:
        """Called for turns committed to chat history."""
        item = event.item
        if not hasattr(item, "role"):
            return
        log_turn(str(item.role), item.text_content)

    session.on("conversation_item_added", on_conversation_item_added)
    logger.info(f"📝 Transcript will be saved to: {transcript_file}")
    # ────────────────────────────────────────────────────────────────────────

    try:
        await session.start(agent=agent, room=ctx.room)
        
        # Keep the session alive until it's closed
        close_future = asyncio.Future()
        session.on("close", lambda _: close_future.set_result(None) if not close_future.done() else None)
        await close_future
    finally:
        # Close audio file and tasks
        logger.info(f"Finalizing audio recording: {audio_file}")
        for t in audio_tasks:
            t.cancel()
        
        # Ensure we wait for tasks to stop
        if audio_tasks:
            await asyncio.gather(*audio_tasks, return_exceptions=True)
            
        wav_out.close()

        # Check file size
        if os.path.exists(audio_file):
            size = os.path.getsize(audio_file)
            logger.info(f"✅ RECORDING COMPLETE: {audio_file} ({size} bytes, {total_frames_written} frames)")
        else:
            logger.error(f"❌ RECORDING FAILED: {audio_file} not found")

        # Print full conversation summary on disconnect
        if conversation_log:
            logger.info("\n" + "="*60)
            logger.info(f"📋 FULL CONVERSATION SUMMARY")
            logger.info("="*60)
            for entry in conversation_log:
                label = "👤 User   " if entry["role"] == "user" else "🤖 Agent  "
                logger.info(f"[{entry['time']}] {label}: {entry['text']}")
            logger.info("="*60)
            logger.info(f"Transcript saved to: {transcript_file}")
            logger.info(f"Audio recorded to: {audio_file}")
        await ctx.room.disconnect()


if __name__ == "__main__":
    cli.run_app(server)