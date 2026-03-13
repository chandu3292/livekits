"""
Outbound call dialer — creates a SIP participant that dials out via Vobiz
and joins a LiveKit room so the AI agent can talk to the callee.
"""
import os
import uuid
import logging
from livekit import api
from .trunk import ensure_outbound_trunk

logger = logging.getLogger("outbound_calls")


async def make_outbound_call(phone_number: str) -> dict:
    """
    Dial a phone number via the Vobiz outbound trunk.
    Returns dict with room_name, participant_identity, and sip_participant info.
    """
    url = "http://localhost:7880"
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    lkapi = api.LiveKitAPI(url, api_key, api_secret)

    try:
        trunk_id = await ensure_outbound_trunk(lkapi)

        room_name = f"outbound_{uuid.uuid4().hex[:8]}"
        participant_identity = f"sip_outbound_{phone_number}"

        sip_participant = await lkapi.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                sip_trunk_id=trunk_id,
                sip_call_to=phone_number,
                room_name=room_name,
                participant_identity=participant_identity,
                participant_name=f"Call to {phone_number}",
                participant_attributes={
                    "call_direction": "outbound",
                    "phone_number": phone_number,
                },
                play_ringtone=True,
            ),
            timeout=30,
        )

        logger.info(f"Outbound call started: {phone_number} in room {room_name}")
        return {
            "status": "success",
            "room_name": room_name,
            "participant_identity": participant_identity,
            "sip_call_id": sip_participant.sip_call_id if hasattr(sip_participant, 'sip_call_id') else "",
        }
    except Exception as e:
        logger.error(f"Outbound call failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        await lkapi.aclose()
