import os
import asyncio
from livekit import api
from dotenv import load_dotenv

load_dotenv()

async def setup_sip():
    # Use localhost for provisioning
    url = "http://localhost:7880"
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    # Initialize LiveKit API
    lkapi = api.LiveKitAPI(url, api_key, api_secret)

    # 1. Create a SIP Inbound Trunk
    print("⏳ Creating Inbound Trunk...")
    try:
        # Replace YOUR_TWILIO_NUMBER with your actual E.164 Twilio number (e.g., "+1234567890")
        trunk = await lkapi.sip.create_inbound_trunk(
            api.CreateSIPInboundTrunkRequest(
                trunk=api.SIPInboundTrunkInfo(
                    name="Twilio Inbound",
                    numbers=["+19378802983"], 
                    allowed_addresses=["*"],
                    auth_username="devkey",
                    auth_password="devsecret_1234567890_abcdefghijklmnopqrstuvwxyz"
                )
            )
        )
        print(f"✅ SIP Inbound Trunk Created: {trunk.sip_trunk_id}")
    except Exception as e:
        print(f"❌ Inbound Trunk Error: {e}")

    # 2. Create a SIP Dispatch Rule
    print("⏳ Creating Dispatch Rule...")
    try:
        # According to LiveKit Python SDK / Protobuf, the field is 'dispatch_rule_direct'
        dispatch = await lkapi.sip.create_sip_dispatch_rule(
            api.CreateSIPDispatchRuleRequest(
                rule=api.SIPDispatchRule(
                    dispatch_rule_direct=api.SIPDispatchRuleDirect(
                        room_name="phone_call_room",
                        pin=""
                    )
                )
            )
        )
        print(f"✅ SIP Dispatch Rule Created: {dispatch.sip_dispatch_rule_id}")
    except Exception as e:
        print(f"❌ Dispatch Rule Error: {e}")

    print("\n🚀 LiveKit SIP Bridge Ready.")
    await lkapi.aclose()

if __name__ == "__main__":
    asyncio.run(setup_sip())
