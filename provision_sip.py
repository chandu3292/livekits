import os
import asyncio
from livekit import api
from dotenv import load_dotenv

load_dotenv()

async def setup_sip():
    url = "http://localhost:7880"
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    lkapi = api.LiveKitAPI(url, api_key, api_secret)

    # -----------------------------
    # CLEAN OLD TRUNKS (inbound + outbound)
    # -----------------------------
    print("Cleaning old inbound trunks...")
    try:
        trunks = await lkapi.sip.list_inbound_trunk(
            api.ListSIPInboundTrunkRequest()
        )
        for t in trunks.items:
            print(f"  Deleting inbound trunk: {t.sip_trunk_id}")
            await lkapi.sip.delete_trunk(
                api.DeleteSIPTrunkRequest(sip_trunk_id=t.sip_trunk_id)
            )
    except Exception as e:
        print(f"Inbound trunk cleanup error: {e}")

    print("Cleaning old outbound trunks...")
    try:
        out_trunks = await lkapi.sip.list_outbound_trunk(
            api.ListSIPOutboundTrunkRequest()
        )
        for t in out_trunks.items:
            print(f"  Deleting outbound trunk: {t.sip_trunk_id}")
            await lkapi.sip.delete_trunk(
                api.DeleteSIPTrunkRequest(sip_trunk_id=t.sip_trunk_id)
            )
    except Exception as e:
        print(f"Outbound trunk cleanup error: {e}")

    # -----------------------------
    # CLEAN OLD DISPATCH RULES
    # -----------------------------
    print("?? Cleaning old dispatch rules...")
    try:
        rules = await lkapi.sip.list_dispatch_rule(
            api.ListSIPDispatchRuleRequest()
        )
        for r in rules.items:
            print(f"?? Deleting rule: {r.sip_dispatch_rule_id}")
            await lkapi.sip.delete_dispatch_rule(
                api.DeleteSIPDispatchRuleRequest(
                    sip_dispatch_rule_id=r.sip_dispatch_rule_id
                )
            )
    except Exception as e:
        print(f"Dispatch cleanup error: {e}")

    # -----------------------------
    # CREATE SINGLE TRUNK (FOR VOBIZ PSTN)
    # -----------------------------
    print("Creating Vobiz PSTN trunk...")
    trunk = await lkapi.sip.create_inbound_trunk(
        api.CreateSIPInboundTrunkRequest(
            trunk=api.SIPInboundTrunkInfo(
                name="VobizInboundTrunk",
                numbers=["+911171366927"],  # Vobiz phone number
                allowed_addresses=["*"]
            )
        )
    )
    print(f"Trunk created: {trunk.sip_trunk_id}")

    # -----------------------------
    # CREATE VOIP EXTENSION TRUNKS (100, 101, 102) WITH AUTH
    # -----------------------------
    # VoIP phones authenticate using the LiveKit API key as username and API secret as password
    voip_extensions = ["100", "101", "102"]

    voip_trunk_ids = []
    for ext in voip_extensions:
        print(f"Creating VoIP trunk for extension {ext}...")
        voip_trunk = await lkapi.sip.create_inbound_trunk(
            api.CreateSIPInboundTrunkRequest(
                trunk=api.SIPInboundTrunkInfo(
                    name=f"VoIP_Ext{ext}",
                    numbers=[ext],               # Extension number (100/101/102)
                    allowed_addresses=["*"],      # Accept from any IP (VoIP phones)
                    auth_username=api_key,        # SIP username = LIVEKIT_API_KEY
                    auth_password=api_secret,     # SIP password = LIVEKIT_API_SECRET
                )
            )
        )
        print(f"  Trunk {ext} created: {voip_trunk.sip_trunk_id}")
        voip_trunk_ids.append((ext, voip_trunk.sip_trunk_id))

    # -----------------------------
    # CREATE DISPATCH RULES
    # -----------------------------

    # PSTN individual rule (multi-user support via Vobiz trunk)
    print("Creating PSTN individual dispatch rule...")
    await lkapi.sip.create_dispatch_rule(
        api.CreateSIPDispatchRuleRequest(
            rule=api.SIPDispatchRule(
                dispatch_rule_individual=api.SIPDispatchRuleIndividual(
                    room_prefix="call_"
                )
            ),
            trunk_ids=[trunk.sip_trunk_id]
        )
    )
    print("PSTN dispatch rule created.")

    # VoIP extension rules — each extension routes to its own fixed room
    for ext_num, trunk_id in voip_trunk_ids:
        room_name = f"ext_{ext_num}_room"
        print(f"Creating dispatch rule for extension {ext_num} -> room '{room_name}'...")
        await lkapi.sip.create_dispatch_rule(
            api.CreateSIPDispatchRuleRequest(
                rule=api.SIPDispatchRule(
                    dispatch_rule_direct=api.SIPDispatchRuleDirect(
                        room_name=room_name
                    )
                ),
                trunk_ids=[trunk_id]
            )
        )
        print(f"  Dispatch rule created for extension {ext_num}.")

    # -----------------------------
    # CREATE OUTBOUND TRUNK (VOBIZ)
    # -----------------------------
    vobiz_host = os.getenv("VOBIZ_SIP_HOST")
    vobiz_user = os.getenv("VOBIZ_SIP_USERNAME")
    vobiz_pass = os.getenv("VOBIZ_SIP_PASSWORD")

    if vobiz_host and vobiz_user and vobiz_pass:
        print("Creating Vobiz outbound trunk...")
        outbound_trunk = await lkapi.sip.create_outbound_trunk(
            api.CreateSIPOutboundTrunkRequest(
                trunk=api.SIPOutboundTrunkInfo(
                    name="VobizOutboundTrunk",
                    address=vobiz_host,
                    numbers=["+911171366927"],
                    auth_username=vobiz_user,
                    auth_password=vobiz_pass,
                )
            )
        )
        print(f"Outbound trunk created: {outbound_trunk.sip_trunk_id}")
    else:
        print("Skipping outbound trunk (VOBIZ_SIP_* not set in .env)")

    await lkapi.aclose()
    print("\nSIP setup complete.")
    print("\nVoIP credentials summary (same for all extensions):")
    print(f"  Username : {api_key}  (LIVEKIT_API_KEY)")
    print(f"  Password : {api_secret}  (LIVEKIT_API_SECRET)")
    for ext in voip_extensions:
        print(f"  Extension {ext} -> room: ext_{ext}_room")


if __name__ == "__main__":
    asyncio.run(setup_sip())