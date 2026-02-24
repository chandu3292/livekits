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
    # CLEAN OLD TRUNKS
    # -----------------------------
    print("?? Cleaning old trunks...")
    try:
        trunks = await lkapi.sip.list_inbound_trunk(
            api.ListSIPInboundTrunkRequest()
        )
        for t in trunks.items:
            print(f"?? Deleting trunk: {t.sip_trunk_id}")
            await lkapi.sip.delete_trunk(
                api.DeleteSIPTrunkRequest(sip_trunk_id=t.sip_trunk_id)
            )
    except Exception as e:
        print(f"Trunk cleanup error: {e}")

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
    # CREATE SINGLE TRUNK (NO NUMBER FILTERING)
    # -----------------------------
    print("? Creating trunk...")
    trunk = await lkapi.sip.create_inbound_trunk(
        api.CreateSIPInboundTrunkRequest(
            trunk=api.SIPInboundTrunkInfo(
                name="WebSIPTrunk",
                numbers=[],  # IMPORTANT: empty
                allowed_addresses=["*"],
                auth_username=api_key,
                auth_password=api_secret
            )
        )
    )

    print(f"? Trunk created: {trunk.sip_trunk_id}")

    # -----------------------------
    # CREATE SINGLE DISPATCH RULE
    # -----------------------------
    print("? Creating dispatch rule...")
    await lkapi.sip.create_dispatch_rule(
        api.CreateSIPDispatchRuleRequest(
            rule=api.SIPDispatchRule(
                dispatch_rule_direct=api.SIPDispatchRuleDirect(
                    room_name="phone_call_room"
                )
            ),
            trunk_ids=[trunk.sip_trunk_id]
        )
    )

    print("? Dispatch rule created.")
    await lkapi.aclose()
    print("\n?? SIP setup complete.")


if __name__ == "__main__":
    asyncio.run(setup_sip())