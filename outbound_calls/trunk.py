"""
Outbound SIP trunk management via Vobiz.
Ensures a single outbound trunk exists and caches its ID.
"""
import os
import logging
from livekit import api

logger = logging.getLogger("outbound_calls")

_outbound_trunk_id: str | None = None


async def ensure_outbound_trunk(lkapi: api.LiveKitAPI) -> str:
    """Create the Vobiz outbound trunk if it doesn't exist. Returns trunk ID."""
    global _outbound_trunk_id
    if _outbound_trunk_id:
        return _outbound_trunk_id

    # Check if one already exists
    existing = await lkapi.sip.list_outbound_trunk(
        api.ListSIPOutboundTrunkRequest()
    )
    for t in existing.items:
        if t.name == "VobizOutboundTrunk":
            _outbound_trunk_id = t.sip_trunk_id
            logger.info(f"Reusing existing outbound trunk: {_outbound_trunk_id}")
            return _outbound_trunk_id

    # Create new outbound trunk
    sip_host = os.getenv("VOBIZ_SIP_HOST")
    sip_user = os.getenv("VOBIZ_SIP_USERNAME")
    sip_pass = os.getenv("VOBIZ_SIP_PASSWORD")

    if not all([sip_host, sip_user, sip_pass]):
        raise ValueError("Missing VOBIZ_SIP_HOST / VOBIZ_SIP_USERNAME / VOBIZ_SIP_PASSWORD in .env")

    trunk = await lkapi.sip.create_outbound_trunk(
        api.CreateSIPOutboundTrunkRequest(
            trunk=api.SIPOutboundTrunkInfo(
                name="VobizOutboundTrunk",
                address=sip_host,
                numbers=["+911171366927"],
                auth_username=sip_user,
                auth_password=sip_pass,
            )
        )
    )
    _outbound_trunk_id = trunk.sip_trunk_id
    logger.info(f"Created outbound trunk: {_outbound_trunk_id}")
    return _outbound_trunk_id


def get_outbound_trunk_id() -> str | None:
    return _outbound_trunk_id
