"""
Agent Personas Configuration
Defines available AI agent personas with voice mappings and language assignments.

Voice IDs sourced from Cartesia API (Sonic-3).
"""

DEFAULT_PERSONA_ID = "sophia"

PERSONAS = {
    "sophia": {
        "id": "sophia",
        "name": "Sophia",
        "language": "en",
        "description": "Professional and friendly (English)",
        "voice": "79a125e8-cd45-4c13-8a67-188112f4dd22",  # Default Cartesia English female
    },
    "alex": {
        "id": "alex",
        "name": "Alex",
        "language": "en",
        "description": "Casual and approachable (English)",
        "voice": "1259b7e3-cb8a-43df-9446-30971a46b8b0",  # Devansh - Warm Support Agent (Indian English male)
    },
    "maya": {
        "id": "maya",
        "name": "Maya",
        "language": "en",
        "description": "Warm Indian English accent (English)",
        "voice": "f8f5f1b2-f02d-4d8e-a40d-fd850a487b3d",  # Kiara - Joyful Woman (Indian accent)
    },
    "priya": {
        "id": "priya",
        "name": "Priya",
        "language": "te",
        "description": "Helpful assistant (Telugu)",
        "voice": "330c4fa0-1da3-4c55-8e97-951bfd724e20",  # Sarika - Calm Spirit (Telugu)
    },
    "arjun": {
        "id": "arjun",
        "name": "Arjun",
        "language": "hi",
        "description": "Friendly assistant (Hindi)",
        "voice": "791d5162-d5eb-40f0-8189-f19db44611d8",  # Ayush - Friendly Neighbor (Hindi male)
    },
}


def list_personas():
    """Return list of all available personas."""
    return [
        {
            "id": p["id"],
            "name": p["name"],
            "language": p["language"],
            "description": p["description"],
        }
        for p in PERSONAS.values()
    ]


def get_persona(persona_id: str):
    """Get persona by ID. Returns None if not found."""
    return PERSONAS.get(persona_id)


def get_voice_id(persona_id: str) -> str:
    """Get the voice ID for a persona."""
    persona = PERSONAS.get(persona_id, PERSONAS[DEFAULT_PERSONA_ID])
    return persona["voice"]
