import os
from datetime import time
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Timezone Configuration
# Default to 330 minutes (5.5 hours) for IST if not specified
DEFAULT_TIMEZONE_OFFSET = int(os.getenv('DEFAULT_TIMEZONE_OFFSET', '330'))

# Business Hours Configuration
# Mapping day of week (0-6) to list of time ranges
DEFAULT_BUSINESS_HOURS: Dict[int, List[Tuple[time, time]]] = {
    0: [(time(9, 0), time(17, 0))],   # Monday
    1: [(time(9, 0), time(17, 0))],   # Tuesday
    2: [(time(9, 0), time(17, 0))],   # Wednesday
    3: [(time(9, 0), time(17, 0))],   # Thursday
    4: [(time(9, 0), time(17, 0))],   # Friday
}

# Appointment Settings
DEFAULT_SLOT_DURATION_MINUTES = 30
DEFAULT_BUFFER_MINUTES = 0
MAX_ADVANCE_DAYS = 10
SLOT_SEARCH_INTERVAL_MINUTES = 30

# Task Types Configuration
TASK_TYPES = {
    "appointment": {
        "name": "Appointment",
        "duration_minutes": 30,  # 0.5 hour
        "description": "Standard appointment session"
    }
}

# Search and Suggestions Settings
DEFAULT_SUGGESTIONS_COUNT = 10
API_MAX_SLOTS_COUNT = 10
NEXT_AVAILABLE_SEARCH_DAYS = 14  # For finding next available slot
MAX_NEXT_DAYS_SEARCH = 30        # For finding next available date
LATE_HOUR_UTC = 16               # 4 PM UTC is already past business hours in IST (9:30 PM)
