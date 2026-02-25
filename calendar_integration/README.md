# Google Calendar Integration

This module provides comprehensive appointment scheduling functionality with Google Calendar for the AI Voice Agent.

## Features

- ✅ **Google Calendar Only**: All appointments stored exclusively in Google Calendar (no database)
- ✅ **Single Appointment Type**: Simple 2-hour appointments with 1-hour breaks
- ✅ **Availability Checking**: Real-time availability validation
- ✅ **Business Hours Management**: Configurable business hours and holidays
- ✅ **AI Voice Integration**: Seamless scheduling via voice conversation
- ✅ **Auto-Suggest**: Automatically suggests next available slot if requested time is unavailable
- ✅ **REST API**: API endpoints for appointment management

## Architecture

```
calendar_integration/
├── __init__.py                    # Package initialization
├── google_calendar.py             # Google Calendar API integration
├── availability_checker.py        # Business hours and availability logic
├── appointment_manager.py         # Central appointment coordination
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── SETUP_GUIDE.md                # Setup and configuration guide
```

## Modules

### 1. Google Calendar Service (`google_calendar.py`)
Manages all Google Calendar API operations:
- Create, update, delete calendar events
- Check availability and conflicts
- List events within time ranges
- Generate free time slots

### 2. Availability Checker (`availability_checker.py`)
Handles business logic for scheduling:
- Business hours configuration
- Holiday/exclusion date management
- Time slot generation
- Appointment validation

### 3. Appointment Manager (`appointment_manager.py`)
Central coordinator for appointments:
- Task type configuration
- Appointment creation workflow
- Calendar synchronization
- Confirmation message generation

## Usage

### Python API

```python
from calendar_integration import (
    get_appointment_manager,
    get_calendar_service,
    get_availability_checker
)

# Get manager instance
manager = get_appointment_manager()

# Check availability
slots = manager.get_available_slots(
    date=datetime(2024, 3, 15),
    task_type="consultation",
    count=5
)

# Create appointment
result = manager.create_appointment(
    user_id=1,
    user_name="John Doe",
    user_email="john@example.com",
    task_type="consultation",
    start_time=datetime(2024, 3, 15, 14, 0),
    notes="Initial consultation"
)
```

### REST API Endpoints

See API documentation in `/api/v1/appointments`:

- `GET /api/v1/appointments/task-types` - List available appointment types
- `POST /api/v1/appointments/availability` - Check available slots
- `GET /api/v1/appointments/suggestions` - Get suggested times
- `POST /api/v1/appointments/` - Create appointment
- `GET /api/v1/appointments/` - List appointments
- `GET /api/v1/appointments/{id}` - Get appointment details
- `PUT /api/v1/appointments/{id}` - Update appointment
- `DELETE /api/v1/appointments/{id}` - Cancel appointment

### Voice Conversation

The AI agent can schedule appointments through natural conversation:

```
User: "I'd like to schedule an appointment"
AI: "I'd be happy to help! What type of appointment would you like to schedule? 
     We offer consultations, interviews, meetings, demos, and support sessions."
User: "A consultation please"
AI: "Great! When would you like to schedule your consultation?"
User: "Tomorrow at 2 PM"
AI: [checks availability] "I have availability tomorrow at 2:00 PM. May I have 
     your name and email to confirm the booking?"
User: "John Doe, john@example.com"
AI: "Perfect! Let me confirm your consultation:
     - Type: Consultation
     - Date: March 15, 2024
     - Time: 2:00 PM
     - Duration: 60 minutes
     Is this correct?"
User: "Yes"
AI: [creates appointment] "Your appointment is confirmed! You'll receive a 
     calendar invitation at john@example.com."
```

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Google Calendar Configuration
GOOGLE_CALENDAR_CREDENTIALS=/path/to/service-account-key.json
GOOGLE_CALENDAR_ID=primary
```

### Business Hours

Configure in `availability_checker.py`:

```python
business_hours = {
    0: [(time(9, 0), time(17, 0))],   # Monday: 9 AM - 5 PM
    1: [(time(9, 0), time(17, 0))],   # Tuesday: 9 AM - 5 PM
    2: [(time(9, 0), time(17, 0))],   # Wednesday: 9 AM - 5 PM
    3: [(time(9, 0), time(17, 0))],   # Thursday: 9 AM - 5 PM
    4: [(time(9, 0), time(17, 0))],   # Friday: 9 AM - 5 PM
}
```

### Task Types

Configure in `appointment_manager.py`:

```python
task_types = {
    "consultation": {
        "name": "Consultation",
        "duration_minutes": 60,
        "description": "Standard consultation session"
    },
    # Add more types...
}
```

## Database Schema

### Appointments Table

```sql
CREATE TABLE appointments (
    id BIGSERIAL PRIMARY KEY,
    appointment_id TEXT UNIQUE NOT NULL,
    user_id BIGINT REFERENCES users(id),
    user_name TEXT NOT NULL,
    user_email TEXT NOT NULL,
    user_phone TEXT,
    task_type TEXT NOT NULL,
    task_name TEXT NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    duration_minutes INTEGER NOT NULL,
    status TEXT NOT NULL,
    calendar_event_id TEXT,
    calendar_link TEXT,
    notes TEXT,
    cancellation_reason TEXT,
    interaction_id BIGINT REFERENCES interactions(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Testing

### Check Calendar Connection

```python
from calendar_integration import get_calendar_service

service = get_calendar_service()
if service.is_available():
    print("✅ Google Calendar connected")
else:
    print("❌ Calendar not available")
```

### Test Appointment Creation

```bash
curl -X POST http://localhost:8000/api/v1/appointments/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "task_type": "consultation",
    "start_time": "2024-03-15T14:00:00",
    "user_name": "Test User",
    "user_email": "test@example.com"
  }'
```

## Error Handling

The system gracefully degrades when Google Calendar is unavailable:
- Appointments can still be created and stored in database
- Availability checks use business hours logic
- Calendar sync will be attempted when service is restored

## Logging

All appointment operations are logged:

```
✅ [APPOINTMENT] Appointment created: abc-123-def
✅ [CALENDAR] Calendar event created: evt_xyz789
⚠️  [CALENDAR] Calendar service not available
```

## Support

For issues or questions:
1. Check the setup guide: `SETUP_GUIDE.md`
2. Review logs in the application output
3. Verify Google Calendar credentials
4. Contact development team

## License

Internal use only
