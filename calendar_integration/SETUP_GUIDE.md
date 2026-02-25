# Google Calendar Integration - Setup Guide

This guide will walk you through setting up Google Calendar integration for the appointment scheduling system.

## Prerequisites

- Python 3.8 or higher
- Google Cloud Platform account
- Administrator access to a Google Calendar

**Note**: No database required! This system stores all appointments in Google Calendar only.

## Step 1: Install Dependencies

Install the required Python packages:

```bash
cd steel/calendar_integration
pip install -r requirements.txt
```

## Step 2: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Note your Project ID

## Step 3: Enable Google Calendar API

1. In Google Cloud Console, navigate to **APIs & Services** > **Library**
2. Search for "Google Calendar API"
3. Click on it and press **ENABLE**

## Step 4: Create Service Account

### 4.1 Create the Account

1. Go to **APIs & Services** > **Credentials**
2. Click **CREATE CREDENTIALS** > **Service Account**
3. Fill in the details:
   - **Service account name**: `appointment-scheduler`
   - **Service account ID**: `appointment-scheduler`
   - **Description**: `Service account for appointment scheduling`
4. Click **CREATE AND CONTINUE**

### 4.2 Grant Permissions (Optional)

You can skip the permissions step or add roles if needed.
Click **CONTINUE** then **DONE**

### 4.3 Create Service Account Key

1. Click on the newly created service account
2. Go to the **KEYS** tab
3. Click **ADD KEY** > **Create new key**
4. Select **JSON** format
5. Click **CREATE**
6. A JSON file will be downloaded - **save it securely!**

## Step 5: Share Calendar with Service Account

The service account needs access to your Google Calendar:

1. Open [Google Calendar](https://calendar.google.com/)
2. Find the calendar you want to use (or create a new one)
3. Click the three dots next to the calendar name
4. Select **Settings and sharing**
5. Scroll to **Share with specific people**
6. Click **Add people**
7. Enter the service account email (found in the JSON file, looks like: `appointment-scheduler@your-project.iam.gserviceaccount.com`)
8. Set permission to **Make changes to events**
9. Click **Send**

## Step 6: Configure Environment Variables

Add the following to your `.env` file:

```bash
# Google Calendar Configuration
GOOGLE_CALENDAR_CREDENTIALS=/absolute/path/to/your-service-account-key.json
GOOGLE_CALENDAR_ID=primary

# Or use a specific calendar ID (found in Calendar Settings > Integrate calendar)
# GOOGLE_CALENDAR_ID=your-calendar-id@group.calendar.google.com
```

### Finding Calendar ID

To use a specific calendar (not your primary):

1. Go to Google Calendar Settings
2. Select the calendar from the left sidebar
3. Scroll to **Integrate calendar**
4. Copy the **Calendar ID**
5. Use this ID in your `.env` file

## Step 7: Verify Setup

Test that Google Calendar is configured correctly:

```bash
# From the steel directory
cd realtime-voice/backend/steel
python -c "from calendar_integration import get_calendar_service; service = get_calendar_service(); print('✅ Google Calendar configured!' if service.is_available() else '❌ Not configured')"
```

You should see: `✅ Google Calendar configured!`
    status TEXT NOT NULL DEFAULT 'confirmed',
    calendar_event_id TEXT,


## Step 8: Configure Business Hours

Edit [calendar_integration/availability_checker.py](calendar_integration/availability_checker.py) to set your business hours:

```python
# Default business hours: Monday-Friday, 9 AM - 5 PM
business_hours = {
    0: [(time(9, 0), time(17, 0))],   # Monday
    1: [(time(9, 0), time(17, 0))],   # Tuesday
    2: [(time(9, 0), time(17, 0))],   # Wednesday
    3: [(time(9, 0), time(17, 0))],   # Thursday
    4: [(time(9, 0), time(17, 0))],   # Friday
    # Saturday and Sunday are closed by default (not in dict)
}

# For multiple time slots per day:
business_hours = {
    0: [(time(9, 0), time(12, 0)), (time(13, 0), time(17, 0))],  # 9-12, 1-5
}
```

## Step 9: Configure Appointment Duration

The system is configured with:
- **Appointment Duration**: 2 hours
- **Break Between Appointments**: 1 hour

To change these, edit [calendar_integration/appointment_manager.py](calendar_integration/appointment_manager.py):

```python
task_types = {
    "appointment": {
        "name": "Appointment",
        "duration_minutes": 120,  # Change to desired duration
        "description": "Standard appointment"
    }
}

# And in availability_checker.py:
buffer_minutes = 60  # Change to desired break time
```

## Step 10: Test the Setup

### Test 1: Check Calendar Connection

Create a test script `test_calendar.py`:

```python
import sys
sys.path.append('steel')

from calendar_integration import get_calendar_service

service = get_calendar_service()
if service.is_available():
    print("✅ Google Calendar is connected!")
    
    # List upcoming events
    from datetime import datetime
    events = service.list_events(time_min=datetime.now())
    print(f"Found {len(events)} upcoming events")
else:
    print("❌ Calendar connection failed")
    print("Check your credentials path and calendar sharing settings")
```

Run it:
```bash
python test_calendar.py
```

### Test 2: Create Test Appointment

```python
from datetime import datetime, timedelta
from calendar_integration import get_appointment_manager

manager = get_appointment_manager()

# Create a test appointment
result = manager.create_appointment(
    user_id=1,
    user_name="Test User",
    user_email="test@example.com",
    task_type="appointment",
    start_time=datetime.now() + timedelta(days=1, hours=2),
    notes="Test appointment"
)

if result["success"]:
    print("✅ Appointment created successfully!")
    print(result["appointment"]["appointment_id"])
else:
    print("❌ Failed:", result["error"])
```

### Test 3: Test REST API

Start your server:
```bash
cd steel
python main.py
```

Test the API:
```bash
# Get task types
curl http://localhost:8000/api/v1/appointments/task-types

# Check availability
curl -X POST http://localhost:8000/api/v1/appointments/availability \
  -H "Content-Type: application/json" \
  -d '{"task_type": "consultation", "date": "2024-03-15"}'
```

## Troubleshooting

### Issue: "Calendar service not available"

**Cause**: Issues with credentials or calendar access

**Solutions**:
1. Verify `GOOGLE_CALENDAR_CREDENTIALS` path is correct and absolute
2. Check that the JSON file is valid (not corrupted)
3. Ensure calendar is shared with service account email
4. Verify Google Calendar API is enabled in your project

### Issue: "403 Forbidden" error

**Cause**: Service account doesn't have calendar access

**Solutions**:
1. Re-share the calendar with the service account
2. Set permission to "Make changes to events"
3. Wait a few minutes for permissions to propagate

### Issue: "Appointments created but not showing in Calendar"

**Cause**: Using wrong calendar ID

**Solutions**:
1. Check `GOOGLE_CALENDAR_ID` in `.env`
2. Use 'primary' for main calendar
3. For shared calendars, use the full calendar ID

### Issue: "No available slots found"

**Cause**: Business hours not configured or date is non-working day

**Solutions**:
1. Check business hours configuration
2. Verify the requested date is a configured working day
3. Check for existing appointments blocking slots

## Security Best Practices

1. **Never commit credentials**: Add `*.json` to `.gitignore`
2. **Restrict service account permissions**: Only grant Calendar access
3. **Rotate keys periodically**: Delete old keys from Google Cloud Console
4. **Use environment variables**: Never hardcode credentials
5. **Monitor API usage**: Set up alerts in Google Cloud Console

## Production Deployment

For production environments:

1. **Use secret management**: Store credentials in a secure vault (e.g., AWS Secrets Manager, Azure Key Vault)
2. **Enable API quotas**: Set reasonable limits in Google Cloud Console
3. **Set up monitoring**: Track API calls, errors, and performance
4. **Configure backup calendar**: Have a fallback calendar for redundancy
5. **Regular backups**: Export appointment data regularly

## Calendar ID Reference

- **Personal calendar**: `primary`
- **Shared calendar**: `calendar-id@group.calendar.google.com`
- **Resource calendar**: `resource-id@resource.calendar.google.com`

## Additional Configuration

### Time Zones

By default, all times are in UTC. To use a different timezone:

```python
from pytz import timezone

# In availability_checker.py
tz = timezone('America/New_York')
local_time = tz.localize(datetime.now())
```

### Holidays

Add company holidays to excluded dates:

```python
from datetime import datetime

checker = get_availability_checker()

# Add US federal holidays
holidays = [
    datetime(2024, 1, 1),   # New Year's Day
    datetime(2024, 7, 4),   # Independence Day
    datetime(2024, 12, 25), # Christmas
]

for holiday in holidays:
    checker.add_excluded_date(holiday)
```

### Email Notifications

Calendar invitations are sent automatically to attendees. To customize:

1. Edit event creation in `google_calendar.py`
2. Modify the `description` field for custom messages
3. Set `sendUpdates` parameter ('all', 'none', 'externalOnly')

## Support and Resources

- [Google Calendar API Documentation](https://developers.google.com/calendar)
- [Service Account Guide](https://cloud.google.com/iam/docs/service-accounts)
- [Python Client Library](https://developers.google.com/calendar/api/quickstart/python)

## Next Steps

After successful setup:

1. ✅ Test voice conversation appointments
2. ✅ Configure frontend appointment UI
3. ✅ Set up appointment reminders
4. ✅ Integrate with email notifications
5. ✅ Add appointment analytics

For questions or issues, contact the development team.
