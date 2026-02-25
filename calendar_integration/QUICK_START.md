# Quick Start Guide - Appointment Scheduling

Get the appointment scheduling system up and running in minutes!

## 🚀 Quick Setup (3 minutes)

### Step 1: Install Dependencies (1 min)
```bash
cd realtime-voice/backend
pip install -r requirements.txt
```

### Step 2: Configure Google Calendar (2 min)
Add to your `.env` file:
```bash
# Required: Google Calendar Integration
GOOGLE_CALENDAR_CREDENTIALS=/path/to/service-account-key.json
GOOGLE_CALENDAR_ID=primary
```

**Important**: This system uses **Google Calendar only** for storage. No database setup required!

To get your service account credentials:
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a service account
3. Download the JSON key file
4. Share your Google Calendar with the service account email

### Step 3: Start Server (1 min)
```bash
cd steel
python main.py
```

Server starts on `http://localhost:8000`

---

## ✅ Test It's Working

### Test 1: Check API Health
```bash
curl http://localhost:8000/health
```

Expected: `{"status": "healthy", ...}`

### Test 2: Get Appointment Info
```bash
curl http://localhost:8000/api/v1/appointments/info
```

Expected: List of available appointment types

### Test 3: Check Availability
```bash
curl -X POST http://localhost:8000/api/v1/appointments/availability \
  -H "Content-Type: application/json" \
  -d '{"task_type": "consultation", "date": "2026-02-11"}'
```

Expected: List of available time slots

---

## 🎤 Voice Testing

### Test via WebSocket

1. Connect to WebSocket: `ws://localhost:8000/ws?user_id=1`
2. Say: **"I'd like to schedule an appointment"**
3. Follow the AI's prompts:
   - Select appointment type
   - Choose date and time
   - Provide contact information
   - Confirm booking

### Expected Flow:
```
User: "Schedule an appointment"
AI: "I can help with that! We offer consultations, interviews, 
     meetings, demos, and support sessions. Which would you like?"
User: "A consultation"
AI: "Great! When would you like to schedule your consultation?"
User: "Tomorrow at 2 PM"
AI: "Let me check availability for you..."
AI: "I have available slots at 2:00 PM, 2:30 PM, 3:00 PM. 
     Which time works best?"
User: "2 PM"
AI: "Perfect! May I have your name and email address?"
User: "John Doe, john@example.com"  
AI: "Just to confirm: Consultation on [date] at 2:00 PM for John Doe. 
     Is this correct?"
User: "Yes"
AI: "Your appointment is confirmed! You'll receive a confirmation 
     email at john@example.com."
```

---

## 📱 REST API Testing

### Create an Appointment
```bash
curl -X POST http://localhost:8000/api/v1/appointments/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "task_type": "consultation",
    "start_time": "2026-02-11T14:00:00",
    "user_name": "John Doe",
    "user_email": "john@example.com",
    "user_phone": "+1234567890",
    "notes": "First consultation"
  }'
```

### List My Appointments
```bash
curl http://localhost:8000/api/v1/appointments/ \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Get Upcoming Appointments
```bash
curl http://localhost:8000/api/v1/appointments/upcoming \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## 🔧 Configuration Options

### Customize Appointment Types

Edit `steel/calendar_integration/appointment_manager.py`:

```python
self.task_types = {
    "consultation": {
        "name": "Consultation",
        "duration_minutes": 60,
        "description": "Standard consultation"
    },
    "quick_call": {  # Add new type
        "name": "Quick Call",
        "duration_minutes": 15,
        "description": "Brief 15-minute call"
    }
}
```

### Customize Business Hours

Edit `steel/calendar_integration/availability_checker.py`:

```python
self.business_hours = {
    0: [(time(9, 0), time(17, 0))],   # Monday: 9 AM - 5 PM
    1: [(time(9, 0), time(17, 0))],   # Tuesday: 9 AM - 5 PM
    2: [(time(9, 0), time(17, 0))],   # Wednesday: 9 AM - 5 PM
    3: [(time(9, 0), time(17, 0))],   # Thursday: 9 AM - 5 PM
    4: [(time(9, 0), time(12, 0))],   # Friday: 9 AM - 12 PM (half day)
    5: [(time(10, 0), time(14, 0))],  # Saturday: 10 AM - 2 PM (optional)
}
```

---

## 🎯 What Works Without Google Calendar

- ✅ Appointment creation and storage
- ✅ Availability checking (based on business hours)
- ✅ REST API operations
- ✅ Voice conversation scheduling
- ✅ Database persistence
- ✅ Appointment management (view, update, cancel)

## 🎯 What Requires Google Calendar

- 📅 Calendar event creation in Google Calendar
- 📧 Automatic email invitations
- 🔗 Calendar links in confirmations
- 🔄 Real-time calendar conflict checking
- 📱 Mobile calendar notifications

**Bottom Line**: System is fully functional without Google Calendar!

---

## 🐛 Troubleshooting

### "Table appointments does not exist"
```bash
# Run migration script
cd steel/calendar_integration
python create_appointments_table.py
```

### "No available slots found"
- Check that business hours are configured
- Verify the date is a working day (not weekend)
- Try a different date

### "Calendar service not available"
- This is OK! System works without calendar
- To enable: See `SETUP_GUIDE.md` for Google Calendar setup
- Appointments still work, just won't sync to Google Calendar

### Server won't start
```bash
# Check if all dependencies are installed
pip install -r requirements.txt

# Check database connection
# Verify DATABASE_URL in .env
```

---

## 📚 Documentation

- **Full Setup**: See `calendar_integration/SETUP_GUIDE.md`
- **API Reference**: See `calendar_integration/README.md`
- **Implementation Details**: See `steel/APPOINTMENT_IMPLEMENTATION.md`

---

## 🎉 Success Indicators

You'll know it's working when:

1. ✅ Server starts without errors
2. ✅ Health check returns "healthy"
3. ✅ `/api/v1/appointments/task-types` returns list of types
4. ✅ Voice agent responds to "schedule an appointment"
5. ✅ Appointments are saved to database
6. ✅ You can view appointments via API

---

## Next Steps

1. ✅ Test voice scheduling
2. ✅ Test REST API endpoints
3. ✅ Review appointment database records
4. 📅 (Optional) Set up Google Calendar integration
5. 🎨 (Optional) Build frontend UI
6. 📧 (Optional) Add email notifications
7. 📊 (Optional) Add analytics dashboard

---

## Support

Need help? Check:
- Error logs in terminal
- Database connection settings
- Environment variables in `.env`

**Ready to go!** Start testing appointments! 🚀
