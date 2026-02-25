"""
Appointment API Routes
REST API endpoints for appointment scheduling and management
No database - uses Google Calendar only
"""

import os
import logging
from typing import Optional
from datetime import datetime, timedelta, timezone, time
from pydantic import BaseModel, EmailStr, Field
from dotenv import load_dotenv

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

# Load environment variables
load_dotenv()

from calendar_integration import (
    AppointmentManager,
    get_appointment_manager,
    GoogleCalendarService,
    get_calendar_service
)
from calendar_integration.constants import (
    DEFAULT_TIMEZONE_OFFSET,
    TASK_TYPES,
    DEFAULT_BUFFER_MINUTES
)

logger = logging.getLogger('APPOINTMENT_API')

router = APIRouter(prefix="/api/v1/appointments", tags=["appointments"])


# ==================== Pydantic Schemas ====================

class CreateAppointmentRequest(BaseModel):
    """Request model for creating an appointment"""
    start_time: str = Field(..., description="Appointment start time (ISO format)")
    user_name: str = Field(..., description="User's full name")
    user_email: EmailStr = Field(..., description="User's email address")
    user_phone: Optional[str] = Field(None, description="User's phone number")
    notes: Optional[str] = Field(None, description="Additional notes")
    timezone_offset: Optional[int] = Field(None, description="Timezone offset in minutes from UTC (e.g., 330 for IST)")


class AvailabilityRequest(BaseModel):
    """Request model for checking availability"""
    date: str = Field(..., description="Date to check (YYYY-MM-DD)")


# ==================== ENDPOINTS ====================

@router.get("/info")
async def get_appointment_info():
    """
    Get appointment information (duration, break time, etc.)
    """
    try:
        duration_min = TASK_TYPES.get("appointment", {}).get("duration_minutes", 120)
        buffer_min = DEFAULT_BUFFER_MINUTES
        return JSONResponse(
            status_code=200,
            content={
                "appointment_duration": f"{duration_min//60} hours",
                "break_between_appointments": f"{buffer_min//60} hour",
                "storage": "Google Calendar",
                "description": f"All appointments are {duration_min//60} hours long with a {buffer_min//60}-hour break between them. Appointments are managed through Google Calendar."
            }
        )
    except Exception as e:
        logger.error(f"❌ Error getting appointment info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/availability")
async def check_availability(request: AvailabilityRequest):
    """
    Check available time slots for a specific date
    """
    try:
        manager = get_appointment_manager()
        
        # Parse date and make it timezone-aware (UTC)
        date = datetime.strptime(request.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
        # Get available slots (using default "appointment" type)
        slots = manager.get_available_slots(date, "appointment", count=10)
        
        # Get offset for local display
        offset = manager.availability_checker.timezone_offset
        duration_min = TASK_TYPES.get("appointment", {}).get("duration_minutes", 120)
        buffer_min = DEFAULT_BUFFER_MINUTES
        
        processed_slots = []
        for slot in slots:
            # Convert UTC start/end to local time for display
            start_local = (slot['start'] + timedelta(minutes=offset)).replace(tzinfo=None)
            end_local = (slot['end'] + timedelta(minutes=offset)).replace(tzinfo=None)
            
            processed_slots.append({
                "start": start_local.isoformat(),
                "end": end_local.isoformat(),
                "formatted": slot['formatted'], 
                "duration": f"{duration_min//60} hours",
                "is_available": True
            })
            
        return JSONResponse(
            status_code=200,
            content={
                "date": request.date,
                "available_slots": processed_slots,
                "timezone_offset": offset,
                "note": f"Each appointment is {duration_min//60} hours with {buffer_min//60}-hour break between appointments"
            }
        )
    except Exception as e:
        logger.error(f"❌ Error checking availability: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/suggestions")
async def get_appointment_suggestions(
    days_ahead: int = Query(7, ge=1, le=30, description="Number of days to look ahead")
):
    """
    Get suggested available appointment times
    """
    try:
        manager = get_appointment_manager()
        
        suggestions = manager.get_appointment_suggestions(
            task_type="appointment",
            days_ahead=days_ahead
        )
        
        return JSONResponse(status_code=200, content=suggestions)
        
    except Exception as e:
        logger.error(f"❌ Error getting suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/")
async def create_appointment(
    request: CreateAppointmentRequest
):
    """
    Create a new appointment in Google Calendar (No authentication required)
    If timezone_offset is provided, the start_time will be treated as local time and converted to UTC.
    Otherwise, it's assumed to be UTC.
    """
    try:
        manager = get_appointment_manager()
        
        # Parse start time
        start_time = datetime.fromisoformat(request.start_time.replace('Z', '+00:00'))
        logger.info(f"📅 Received start_time: {request.start_time} (parsed as: {start_time.isoformat()})")
        
        # Get timezone offset (use provided, or default from env, or assume UTC)
        timezone_offset = request.timezone_offset
        if timezone_offset is None:
            # Try to get default from environment
            default_offset = os.getenv('DEFAULT_TIMEZONE_OFFSET')
            if default_offset:
                timezone_offset = int(default_offset)
                logger.info(f"🌍 Using default timezone offset: {timezone_offset} minutes from .env")
            else:
                logger.warning("⚠️ No timezone offset provided and DEFAULT_TIMEZONE_OFFSET not set in .env")
        else:
            logger.info(f"🌍 Using provided timezone offset: {timezone_offset} minutes")
        
        # If timezone offset is provided, convert local time to UTC
        if timezone_offset is not None:
            # The start_time is in local time, convert to UTC
            # timezone_offset is in minutes (e.g., 330 for IST which is UTC+5:30)
            original_time = start_time
            start_time = start_time - timedelta(minutes=timezone_offset)
            logger.info(f"⏰ Converted local time to UTC: {original_time.isoformat()} -> {start_time.isoformat()}")
        else:
            logger.info(f"⏰ No timezone conversion applied, treating as UTC: {start_time.isoformat()}")
        
        # Create appointment (Google Calendar only)
        result = manager.create_appointment(
            user_id=0,  # No user ID required
            user_name=request.user_name,
            user_email=request.user_email,
            user_phone=request.user_phone,
            task_type="appointment",
            start_time=start_time,
            notes=request.notes
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        appointment_data = result["appointment"]
        
        logger.info(f"✅ Appointment created in Google Calendar: {appointment_data['appointment_id']}")
        
        return JSONResponse(
            status_code=201,
            content={
                "success": True,
                "appointment": appointment_data,
                "message": manager.format_appointment_confirmation(appointment_data),
                "note": "Appointment created in Google Calendar. You will receive a calendar invitation."
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating appointment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/my-appointments")
async def get_my_appointments_by_date_range(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    user_email: Optional[str] = Query(None, description="Filter by user email (optional)")
):
    """
    Get appointments from Google Calendar within a specified date range.
    Returns all calendar events between start_date and end_date.
    Optionally filter by user email.
    """
    try:
        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format. Use YYYY-MM-DD format. Error: {str(e)}"
            )
        
        # Validate date range
        if start_dt > end_dt:
            raise HTTPException(
                status_code=400,
                detail="start_date must be before or equal to end_date"
            )
        
        # Get calendar service
        calendar_service = get_calendar_service()
        
        if not calendar_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Google Calendar service is not available"
            )
        
        # Fetch events from Google Calendar
        events = calendar_service.list_events(
            time_min=start_dt,
            time_max=end_dt,
            max_results=250
        )
        
        # Filter events if user_email is provided
        user_appointments = []
        for event in events:
            # If user_email filter is provided, check if user is involved
            if user_email:
                is_user_event = False
                
                # Check creator
                creator_email = event.get('creator', {}).get('email', '')
                if creator_email.lower() == user_email.lower():
                    is_user_event = True
                
                # Check attendees
                if not is_user_event:
                    attendees = event.get('attendees', [])
                    for attendee in attendees:
                        if attendee.get('email', '').lower() == user_email.lower():
                            is_user_event = True
                            break
                
                if not is_user_event:
                    continue
            
            # Parse event data
            start = event.get('start', {})
            end = event.get('end', {})
            
            # Handle both dateTime and date formats
            start_time = start.get('dateTime') or start.get('date')
            end_time = end.get('dateTime') or end.get('date')
            
            appointment_data = {
                "id": event.get('id', ''),
                "summary": event.get('summary', 'No Title'),
                "description": event.get('description', ''),
                "start_time": start_time,
                "end_time": end_time,
                "status": event.get('status', 'confirmed'),
                "location": event.get('location', ''),
                "html_link": event.get('htmlLink', ''),
                "attendees": [
                    {
                        "email": att.get('email', ''),
                        "response_status": att.get('responseStatus', 'needsAction')
                    }
                    for att in event.get('attendees', [])
                ],
                "created": event.get('created', ''),
                "updated": event.get('updated', '')
            }
            user_appointments.append(appointment_data)
        
        filter_msg = f" for user {user_email}" if user_email else ""
        logger.info(f"📅 Retrieved {len(user_appointments)} appointments{filter_msg} from {start_date} to {end_date}")
        
        return JSONResponse(
            status_code=200,
            content={
                "count": len(user_appointments),
                "start_date": start_date,
                "end_date": end_date,
                "appointments": user_appointments
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving appointments: {e}")
        raise HTTPException(status_code=500, detail=str(e))

