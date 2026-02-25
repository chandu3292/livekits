"""
Appointment Manager
Central coordinator for appointment scheduling, validation, and calendar synchronization
"""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone, time
from typing import Optional, Dict, List, Any, Tuple
from enum import Enum

from .google_calendar import GoogleCalendarService, get_calendar_service, parse_calendar_datetime
from .availability_checker import AvailabilityChecker, get_availability_checker
from .constants import TASK_TYPES, DEFAULT_SUGGESTIONS_COUNT, NEXT_AVAILABLE_SEARCH_DAYS, LATE_HOUR_UTC

logger = logging.getLogger('APPOINTMENT')


class AppointmentStatus(Enum):
    """Appointment status types"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    RESCHEDULED = "rescheduled"


class TaskType(Enum):
    """Configurable task types for appointments"""
    CONSULTATION = "consultation"
    INTERVIEW = "interview"
    MEETING = "meeting"
    DEMO = "demo"
    SUPPORT = "support"
    CUSTOM = "custom"


class AppointmentManager:
    """Manages the complete appointment scheduling lifecycle"""
    
    def __init__(
        self,
        calendar_service: GoogleCalendarService = None,
        availability_checker: AvailabilityChecker = None
    ):
        """
        Initialize Appointment Manager
        
        Args:
            calendar_service: Google Calendar service instance
            availability_checker: Availability checker instance
        """
        self.calendar_service = calendar_service or get_calendar_service()
        self.availability_checker = availability_checker or get_availability_checker()
        
        # Task type configuration
        self.task_types = TASK_TYPES
    
    def get_available_task_types(self) -> List[Dict[str, Any]]:
        """Get list of available task types"""
        return [
            {
                "id": task_id,
                "name": config["name"],
                "duration": config["duration_minutes"],
                "description": config["description"]
            }
            for task_id, config in self.task_types.items()
        ]
    
    def validate_task_type(self, task_type: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a task type is supported
        
        Args:
            task_type: Task type identifier
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if task_type not in self.task_types:
            available = ", ".join(self.task_types.keys())
            return False, f"Invalid task type. Available types: {available}"
        return True, None
    
    def parse_date_time(self, date_str: str, time_str: str = None) -> Optional[datetime]:
        """
        Parse date and time strings into datetime object
        
        Args:
            date_str: Date string (e.g., "2024-03-15", "tomorrow", "next Monday")
            time_str: Time string (e.g., "14:00", "2:00 PM")
            
        Returns:
            Parsed datetime (UTC) or None if invalid
        """
        try:
            offset = self.availability_checker.timezone_offset
            now_utc = datetime.now(timezone.utc)
            local_now = now_utc + timedelta(minutes=offset)
            
            # 1. Handle Full ISO/DateTime strings first (especially when time_str is None)
            if time_str is None and ('T' in date_str or (' ' in date_str and len(date_str) > 10)):
                try:
                    # Clean up Z and convert
                    date_str_clean = date_str.replace('Z', '+00:00')
                    dt = datetime.fromisoformat(date_str_clean)
                    
                    if dt.tzinfo is None:
                        # Naive -> assume it's Local (IST) and convert to UTC
                        return (dt - timedelta(minutes=offset)).replace(tzinfo=timezone.utc)
                    return dt.astimezone(timezone.utc) # Already aware, just normalize to UTC
                except ValueError:
                    pass

            # 2. Extract Base Date
            base_date = None
            if date_str.lower() == "today":
                base_date = local_now.date()
            elif date_str.lower() == "tomorrow":
                base_date = (local_now + timedelta(days=1)).date()
            elif "next" in date_str.lower():
                # Simplified: skip 7 days
                base_date = (local_now + timedelta(days=7)).date()
            else:
                # Try parsing as ISO date or full ISO datetime (extracting just the date)
                try:
                    if 'T' in date_str or ' ' in date_str:
                        date_str_clean = date_str.replace('Z', '+00:00')
                        base_date = datetime.fromisoformat(date_str_clean).date()
                    else:
                        base_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except ValueError:
                    try:
                        base_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                    except (ValueError, IndexError):
                        return None
            
            # 3. Parse Time
            if time_str:
                # Try multiple time formats
                for fmt in ["%H:%M", "%I:%M %p", "%I:%M%p", "%H:%M:%S"]:
                    try:
                        parsed_time = datetime.strptime(time_str.strip(), fmt).time()
                        break
                    except ValueError:
                        continue
                else:
                    return None
            else:
                # Default to 9 AM Local
                parsed_time = time(9, 0)
            
            # 4. Combine and convert Local to UTC
            dt_local = datetime.combine(base_date, parsed_time)
            return (dt_local - timedelta(minutes=offset)).replace(tzinfo=timezone.utc)
            
        except Exception as e:
            logger.error(f"❌ Error parsing date/time: {e}")
            return None
    
    def check_availability(
        self,
        start_time: datetime,
        task_type: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a time slot is available for an appointment
        
        Args:
            start_time: Proposed start time
            task_type: Type of appointment
            
        Returns:
            Tuple of (is_available, message)
        """
        # Get task configuration
        task_config = self.task_types.get(task_type)
        if not task_config:
            return False, "Invalid task type"
        
        duration = task_config["duration_minutes"]
        
        # Validate appointment time (business hours, etc.)
        is_valid, error_msg = self.availability_checker.validate_appointment_time(
            start_time, duration
        )
        if not is_valid:
            return False, error_msg
        
        # Check calendar availability
        end_time = start_time + timedelta(minutes=duration)
        if self.calendar_service.is_available():
            is_free = self.calendar_service.check_availability(start_time, end_time)
            if not is_free:
                return False, "Time slot is already booked"
        
        return True, "Time slot is available"
    
    def get_available_slots(
        self,
        date: datetime,
        task_type: str,
        count: int = DEFAULT_SUGGESTIONS_COUNT
    ) -> List[Dict[str, Any]]:
        """
        Get available time slots for a specific date and task type
        
        Args:
            date: Date to check
            task_type: Type of appointment
            count: Number of slots to return
            
        Returns:
            List of available time slots
        """
        task_config = self.task_types.get(task_type)
        if not task_config:
            logger.error(f"❌ Invalid task type: {task_type}")
            return []
        
        duration = task_config["duration_minutes"]
        
        # Calculate the UTC range for the local day
        # If date is '2024-03-15' (human intended local day)
        # We need to query from (Local 00:00 to Local 23:59) converted to UTC
        offset = self.availability_checker.timezone_offset
        local_start = datetime.combine(date.date(), time(0, 0, 0)).replace(tzinfo=timezone.utc)
        local_end = datetime.combine(date.date(), time(23, 59, 59)).replace(tzinfo=timezone.utc)
        
        utc_min = local_start - timedelta(minutes=offset)
        utc_max = local_end - timedelta(minutes=offset)
        
        # Get booked slots from calendar if available
        booked_slots = []
        if self.calendar_service.is_available():
            events = self.calendar_service.list_events(
                time_min=utc_min,
                time_max=utc_max
            )
            for event in events:
                start = parse_calendar_datetime(
                    event['start'].get('dateTime', event['start'].get('date'))
                )
                end = parse_calendar_datetime(
                    event['end'].get('dateTime', event['end'].get('date'))
                )
                booked_slots.append((start, end))
        
        # Generate available slots
        slots = self.availability_checker.generate_available_slots(
            date, booked_slots, duration
        )
        
        # Format and return
        formatted_slots = []
        for slot in slots[:count]:
            formatted_slots.append({
                'start': slot['start'],
                'end': slot['end'],
                'formatted': self.availability_checker.format_time_slot(slot),
                'duration_minutes': duration
            })
        
        return formatted_slots
    
    def create_appointment(
        self,
        user_id: int,
        user_name: str,
        user_email: str,
        task_type: str,
        start_time: datetime,
        notes: str = None,
        phone: str = None
    ) -> Dict[str, Any]:
        """
        Create a new appointment
        
        Args:
            user_id: User identifier
            user_name: User's full name
            user_email: User's email address
            task_type: Type of appointment
            start_time: Appointment start time
            notes: Additional notes (optional)
            phone: User's phone number (optional)
            
        Returns:
            Appointment details dictionary
        """
        # Validate task type
        is_valid, error = self.validate_task_type(task_type)
        if not is_valid:
            logger.error(f"❌ Invalid task type: {error}")
            return {"success": False, "error": error}
        
        task_config = self.task_types[task_type]
        duration = task_config["duration_minutes"]
        end_time = start_time + timedelta(minutes=duration)
        
        # Check availability
        is_available, msg = self.check_availability(start_time, task_type)
        if not is_available:
            logger.warning(f"⚠️ Slot not available: {msg}")
            return {"success": False, "error": msg}
        
        # Generate unique appointment ID
        appointment_id = str(uuid.uuid4())
        
        # Prepare appointment details
        appointment = {
            "appointment_id": appointment_id,
            "user_id": user_id,
            "user_name": user_name,
            "user_email": user_email,
            "phone": phone,
            "task_type": task_type,
            "task_name": task_config["name"],
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": duration,
            "status": AppointmentStatus.CONFIRMED.value,
            "notes": notes,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "calendar_event_id": None
        }
        
        # Create calendar event if service is available
        if self.calendar_service.is_available():
            description = f"""
Appointment Details:
- Type: {task_config['name']}
- Client: {user_name}
- Email: {user_email}
- Phone: {phone or 'N/A'}
- Notes: {notes or 'N/A'}

Appointment ID: {appointment_id}
            """.strip()
            
            event = self.calendar_service.create_event(
                summary=f"{task_config['name']} - {user_name}",
                start_time=start_time,
                end_time=end_time,
                description=description
                # Note: Attendees removed - service accounts need Domain-Wide Delegation to invite attendees
            )
            
            if event:
                appointment["calendar_event_id"] = event.get("id")
                appointment["calendar_link"] = event.get("htmlLink")
                logger.info(f"✅ Calendar event created: {event.get('id')}")
        
        logger.info(f"✅ Appointment created: {appointment_id}")
        return {"success": True, "appointment": appointment}
    
    def update_appointment(
        self,
        appointment_id: str,
        new_start_time: datetime = None,
        new_task_type: str = None,
        notes: str = None,
        calendar_event_id: str = None
    ) -> Dict[str, Any]:
        """
        Update an existing appointment
        
        Args:
            appointment_id: Appointment identifier
            new_start_time: New start time (optional)
            new_task_type: New task type (optional)
            notes: Updated notes (optional)
            calendar_event_id: Google Calendar event ID
            
        Returns:
            Updated appointment details
        """
        try:
            # If rescheduling, check new availability
            if new_start_time and new_task_type:
                is_available, msg = self.check_availability(new_start_time, new_task_type)
                if not is_available:
                    return {"success": False, "error": msg}
                
                task_config = self.task_types[new_task_type]
                duration = task_config["duration_minutes"]
                new_end_time = new_start_time + timedelta(minutes=duration)
                
                # Update calendar event if available
                if calendar_event_id and self.calendar_service.is_available():
                    self.calendar_service.update_event(
                        event_id=calendar_event_id,
                        start_time=new_start_time,
                        end_time=new_end_time,
                        summary=f"{task_config['name']}"
                    )
            
            logger.info(f"✅ Appointment updated: {appointment_id}")
            return {"success": True, "appointment_id": appointment_id}
            
        except Exception as e:
            logger.error(f"❌ Error updating appointment: {e}")
            return {"success": False, "error": str(e)}
    
    def cancel_appointment(
        self,
        appointment_id: str,
        calendar_event_id: str = None
    ) -> Dict[str, Any]:
        """
        Cancel an appointment
        
        Args:
            appointment_id: Appointment identifier
            calendar_event_id: Google Calendar event ID
            
        Returns:
            Cancellation status
        """
        try:
            # Delete calendar event if available
            if calendar_event_id and self.calendar_service.is_available():
                success = self.calendar_service.delete_event(calendar_event_id)
                if not success:
                    logger.warning(f"⚠️ Failed to delete calendar event: {calendar_event_id}")
            
            logger.info(f"✅ Appointment cancelled: {appointment_id}")
            return {"success": True, "appointment_id": appointment_id}
            
        except Exception as e:
            logger.error(f"❌ Error cancelling appointment: {e}")
            return {"success": False, "error": str(e)}
    
    def get_appointment_suggestions(
        self,
        task_type: str,
        preferred_date: datetime = None,
        days_ahead: int = 7
    ) -> Dict[str, Any]:
        """
        Get suggested available appointments
        
        Args:
            task_type: Type of appointment
            preferred_date: Preferred date (default: tomorrow)
            days_ahead: Number of days to look ahead
            
        Returns:
            Dictionary with suggested dates and times
        """
        if not preferred_date:
            preferred_date = datetime.now(timezone.utc) + timedelta(days=1)
        
        suggestions = {}
        
        for i in range(days_ahead):
            check_date = preferred_date + timedelta(days=i)
            
            # Get available slots for this date
            slots = self.get_available_slots(check_date, task_type, count=3)
            
            if slots:
                suggestions[check_date.strftime("%Y-%m-%d")] = {
                    "date": check_date.strftime("%A, %B %d, %Y"),
                    "slots": slots
                }
        
        return {
            "task_type": task_type,
            "suggestions": suggestions
        }
    
    def find_next_available_slot(
        self,
        task_type: str,
        from_date: datetime = None,
        max_days_ahead: int = NEXT_AVAILABLE_SEARCH_DAYS,
        count: int = DEFAULT_SUGGESTIONS_COUNT
    ) -> Optional[Dict[str, Any]]:
        """
        Find the next available appointment slot
        
        Args:
            task_type: Type of appointment
            from_date: Start searching from this date (default: now)
            max_days_ahead: Maximum days to search ahead
            count: Number of slots to return
            
        Returns:
            Dictionary with next available slot details or None
        """
        if not from_date:
            from_date = datetime.now(timezone.utc)
        
        # Start from next day if time is late
        if from_date.hour >= LATE_HOUR_UTC:
            from_date = from_date + timedelta(days=1)
        from_date = from_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
        for i in range(max_days_ahead):
            check_date = from_date + timedelta(days=i)
            
            # Get available slots for this date
            slots = self.get_available_slots(check_date, task_type, count=count)
            
            if slots:
                return {
                    "found": True,
                    "date": check_date.strftime("%Y-%m-%d"),
                    "date_formatted": check_date.strftime("%A, %B %d, %Y"),
                    "first_slot": slots[0],
                    "all_slots": slots[:count]
                }
        
        return {
            "found": False,
            "message": f"No available slots found in the next {max_days_ahead} days"
        }
    
    def format_appointment_confirmation(
        self,
        appointment: Dict[str, Any]
    ) -> str:
        """
        Format appointment details for confirmation message
        
        Args:
            appointment: Appointment dictionary
            
        Returns:
            Formatted confirmation message
        """
        start_utc = datetime.fromisoformat(appointment['start_time'])
        end_utc = datetime.fromisoformat(appointment['end_time'])
        
        # Apply timezone offset for local display
        offset = self.availability_checker.timezone_offset
        start = start_utc + timedelta(minutes=offset)
        end = end_utc + timedelta(minutes=offset)
        
        message = f"""
✅ Appointment Confirmed!

Date: {start.strftime('%A, %B %d, %Y')}
Time: {start.strftime('%I:%M %p')} (IST)
Duration: {appointment['duration_minutes']} minutes

This appointment has been added to Google Calendar.
        """.strip()
        
        if appointment.get('calendar_link'):
            message += f"\n\nYou will receive a calendar invitation at {appointment.get('user_email', 'your email')}."
        
        return message


# Global appointment manager instance
appointment_manager = None

def get_appointment_manager() -> AppointmentManager:
    """Get or create global appointment manager instance"""
    global appointment_manager
    if appointment_manager is None:
        appointment_manager = AppointmentManager()
    return appointment_manager
