"""
Google Calendar Service
Manages Google Calendar API integration for appointment scheduling
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# Load .env from backend directory
backend_dir = Path(__file__).resolve().parent.parent
env_path = backend_dir / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger('CALENDAR')


def parse_calendar_datetime(datetime_str: str) -> datetime:
    """
    Parse datetime string from Google Calendar API
    Google Calendar returns UTC times with 'Z' suffix OR date strings for all-day events
    
    Args:
        datetime_str: ISO format datetime string (e.g., '2024-03-25T10:00:00Z' or '2024-03-25')
        
    Returns:
        Parsed datetime object (timezone-aware, defaults to UTC)
    """
    # Replace 'Z' with '+00:00' for compatibility with fromisoformat
    if datetime_str.endswith('Z'):
        datetime_str = datetime_str[:-1] + '+00:00'
    
    dt = datetime.fromisoformat(datetime_str)
    
    # If it's a date string (all-day event), it will be naive
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
        
    return dt


class GoogleCalendarService:
    """Handles all Google Calendar API operations"""
    
    def __init__(self, credentials_path: str = None, calendar_id: str = None):
        """
        Initialize Google Calendar Service
        
        Args:
            credentials_path: Path to service account JSON file (absolute or relative to backend dir)
            calendar_id: Calendar ID (default: reads from GOOGLE_CALENDAR_ID env var or 'primary')
        """
        # Get calendar ID from parameter or environment variable
        self.calendar_id = calendar_id or os.getenv('GOOGLE_CALENDAR_ID', 'primary')
        
        # Get credentials path from parameter or environment
        creds_path = os.getenv('GOOGLE_CALENDAR_CREDENTIALS') or credentials_path
        
        # Resolve path relative to backend directory if not absolute
        if creds_path and not os.path.isabs(creds_path):
            creds_path = str(backend_dir / creds_path)
        
        self.credentials_path = creds_path
        self.service = None
        self._initialize_service()
        
    def _initialize_service(self):
        """Initialize Google Calendar API service"""
        try:
            if not self.credentials_path:
                logger.warning("⚠️ No Google Calendar credentials provided")
                return
                
            # Use service account credentials
            SCOPES = ['https://www.googleapis.com/auth/calendar']
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=SCOPES
            )
            
            self.service = build('calendar', 'v3', credentials=credentials)
            logger.info(f"✅ Google Calendar service initialized successfully (Calendar ID: {self.calendar_id})")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Google Calendar service: {e}")
            self.service = None
    
    def is_available(self) -> bool:
        """Check if calendar service is available"""
        return self.service is not None
    
    def create_event(
        self,
        summary: str,
        start_time: datetime,
        end_time: datetime,
        description: str = None,
        attendees: List[str] = None,
        location: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a calendar event
        
        Args:
            summary: Event title/summary
            start_time: Event start datetime
            end_time: Event end datetime
            description: Event description (optional)
            attendees: List of attendee email addresses (optional)
            location: Event location (optional)
            
        Returns:
            Created event details or None if failed
        """
        if not self.is_available():
            logger.error("❌ Calendar service not available")
            return None
            
        try:
            # Ensure aware datetimes for API
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)

            # Prepare event body
            event = {
                'summary': summary,
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC',
                },
            }
            
            if description:
                event['description'] = description
                
            if location:
                event['location'] = location
                
            if attendees:
                event['attendees'] = [{'email': email} for email in attendees]
                
            # Create the event
            # Use 'none' for sendUpdates to avoid permission issues with service accounts
            created_event = self.service.events().insert(
                calendarId=self.calendar_id,
                body=event,
                sendUpdates='none'  # Don't send notifications (service accounts need Domain-Wide Delegation for 'all')
            ).execute()
            
            logger.info(f"✅ Event created: {created_event.get('id')}")
            return created_event
            
        except HttpError as error:
            logger.error(f"❌ HTTP error creating event: {error}")
            return None
        except Exception as e:
            logger.error(f"❌ Error creating event: {e}")
            return None
    
    def update_event(
        self,
        event_id: str,
        summary: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        description: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update an existing calendar event
        
        Args:
            event_id: Google Calendar event ID
            summary: Updated event title (optional)
            start_time: Updated start datetime (optional)
            end_time: Updated end datetime (optional)
            description: Updated description (optional)
            
        Returns:
            Updated event details or None if failed
        """
        if not self.is_available():
            logger.error("❌ Calendar service not available")
            return None
            
        try:
            # Get existing event
            event = self.service.events().get(
                calendarId=self.calendar_id,
                eventId=event_id
            ).execute()
            
            # Update fields
            if summary:
                event['summary'] = summary
            if start_time:
                if start_time.tzinfo is None:
                    start_time = start_time.replace(tzinfo=timezone.utc)
                event['start'] = {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC',
                }
            if end_time:
                if end_time.tzinfo is None:
                    end_time = end_time.replace(tzinfo=timezone.utc)
                event['end'] = {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC',
                }
            if description:
                event['description'] = description
            
            # Update the event
            updated_event = self.service.events().update(
                calendarId=self.calendar_id,
                eventId=event_id,
                body=event,
                sendUpdates='none'  # Don't send notifications (service accounts need Domain-Wide Delegation)
            ).execute()
            
            logger.info(f"✅ Event updated: {event_id}")
            return updated_event
            
        except HttpError as error:
            logger.error(f"❌ HTTP error updating event: {error}")
            return None
        except Exception as e:
            logger.error(f"❌ Error updating event: {e}")
            return None
    
    def delete_event(self, event_id: str) -> bool:
        """
        Delete a calendar event
        
        Args:
            event_id: Google Calendar event ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.error("❌ Calendar service not available")
            return False
            
        try:
            self.service.events().delete(
                calendarId=self.calendar_id,
                eventId=event_id,
                sendUpdates='none'  # Don't send notifications (service accounts need Domain-Wide Delegation)
            ).execute()
            
            logger.info(f"✅ Event deleted: {event_id}")
            return True
            
        except HttpError as error:
            logger.error(f"❌ HTTP error deleting event: {error}")
            return False
        except Exception as e:
            logger.error(f"❌ Error deleting event: {e}")
            return False
    
    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Get event details
        
        Args:
            event_id: Google Calendar event ID
            
        Returns:
            Event details or None if not found
        """
        if not self.is_available():
            logger.error("❌ Calendar service not available")
            return None
            
        try:
            event = self.service.events().get(
                calendarId=self.calendar_id,
                eventId=event_id
            ).execute()
            
            return event
            
        except HttpError as error:
            logger.error(f"❌ HTTP error getting event: {error}")
            return None
        except Exception as e:
            logger.error(f"❌ Error getting event: {e}")
            return None
    
    def list_events(
        self,
        time_min: datetime = None,
        time_max: datetime = None,
        max_results: int = 250
    ) -> List[Dict[str, Any]]:
        """
        List calendar events within a time range
        
        Args:
            time_min: Start of time range (default: now)
            time_max: End of time range (default: 30 days from now)
            max_results: Maximum number of events to return
            
        Returns:
            List of events
        """
        if not self.is_available():
            logger.error("❌ Calendar service not available")
            return []
            
        try:
            if not time_min:
                time_min = datetime.now(timezone.utc)
            if not time_max:
                time_max = time_min + timedelta(days=30)
            
            # Ensure awareness for format
            if time_min.tzinfo is None:
                time_min = time_min.replace(tzinfo=timezone.utc)
            if time_max.tzinfo is None:
                time_max = time_max.replace(tzinfo=timezone.utc)
                
            # Format datetime for Google Calendar API (RFC3339 with Z suffix for UTC)
            # Remove timezone info and add 'Z' to avoid +00:00Z format
            time_min_str = time_min.astimezone(timezone.utc).replace(tzinfo=None).isoformat() + 'Z'
            time_max_str = time_max.astimezone(timezone.utc).replace(tzinfo=None).isoformat() + 'Z'
            
            events_result = self.service.events().list(
                calendarId=self.calendar_id,
                timeMin=time_min_str,
                timeMax=time_max_str,
                maxResults=max_results,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            events = events_result.get('items', [])
            logger.info(f"✅ Retrieved {len(events)} events")
            return events
            
        except HttpError as error:
            logger.error(f"❌ HTTP error listing events: {error}")
            return []
        except Exception as e:
            logger.error(f"❌ Error listing events: {e}")
            return []
    
    def check_availability(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> bool:
        """
        Check if a time slot is available (no conflicts)
        
        Args:
            start_time: Proposed start datetime
            end_time: Proposed end datetime
            
        Returns:
            True if available, False if conflict exists
        """
        if not self.is_available():
            logger.warning("⚠️ Calendar service not available, assuming available")
            return True
            
        try:
            # Ensure awareness for comparison
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=timezone.utc)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
                
            # Get events in the requested time range
            events = self.list_events(time_min=start_time, time_max=end_time)
            
            # Check for conflicts
            for event in events:
                event_start = parse_calendar_datetime(
                    event['start'].get('dateTime', event['start'].get('date'))
                )
                event_end = parse_calendar_datetime(
                    event['end'].get('dateTime', event['end'].get('date'))
                )
                
                # Check if there's an overlap
                if (start_time < event_end and end_time > event_start):
                    logger.info(f"⚠️ Conflict found with event: {event.get('summary')}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error checking availability: {e}")
            return False
    
    def get_free_slots(
        self,
        date: datetime,
        duration_minutes: int = 60,
        working_hours: tuple = (9, 17)  # 9 AM to 5 PM
    ) -> List[Dict[str, datetime]]:
        """
        Get available time slots for a given date
        
        Args:
            date: Date to check availability
            duration_minutes: Duration of appointment in minutes
            working_hours: Tuple of (start_hour, end_hour) in 24-hour format
            
        Returns:
            List of available time slots with start and end times
        """
        if not self.is_available():
            logger.warning("⚠️ Calendar service not available")
            return []
            
        try:
            # Define working hours for the day
            start_hour, end_hour = working_hours
            day_start = date.replace(hour=start_hour, minute=0, second=0, microsecond=0)
            day_end = date.replace(hour=end_hour, minute=0, second=0, microsecond=0)
            
            # Get all events for the day
            events = self.list_events(time_min=day_start, time_max=day_end)
            
            # Generate all possible slots
            free_slots = []
            current_slot = day_start
            duration = timedelta(minutes=duration_minutes)
            
            while current_slot + duration <= day_end:
                slot_end = current_slot + duration
                is_free = True
                
                # Check if slot conflicts with any existing event
                for event in events:
                    event_start = parse_calendar_datetime(
                        event['start'].get('dateTime', event['start'].get('date'))
                    )
                    event_end = parse_calendar_datetime(
                        event['end'].get('dateTime', event['end'].get('date'))
                    )
                    
                    if (current_slot < event_end and slot_end > event_start):
                        is_free = False
                        break
                
                if is_free:
                    free_slots.append({
                        'start': current_slot,
                        'end': slot_end
                    })
                
                # Move to next slot (30-minute intervals)
                current_slot += timedelta(minutes=30)
            
            logger.info(f"✅ Found {len(free_slots)} free slots for {date.date()}")
            return free_slots
            
        except Exception as e:
            logger.error(f"❌ Error getting free slots: {e}")
            return []


# Global calendar service instance
calendar_service = None

def get_calendar_service() -> GoogleCalendarService:
    """Get or create global calendar service instance"""
    global calendar_service
    if calendar_service is None:
        calendar_service = GoogleCalendarService()
    return calendar_service
