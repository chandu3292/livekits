"""
Availability Checker
Manages business hours, time zone handling, and availability validation
"""

import os
import logging
from datetime import datetime, timedelta, time, timezone
from typing import Optional, Dict, List, Tuple
from enum import Enum
from dotenv import load_dotenv

from .constants import (
    DEFAULT_TIMEZONE_OFFSET,
    DEFAULT_BUSINESS_HOURS,
    DEFAULT_SLOT_DURATION_MINUTES,
    DEFAULT_BUFFER_MINUTES,
    MAX_ADVANCE_DAYS,
    MAX_NEXT_DAYS_SEARCH,
    SLOT_SEARCH_INTERVAL_MINUTES
)

# Load environment variables
load_dotenv()

logger = logging.getLogger('AVAILABILITY')


class DayOfWeek(Enum):
    """Days of the week"""
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


class AvailabilityChecker:
    """Handles availability checking logic and business hours management"""
    
    def __init__(
        self,
        business_hours: Dict[int, List[Tuple[time, time]]] = None,
        excluded_dates: List[datetime] = None,
        slot_duration_minutes: int = DEFAULT_SLOT_DURATION_MINUTES,
        buffer_minutes: int = DEFAULT_BUFFER_MINUTES
    ):
        """
        Initialize Availability Checker
        
        Args:
            business_hours: Dictionary mapping day of week (0-6) to list of time ranges
                           e.g., {0: [(time(9, 0), time(17, 0))]} for Monday 9am-5pm
            excluded_dates: List of dates to exclude (holidays, etc.)
            slot_duration_minutes: Default appointment duration
            buffer_minutes: Buffer time between appointments
        """
        # Default business hours
        self.business_hours = business_hours or DEFAULT_BUSINESS_HOURS
        
        self.excluded_dates = excluded_dates or []
        self.slot_duration_minutes = slot_duration_minutes
        self.buffer_minutes = buffer_minutes if buffer_minutes is not None else DEFAULT_BUFFER_MINUTES
        
        # Get global timezone offset
        self.timezone_offset = DEFAULT_TIMEZONE_OFFSET
            
        logger.info(f"⚙️ AvailabilityChecker initialized with {self.timezone_offset} min offset")
        
    def is_business_day(self, date: datetime) -> bool:
        """
        Check if a date is a business day
        
        Args:
            date: Date to check
            
        Returns:
            True if business day, False otherwise
        """
        # Check if date is excluded (holiday, etc.)
        if date.date() in [d.date() for d in self.excluded_dates]:
            return False
        
        # Check if day of week has business hours
        return date.weekday() in self.business_hours
    
    def get_business_hours(self, date: datetime) -> List[Tuple[time, time]]:
        """
        Get business hours for a specific date
        
        Args:
            date: Date to get business hours for
            
        Returns:
            List of time ranges (start_time, end_time)
        """
        if not self.is_business_day(date):
            return []
        
        return self.business_hours.get(date.weekday(), [])
    
    def is_within_business_hours(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> bool:
        """
        Check if a time slot is within business hours
        Business hours are defined in local timezone (IST), but times may be passed in UTC.
        
        Args:
            start_time: Appointment start time (may be UTC)
            end_time: Appointment end time (may be UTC)
            
        Returns:
            True if within business hours, False otherwise
        """
        # Use global timezone offset to convert UTC to Local Time
        start_time_local = start_time + timedelta(minutes=self.timezone_offset)
        end_time_local = end_time + timedelta(minutes=self.timezone_offset)
        
        if not self.is_business_day(start_time_local):
            return False
        
        business_hours = self.get_business_hours(start_time_local)
        if not business_hours:
            return False
        
        appointment_start = start_time_local.time()
        appointment_end = end_time_local.time()
        
        # Check if appointment falls within any business hour range
        for bh_start, bh_end in business_hours:
            if appointment_start >= bh_start and appointment_end <= bh_end:
                return True
        
        return False
    
    def generate_available_slots(
        self,
        date: datetime,
        booked_slots: List[Tuple[datetime, datetime]] = None,
        duration_minutes: int = None
    ) -> List[Dict[str, datetime]]:
        """
        Generate available time slots for a given date
        
        Args:
            date: Date to generate slots for
            booked_slots: List of already booked time ranges
            duration_minutes: Slot duration (uses default if not specified)
            
        Returns:
            List of available time slots with start and end times
        """
        # Convert UTC anchor to local time to identify the target day
        local_date = date + timedelta(minutes=self.timezone_offset)
        
        if not self.is_business_day(local_date):
            logger.info(f"ℹ️ {local_date.date()} is not a business day in local time")
            return []
        
        duration = duration_minutes or self.slot_duration_minutes
        buffer = self.buffer_minutes
        booked_slots = booked_slots or []
        
        logger.info(f"📋 Generating slots for {local_date.date()} (Local IST) | Booked count: {len(booked_slots)}")
        if booked_slots:
            logger.info(f"   🚫 Booked (Local IST):")
            for b_start, b_end in booked_slots:
                b_s_local = b_start + timedelta(minutes=self.timezone_offset)
                b_e_local = b_end + timedelta(minutes=self.timezone_offset)
                logger.info(f"      - {b_s_local.strftime('%I:%M %p')} to {b_e_local.strftime('%I:%M %p')}")
        
        available_slots = []
        business_hours = self.get_business_hours(local_date)
        
        # Get timezone awareness from input date
        tz = date.tzinfo
        
        for bh_start, bh_end in business_hours:
            # Create Local datetime for the slot boundaries
            # Note: bh_start is local time (e.g. 9:00 AM)
            start_local = datetime.combine(local_date.date(), bh_start)
            end_local = datetime.combine(local_date.date(), bh_end)
            
            # Convert these Local boundaries back to UTC
            current_slot = start_local - timedelta(minutes=self.timezone_offset)
            end_of_period = end_local - timedelta(minutes=self.timezone_offset)
            
            # Ensure awareness matches input date
            if tz:
                current_slot = current_slot.replace(tzinfo=tz)
                end_of_period = end_of_period.replace(tzinfo=tz)
            
            while current_slot + timedelta(minutes=duration) <= end_of_period:
                slot_end = current_slot + timedelta(minutes=duration)
                
                # Check if slot conflicts with any booked slots
                is_available = True
                for booked_start, booked_end in booked_slots:
                    # Normalize booked times
                    b_start = booked_start
                    b_end = booked_end
                    
                    if tz:
                        if not b_start.tzinfo: b_start = b_start.replace(tzinfo=tz)
                        if not b_end.tzinfo: b_end = b_end.replace(tzinfo=tz)
                    else:
                        if b_start.tzinfo: b_start = b_start.replace(tzinfo=None)
                        if b_end.tzinfo: b_end = b_end.replace(tzinfo=None)
                        
                    # Add buffer time
                    booked_start_with_buffer = b_start - timedelta(minutes=buffer)
                    booked_end_with_buffer = b_end + timedelta(minutes=buffer)
                    
                    # Check for overlap
                    if (current_slot < booked_end_with_buffer and 
                        slot_end > booked_start_with_buffer):
                        is_available = False
                        break
                
                if is_available:
                    available_slots.append({'start': current_slot, 'end': slot_end})
                
                # Move to next slot
                current_slot += timedelta(minutes=SLOT_SEARCH_INTERVAL_MINUTES)
        
        logger.info(f"✅ Generated {len(available_slots)} available slots for {local_date.date()}")
        return available_slots
    
    def validate_appointment_time(
        self,
        start_time: datetime,
        duration_minutes: int = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate if an appointment time is acceptable
        
        Args:
            start_time: Proposed appointment start time
            duration_minutes: Appointment duration
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        duration = duration_minutes or self.slot_duration_minutes
        end_time = start_time + timedelta(minutes=duration)
        
        # Check if in the past - use timezone-aware comparison
        now_utc = datetime.now(timezone.utc) if start_time.tzinfo else datetime.now()
        if start_time < now_utc:
            # Convert to local time for better error message
            local_time = start_time + timedelta(minutes=self.timezone_offset)
            now_local = now_utc + timedelta(minutes=self.timezone_offset)
            return False, f"The requested time ({local_time.strftime('%I:%M %p')} IST on {local_time.strftime('%B %d')}) has already passed. Current time is {now_local.strftime('%I:%M %p')} IST. Please book for a future time."
        
        # Check if too far in the future
        if start_time > now_utc + timedelta(days=MAX_ADVANCE_DAYS):
            return False, f"Cannot schedule more than {MAX_ADVANCE_DAYS} days in advance"
        
        # Check if it's a business day
        if not self.is_business_day(start_time):
            return False, "Selected date is not available (weekend or holiday)"
        
        # Check if within business hours
        if not self.is_within_business_hours(start_time, end_time):
            return False, "Selected time is outside business hours"
        
        return True, None
    
    def get_next_available_date(
        self,
        from_date: datetime = None,
        max_days_ahead: int = MAX_NEXT_DAYS_SEARCH
    ) -> Optional[datetime]:
        """
        Find the next available business day
        
        Args:
            from_date: Starting date (default: today)
            max_days_ahead: Maximum days to look ahead
            
        Returns:
            Next available date or None if not found
        """
        current_date = from_date or datetime.now()
        
        for i in range(max_days_ahead):
            check_date = current_date + timedelta(days=i)
            if self.is_business_day(check_date):
                return check_date
        
        return None
    
    def get_suggested_times(
        self,
        date: datetime,
        booked_slots: List[Tuple[datetime, datetime]] = None,
        count: int = 5
    ) -> List[Dict[str, datetime]]:
        """
        Get suggested available time slots for a date
        
        Args:
            date: Date to get suggestions for
            booked_slots: Already booked time slots
            count: Number of suggestions to return
            
        Returns:
            List of suggested time slots
        """
        all_slots = self.generate_available_slots(date, booked_slots)
        
        # Return first 'count' slots
        return all_slots[:count]
    
    def format_time_slot(self, slot: Dict[str, datetime]) -> str:
        """Format a time slot for display in local time"""
        # Apply offset to format in Local time
        start_local = slot['start'] + timedelta(minutes=self.timezone_offset)
        end_local = slot['end'] + timedelta(minutes=self.timezone_offset)
        
        return f"{start_local.strftime('%I:%M %p')} (IST)"
    
    def add_excluded_date(self, date: datetime):
        """Add a date to the exclusion list (e.g., company holiday)"""
        if date not in self.excluded_dates:
            self.excluded_dates.append(date)
            logger.info(f"✅ Added excluded date: {date.date()}")
    
    def remove_excluded_date(self, date: datetime):
        """Remove a date from the exclusion list"""
        self.excluded_dates = [d for d in self.excluded_dates if d.date() != date.date()]
        logger.info(f"✅ Removed excluded date: {date.date()}")
    
    def set_business_hours(self, day_of_week: int, hours: List[Tuple[time, time]]):
        """
        Set business hours for a specific day
        
        Args:
            day_of_week: Day (0=Monday, 6=Sunday)
            hours: List of time ranges
        """
        self.business_hours[day_of_week] = hours
        logger.info(f"✅ Updated business hours for day {day_of_week}")


# Global availability checker instance
availability_checker = None

def get_availability_checker() -> AvailabilityChecker:
    """Get or create global availability checker instance"""
    global availability_checker
    if availability_checker is None:
        availability_checker = AvailabilityChecker()
    return availability_checker
