"""
Calendar Integration Package
Handles Google Calendar API integration and appointment scheduling
"""

from .google_calendar import GoogleCalendarService, get_calendar_service
from .appointment_manager import AppointmentManager, get_appointment_manager
from .availability_checker import AvailabilityChecker, get_availability_checker

__all__ = [
    'GoogleCalendarService',
    'get_calendar_service',
    'AppointmentManager',
    'get_appointment_manager',
    'AvailabilityChecker',
    'get_availability_checker'
]
