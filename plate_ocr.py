"""
TrafficSentinel AI — Plate OCR Wrapper
Re-exports plate_scanner functions for backward compatibility.
"""

from plate_scanner import scan_license_plate as read_plate, _REGISTRATION_PATTERNS, _VALID_INDIAN_STATES

__all__ = ["read_plate"]
