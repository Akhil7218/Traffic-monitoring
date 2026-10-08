"""
TrafficSentinel AI — Notifications Wrapper Module
Re-exports alert_dispatcher functions for backward compatibility.
"""

from alert_dispatcher import (
    dispatch_infraction_alert as notify_violation,
    dispatch_daily_summary as send_daily_summary,
)

__all__ = ["notify_violation", "send_daily_summary"]
