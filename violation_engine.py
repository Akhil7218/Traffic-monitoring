"""
TrafficSentinel AI — Violation Engine Module Link
Re-exports TrafficInfractionAnalyzer as ViolationEngine for complete backwards compatibility.
"""

from infraction_detector import TrafficInfractionAnalyzer as ViolationEngine

__all__ = ["ViolationEngine"]
