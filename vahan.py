"""
TrafficSentinel AI — Vahan Wrapper Module
Re-exports registry_lookup functions for backward compatibility.
"""

from registry_lookup import lookup_vehicle_record as lookup_owner, MOCK_REGISTRY

__all__ = ["lookup_owner"]
