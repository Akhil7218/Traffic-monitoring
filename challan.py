"""
TrafficSentinel AI — Challan Wrapper Module
Re-exports citation_builder functions for backward compatibility.
"""

from citation_builder import (
    build_citation_pdf as generate_challan,
    compute_infraction_fee as calculate_fine,
    query_prior_offences as get_offence_count,
    generate_payment_qr as generate_qr,
    BASE_FINES,
    SEVERITY_MULTIPLIER,
)

__all__ = [
    "generate_challan",
    "calculate_fine",
    "get_offence_count",
    "generate_qr",
    "BASE_FINES",
    "SEVERITY_MULTIPLIER",
]
