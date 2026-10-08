"""
TrafficSentinel AI — Test Suite Wrapper
Re-exports test cases from test_infractions.py for backward compatibility.
"""

from tests.test_infractions import (
    TestNoHelmet,
    TestTripleRiding,
    TestWrongWay,
    TestFineCalculation,
    TestPlateRegex,
)

__all__ = [
    "TestNoHelmet",
    "TestTripleRiding",
    "TestWrongWay",
    "TestFineCalculation",
    "TestPlateRegex",
]
