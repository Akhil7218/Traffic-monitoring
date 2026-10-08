"""
TrafficSentinel AI — National Vehicle Registration Lookup Service
Queries official registration databases (e.g., Vahan API) or fallback mock database for owner identification.
"""

import os
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REGISTRY_API_KEY = os.environ.get("VAHAN_API_KEY", "")
REGISTRY_API_URL = "https://vahan.parivahan.gov.in/vahanservice/vahan/api/rc-details"

DEMO_OWNER_NAME  = os.environ.get("DEMO_NAME",  "Demo Owner")
DEMO_OWNER_PHONE = os.environ.get("DEMO_PHONE", "+910000000000")
DEMO_OWNER_EMAIL = os.environ.get("DEMO_EMAIL", "demo@example.com")

MOCK_REGISTRY = {
    "KA0112234":  {"name": DEMO_OWNER_NAME, "phone": DEMO_OWNER_PHONE, "email": DEMO_OWNER_EMAIL, "city": "Bengaluru"},
    "KA015678":   {"name": DEMO_OWNER_NAME, "phone": DEMO_OWNER_PHONE, "email": DEMO_OWNER_EMAIL, "city": "Bengaluru"},
    "TN05AT7024": {"name": DEMO_OWNER_NAME, "phone": DEMO_OWNER_PHONE, "email": DEMO_OWNER_EMAIL, "city": "Chennai"},
    "MH04CD1234": {"name": DEMO_OWNER_NAME, "phone": DEMO_OWNER_PHONE, "email": DEMO_OWNER_EMAIL, "city": "Mumbai"},
    "DL09W6392":  {"name": DEMO_OWNER_NAME, "phone": DEMO_OWNER_PHONE, "email": DEMO_OWNER_EMAIL, "city": "Delhi"},
    "KA0112236":  {"name": DEMO_OWNER_NAME, "phone": DEMO_OWNER_PHONE, "email": DEMO_OWNER_EMAIL, "city": "Bengaluru"},
    "KL11AB1234": {"name": DEMO_OWNER_NAME, "phone": DEMO_OWNER_PHONE, "email": DEMO_OWNER_EMAIL, "city": "Kerala"},
    "KL09CA1671": {"name": DEMO_OWNER_NAME, "phone": DEMO_OWNER_PHONE, "email": DEMO_OWNER_EMAIL, "city": "Kerala"},
    "KL07CD5678": {"name": DEMO_OWNER_NAME, "phone": DEMO_OWNER_PHONE, "email": DEMO_OWNER_EMAIL, "city": "Kerala"},
    "KL15EF9012": {"name": DEMO_OWNER_NAME, "phone": DEMO_OWNER_PHONE, "email": DEMO_OWNER_EMAIL, "city": "Kerala"},
}


def lookup_vehicle_record(plate_number):
    """
    Performs lookup for vehicle registration record.
    Returns dict with keys: name, phone, email, city
    Returns None if plate is invalid or unrecorded.
    """
    if not plate_number or plate_number == "UNKNOWN":
        return None

    clean_plate = plate_number.upper().replace(" ", "").replace("-", "")

    if REGISTRY_API_KEY:
        try:
            resp = requests.post(
                REGISTRY_API_URL,
                json={"regNo": clean_plate},
                headers={"x-api-key": REGISTRY_API_KEY},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "name":  data.get("ownerName", "Unknown"),
                    "phone": data.get("mobileNo", ""),
                    "email": "",
                    "city":  data.get("regDistrict", ""),
                }
        except Exception as err:
            print(f"  [RegistryLookup] API query failed ({err}); falling back to local registry table")

    return MOCK_REGISTRY.get(clean_plate, None)


# Backward Compatibility Alias
lookup_owner = lookup_vehicle_record
