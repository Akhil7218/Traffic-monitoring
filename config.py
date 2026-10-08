"""
TrafficSentinel AI — Central System Configuration
Centralized configuration management with environment variable overrides.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent

# Application Information & Security
PROJECT_NAME = os.environ.get("PROJECT_NAME", "TrafficSentinel AI")
SECRET_KEY = os.environ.get("SECRET_KEY", "sentinel-traffic-ai-secure-secret-key-2026")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "")
PORT = int(os.environ.get("PORT", 5001))
HOST = os.environ.get("HOST", "0.0.0.0")

# Vision Model Directory & Paths
MODEL_DIR = os.environ.get("MODEL_DIR", str(BASE_DIR / "models"))
TRAFFIC_MODEL_PATH = os.environ.get("TRAFFIC_MODEL_PATH", str(BASE_DIR / "models" / "yolov8s.pt"))
HELMET_MODEL_PATH = os.environ.get("HELMET_MODEL_PATH", str(BASE_DIR / "models" / "best.pt"))
PLATE_MODEL_PATH = os.environ.get("PLATE_MODEL_PATH", str(BASE_DIR / "models" / "Plate.pt"))
EASYOCR_MODEL_DIR = os.environ.get("EASYOCR_MODEL_DIR", str(BASE_DIR / "easyocr_models"))

# Data Directories
DATA_DIR = os.environ.get("DATA_DIR", str(BASE_DIR / "data"))
INPUT_VIDEO_DIR = os.environ.get("INPUT_VIDEO_DIR", str(BASE_DIR / "data" / "input_videos"))
TEST_VIDEO_DIR = os.environ.get("TEST_VIDEO_DIR", str(BASE_DIR / "data" / "test_videos"))
VIDEO_FOLDER = os.environ.get("VIDEO_FOLDER", str(BASE_DIR / "videos"))

# Output Directories & Database Path
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", str(BASE_DIR / "outputs"))
VIOLATION_OUTPUT_DIR = os.environ.get("VIOLATION_OUTPUT_DIR", str(BASE_DIR / "outputs" / "violations"))
SCREENSHOT_DIR = os.environ.get("SCREENSHOT_DIR", str(BASE_DIR / "static" / "screenshots"))
CHALLAN_DIR = os.environ.get("CHALLAN_DIR", str(BASE_DIR / "static" / "challans"))
DB_PATH = os.environ.get("DB_PATH", str(BASE_DIR / "violations.db"))

# Create required directories dynamically
for _directory in [
    INPUT_VIDEO_DIR,
    TEST_VIDEO_DIR,
    VIDEO_FOLDER,
    OUTPUT_DIR,
    VIOLATION_OUTPUT_DIR,
    SCREENSHOT_DIR,
    CHALLAN_DIR,
    EASYOCR_MODEL_DIR,
]:
    os.makedirs(_directory, exist_ok=True)
