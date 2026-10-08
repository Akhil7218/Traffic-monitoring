"""
TrafficSentinel AI — Detect Wrapper Module
Re-exports process_images script execution for backward compatibility.
"""

from process_images import process_image_directory

if __name__ == "__main__":
    process_image_directory()
