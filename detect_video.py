"""
TrafficSentinel AI — Detect Video Wrapper Module
Re-exports process_videos script execution for backward compatibility.
"""

from process_videos import process_all_input_videos

if __name__ == "__main__":
    process_all_input_videos()
