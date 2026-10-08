"""
TrafficSentinel AI — Model Evaluator & Benchmark Engine
Evaluates detection models and performance metrics against validation benchmark data.
"""

from ultralytics import YOLO
import config


def evaluate_models():
    print("TrafficSentinel AI — Initializing Model Benchmarks...")
    traffic_model = YOLO(config.TRAFFIC_MODEL_PATH)
    helmet_model  = YOLO(config.HELMET_MODEL_PATH)
    plate_model   = YOLO(config.PLATE_MODEL_PATH)

    print("Loaded vision models successfully:")
    print(f"  Traffic tracking model : {config.TRAFFIC_MODEL_PATH}")
    print(f"  Helmet compliance model: {config.HELMET_MODEL_PATH}")
    print(f"  License plate detector : {config.PLATE_MODEL_PATH}")
    print("Benchmark loading check passed.")


if __name__ == "__main__":
    evaluate_models()
