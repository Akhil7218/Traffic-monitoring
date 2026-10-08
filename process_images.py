"""
TrafficSentinel AI — Single & Batch Image Processor
Runs traffic object detection, helmet compliance verification, and license plate recognition on static image files.
"""

import os
import cv2
import numpy as np
import threading
from ultralytics import YOLO
import easyocr

import config
from plate_scanner import scan_license_plate

# Load Computer Vision Inference Models
traffic_detector = YOLO(config.TRAFFIC_MODEL_PATH)
helmet_detector  = YOLO(config.HELMET_MODEL_PATH)
plate_detector   = YOLO(config.PLATE_MODEL_PATH)

_ocr_engine = easyocr.Reader(['en'], gpu=False, model_storage_directory=config.EASYOCR_MODEL_DIR)
_ocr_lock   = threading.Lock()


def read_plate(crop):
    return scan_license_plate(crop, _ocr_engine, _ocr_lock)


def calculate_overlap(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    return intersection / max((ax2 - ax1) * (ay2 - ay1), 1)


def process_image_directory():
    input_folder = os.environ.get("IMAGE_FOLDER", "images")
    output_folder = config.OUTPUT_DIR
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"Directory '{input_folder}' does not exist. Create it and place input images there.")
        return

    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    if not image_files:
        print(f"No valid images found in {input_folder}/")
        return

    print(f"Starting TrafficSentinel AI image analysis on {len(image_files)} file(s)...\n")

    for img_name in image_files:
        filepath = os.path.join(input_folder, img_name)
        img = cv2.imread(filepath)
        if img is None:
            print(f"  ❌ Failed to read {img_name}\n")
            continue

        img_h, img_w = img.shape[:2]
        print(f"Processing image: {img_name}")

        t_results = traffic_detector(img, verbose=False)[0]
        traffic_labels = []
        motorcycle_boxes = []
        person_boxes = []

        for box in t_results.boxes:
            lbl = traffic_detector.names[int(box.cls)]
            conf = float(box.conf)
            traffic_labels.append(lbl)
            if lbl == "motorcycle" and conf >= 0.3:
                motorcycle_boxes.append(list(map(int, box.xyxy[0])))
            if lbl == "person" and conf >= 0.4:
                person_boxes.append(list(map(int, box.xyxy[0])))

        h_results = helmet_detector(img, verbose=False)[0]
        headwear_labels = [helmet_detector.names[int(b.cls)] for b in h_results.boxes]

        no_helmet_count = headwear_labels.count("nohelmet")
        helmet_count = headwear_labels.count("helmet") + headwear_labels.count("motorcyclist")
        no_helmet_flag = "motorcycle" in traffic_labels and no_helmet_count > 0 and no_helmet_count > helmet_count

        triple_riding_flag = False
        for (mx1, my1, mx2, my2) in motorcycle_boxes:
            mw, mh = mx2 - mx1, my2 - my1
            expanded = (
                max(0, mx1 - int(mw * 0.05)), max(0, my1 - int(mh * 0.05)),
                min(img_w, mx2 + int(mw * 0.05)), min(img_h, my2 + int(mh * 0.05))
            )
            overlapping_persons = sum(
                1 for (px1, py1, px2, py2) in person_boxes
                if calculate_overlap((px1, py1, px2, py2), expanded) > 0.50
            )
            if overlapping_persons >= 3:
                triple_riding_flag = True
                break

        detected_infractions = []
        if no_helmet_flag:
            detected_infractions.append("NO HELMET")
        if triple_riding_flag:
            detected_infractions.append("TRIPLE RIDING")

        for item in detected_infractions:
            print(f"  🚨 {item}")
        if not detected_infractions:
            print("  ✅ Fully compliant (No infractions)")

        annotated_img = t_results.plot()
        for box in h_results.boxes:
            lbl = helmet_detector.names[int(box.cls)]
            if lbl != "licenseplate":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(
                    annotated_img, (x1, y1), (x2, y2),
                    (0, 0, 255) if lbl == "nohelmet" else (255, 128, 0), 2
                )

        y_offset = 50
        for item in detected_infractions:
            cv2.putText(annotated_img, item, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
            y_offset += 55

        if detected_infractions and motorcycle_boxes:
            mx1, my1, mx2, my2 = max(motorcycle_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            pad_x = int((mx2 - mx1) * 0.2)
            pad_y_top = int((my2 - my1) * 0.05)
            pad_y_bot = int((my2 - my1) * 0.35)
            cx1 = max(0, mx1 - pad_x)
            cy1 = max(0, my1 - pad_y_top)
            cx2 = min(img_w, mx2 + pad_x)
            cy2 = min(img_h, my2 + pad_y_bot)

            plate_crop_region = img[cy1:cy2, cx1:cx2]
            crop_w, crop_h = cx2 - cx1, cy2 - cy1
            plate_str = ""

            if plate_crop_region.size > 0:
                p_results = plate_detector(plate_crop_region, verbose=False)[0]
                candidates = []
                for pb in p_results.boxes:
                    conf = float(pb.conf)
                    if conf < 0.40:
                        continue
                    px1, py1, px2, py2 = map(int, pb.xyxy[0])
                    pw, ph = px2 - px1, py2 - py1
                    if ph == 0 or (pw / ph) < 1.0 or pw > crop_w * 0.95:
                        continue
                    if py1 < crop_h * 0.25:
                        continue
                    candidates.append((px1, py1, px2, py2, conf))

                if candidates:
                    px1, py1, px2, py2, conf = max(candidates, key=lambda c: c[4])
                    plate_str = read_plate(plate_crop_region[py1:py2, px1:px2])
                    print(f"  [Plate Recognition] Confidence={conf:.2f} | Text='{plate_str}'")

                    px1 += cx1
                    px2 += cx1
                    py1 += cy1
                    py2 += cy1
                    cv2.rectangle(annotated_img, (px1, py1), (px2, py2), (0, 255, 255), 2)
                    cv2.putText(
                        annotated_img, plate_str if plate_str else "PLATE",
                        (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
                    )

        out_path = os.path.join(output_folder, img_name)
        cv2.imwrite(out_path, annotated_img)
        print(f"  Saved output → {out_path}\n")

    print("✅ Image processing batch complete.")


if __name__ == "__main__":
    process_image_directory()
