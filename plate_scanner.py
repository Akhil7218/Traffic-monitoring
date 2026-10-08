"""
TrafficSentinel AI — Automatic License Plate Recognition (ALPR) & OCR Scanner
High-precision license plate extraction and text normalization for Indian vehicular registration patterns.
"""

import re
import cv2
import numpy as np

# Optimized regex search patterns for registration numbers
_REGISTRATION_PATTERNS = [
    re.compile(r'[A-Z]{2}\d{2}[A-Z]{2}\d{4}'),     # Standard format e.g. KA03XY1234
    re.compile(r'[A-Z]{2}\d{2}[A-Z]{1}\d{4}'),      # Single-letter series e.g. KL30G1234
    re.compile(r'[A-Z]{2}\d{2}[A-Z]{1,3}\d{3,4}'),  # Extended variant catch-all
]

_VALID_INDIAN_STATES = {
    'AP','AR','AS','BR','CG','CH','DD','DL','DN','GA','GJ',
    'HR','HP','JH','JK','KA','KL','LA','LD','MH','ML','MN',
    'MP','MZ','NL','OD','PB','PY','RJ','SK','TN','TR','TS',
    'TG','UK','UP','WB','AN'
}

_COMMON_STATE_CORRECTIONS = {
    'HH': 'MH', 'HM': 'MH', 'IH': 'MH', 'NH': 'MH',
    'EL': 'KL', 'IL': 'KL',
    'KI': 'KA',
    'OD': 'OD',
    'TZ': 'TN',
    'IK': 'UK',
}


def _normalize_plate_string(raw_text):
    clean = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    clean = clean.replace('IND', '').replace('INDIA', '')
    if len(clean) < 4:
        return clean

    char_list = list(clean)
    num_to_letter = {'0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G', '2': 'Z'}

    # Normalize state code characters (indices 0 and 1)
    for idx in (0, 1):
        if idx < len(char_list) and char_list[idx].isdigit():
            char_list[idx] = num_to_letter.get(char_list[idx], char_list[idx])

    state_prefix = ''.join(char_list[:2])
    if state_prefix not in _VALID_INDIAN_STATES and state_prefix in _COMMON_STATE_CORRECTIONS:
        corrected_state = _COMMON_STATE_CORRECTIONS[state_prefix]
        char_list[0], char_list[1] = corrected_state[0], corrected_state[1]

    # Normalize RTO numerical digits (indices 2 and 3)
    letter_to_num = {'O': '0', 'I': '1', 'S': '5', 'B': '8', 'Z': '2', 'G': '6', 'A': '4', 'T': '7', 'L': '1'}
    for idx in (2, 3):
        if idx < len(char_list) and char_list[idx].isalpha():
            char_list[idx] = letter_to_num.get(char_list[idx], char_list[idx])

    return ''.join(char_list)


def _search_registration_pattern(text_string):
    normalized = _normalize_plate_string(text_string)
    for pattern in _REGISTRATION_PATTERNS:
        match = pattern.search(normalized)
        if match:
            return match.group()
    return ""


def _evaluate_swapped_lines(text_string):
    normalized = _normalize_plate_string(text_string)
    if len(normalized) < 8:
        return ""
    for split_point in range(3, 7):
        if split_point >= len(normalized):
            continue
        swapped = normalized[split_point:] + normalized[:split_point]
        swapped_normalized = _normalize_plate_string(swapped)
        for pattern in _REGISTRATION_PATTERNS:
            match = pattern.search(swapped_normalized)
            if match:
                return match.group()
    return ""


def _find_best_registration_match(raw_text):
    primary = _search_registration_pattern(raw_text)
    if primary:
        return primary
    return _evaluate_swapped_lines(raw_text)


def _perform_easyocr_read(img_matrix, ocr_reader, thread_lock):
    try:
        with thread_lock:
            detected_items = ocr_reader.readtext(img_matrix, detail=1)
        if not detected_items:
            return ""
        sorted_items = sorted(detected_items, key=lambda item: (item[0][0][1], item[0][0][0]))
        extracted_strings = [item[1] for item in sorted_items if item[2] > 0.1]
        return " ".join(extracted_strings)
    except Exception:
        return ""


def scan_license_plate(plate_crop_image, ocr_reader, thread_lock):
    """
    Scans and extracts license plate text from cropped image array.

    Args:
        plate_crop_image : np.ndarray - Crop of license plate region
        ocr_reader       : EasyOCR reader instance
        thread_lock      : Lock for thread safety during OCR inference

    Returns:
        String representing the recognized plate (e.g. 'KA01AB1234') or '' if invalid.
    """
    if plate_crop_image is None or plate_crop_image.size == 0:
        return ""
    height, width = plate_crop_image.shape[:2]
    if height < 5 or width < 5:
        return ""

    MAX_WIDTH = 300
    if width > MAX_WIDTH:
        scale_ratio = MAX_WIDTH / width
        height = max(1, int(height * scale_ratio))
        width = MAX_WIDTH
        plate_crop_image = cv2.resize(plate_crop_image, (width, height), interpolation=cv2.INTER_AREA)

    scale_factor = min(4, max(2, int(300 / max(width, 1))))
    enlarged = cv2.resize(plate_crop_image, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_CUBIC)
    grayscale = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

    _, otsu_bin = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_bin = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(grayscale, -1, sharp_kernel)

    best_result = ""
    for candidate_img in [otsu_bin, adaptive_bin, sharpened, grayscale]:
        raw_ocr_output = _perform_easyocr_read(candidate_img, ocr_reader, thread_lock)
        if not raw_ocr_output:
            continue
        matched_plate = _find_best_registration_match(raw_ocr_output)
        if len(matched_plate) >= 8:
            return matched_plate
        if len(matched_plate) > len(best_result):
            best_result = matched_plate

    return best_result


# Backward compatibility export
read_plate = scan_license_plate
