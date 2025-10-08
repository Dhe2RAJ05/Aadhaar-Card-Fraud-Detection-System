from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import re
import json
from PIL import Image, ImageEnhance
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from supervision import Detections, BoxAnnotator
import requests
import uuid

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

UPLOAD_FOLDER = "uploads"
CROPPED_FOLDER = "cropped_fields"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

REPO_CONFIG = dict(
    repo_id="arnabdhar/YOLOv8-nano-aadhar-card",
    filename="model.pt",
    local_dir=MODEL_FOLDER
)

OCR_API_KEY = "K83329092688957"
OCR_URL = "https://api.ocr.space/parse/image"

OCR_CORRECTIONS = {
    'O': '0', 'o': '0',
    'I': '1', 'l': '1', '|': '1',
    'S': '5', 's': '5',
    'B': '8', 'b': '8',
    'Z': '2', 'z': '2',
    'G': '6', 'g': '6'
}

def verhoeff_checksum(number):
    """
    Validates Aadhaar number using Verhoeff algorithm
    Returns True if valid, False otherwise
    
    The Verhoeff algorithm uses three tables:
    - Multiplication table (d)
    - Permutation table (p)
    """
    d = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
        [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
        [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
        [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
        [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
        [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    
    p = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
    ]
    
    number_str = str(number).replace(" ", "").strip()
    
    if len(number_str) != 12 or not number_str.isdigit():
        return False
    
    c = 0
    
    reversed_str = number_str[::-1]
    
    for i, digit_char in enumerate(reversed_str):
        digit = int(digit_char)
        p_index = i % 8
        permuted = p[p_index][digit]
        c = d[c][permuted]
    
    return c == 0


def generate_verhoeff_digit(number_str):
    """
    Generate the Verhoeff check digit for a given number
    Useful for testing and debugging
    """
    d = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
        [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
        [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
        [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
        [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
        [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]
    
    p = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
    ]
    
    inv = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]
    
    c = 0
    reversed_str = number_str[::-1]
    
    for i, digit_char in enumerate(reversed_str):
        digit = int(digit_char)
        c = d[c][p[(i + 1) % 8][digit]]
    
    return inv[c]


def preprocess_image_for_ocr(image_array):
    pil_img = Image.fromarray(image_array)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(2.0)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(2.0)
    pil_img = pil_img.convert('L')
    img_array = np.array(pil_img)
    _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_array


def ocr_space_file(image_path):
    with open(image_path, 'rb') as f:
        response = requests.post(
            OCR_URL,
            files={'filename': f},
            data={'apikey': OCR_API_KEY, 'OCREngine': 2}
        )
    try:
        text = response.json()['ParsedResults'][0]['ParsedText']
        return text.strip()
    except Exception:
        return ""


def auto_correct_ocr(text):
    corrected = ""
    for char in text:
        corrected += OCR_CORRECTIONS.get(char, char)
    return corrected


def extract_aadhaar_number(text):
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = auto_correct_ocr(text)
    match = re.search(r'\b(\d{4}\s?\d{4}\s?\d{4})\b', text)
    if match:
        return match.group(1).replace(" ", "")
    return None


def extract_dob(text):
    """
    Extract Date of Birth from text in various formats
    Supports: DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY/MM/DD, etc.
    Removes any alphabetic characters before extraction
    """
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    label_patterns = [
        r'(?:DoB|DOB|dob|Date of Birth)[:\s]*([0-9\/\-\.]+)',
    ]
    
    for pattern in label_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_part = match.group(1)
            # Clean the date part - remove any remaining letters
            date_part = re.sub(r'[a-zA-Z]', '', date_part)
            # Check if it matches a valid date pattern
            if re.match(r'\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4}', date_part):
                return date_part
            elif re.match(r'\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2}', date_part):
                return date_part
    
    # If no label found, remove ALL letters and search for date pattern
    cleaned_text = re.sub(r'[a-zA-Z]', ' ', text)
    
    # Common DOB patterns
    patterns = [
        r'\b(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})\b',  # DD/MM/YYYY or DD-MM-YYYY
        r'\b(\d{4}[\/\-\.]\d{2}[\/\-\.]\d{2})\b',  # YYYY/MM/DD or YYYY-MM-DD
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cleaned_text)
        if match:
            return match.group(1)
    
    return None


def remove_aadhaar_from_text(text):
    text = re.sub(r'\b\d{12}\b', '', text)
    text = re.sub(r'\b\d{4}\s+\d{4}\s+\d{4}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_gender(text):
    """
    Extract gender from text
    Looks for MALE, FEMALE, or other gender indicators
    """
    text_upper = text.upper().strip()
    text_clean = text.strip()
    
    text_upper = re.sub(r'[^A-Z\s]', '', text_upper).strip()
    
    if text_upper == 'MALE' or text_upper == 'M':
        return 'MALE'
    elif text_upper == 'FEMALE' or text_upper == 'F':
        return 'FEMALE'
    
    if 'FEMALE' in text_upper:
        return 'FEMALE'
    elif 'MALE' in text_upper:
        return 'MALE'
    
    if any(keyword in text_upper for keyword in ['TRANSGENDER', 'OTHER', 'THIRD']):
        return 'OTHER'
    
    if len(text_clean) <= 6:  
        if 'M' in text_upper and 'F' not in text_upper:
            return 'MALE'
        elif 'F' in text_upper:
            return 'FEMALE'
    
    return None


def clean_address(text):
    """
    Clean address text by removing common prefixes and extra whitespace
    """
    text = re.sub(r'^Address\s*:?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*:\s*', '', text)  # Remove leading colon
    
    # Remove Aadhaar numbers
    text = remove_aadhaar_from_text(text)
    
    return text.strip()


model_path = hf_hub_download(**REPO_CONFIG)
model = YOLO(model_path)
print("✅ YOLO model loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def process_aadhaar():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Sav uploaded file
    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Running YOLO detection
    img = Image.open(filepath).convert("RGB")
    img_np = np.array(img)
    results = model.predict(img_np, conf=0.25, verbose=False)
    detections = Detections.from_ultralytics(results[0])

    annotator = BoxAnnotator()
    annotated_image = annotator.annotate(scene=img_np.copy(), detections=detections)
    annotated_path = os.path.join("static", "annotated_output.jpg")
    cv2.imwrite(annotated_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    output_data = {"image": filename, "fields": []}
    details = {"name": "", "dob": "", "gender": "", "aadhaar_number": "", "address": ""}
    all_ocr_text = []  # Store all OCR text for fallback extraction

    for i, class_id in enumerate(detections.class_id):
        label = model.names[class_id]
        box = detections.xyxy[i].astype(int)
        x1, y1, x2, y2 = box

        crop_img = img_np[y1:y2, x1:x2].copy()
        crop_path = os.path.join(CROPPED_FOLDER, f"{label}_{i}.jpg")
        cv2.imwrite(crop_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))

        ocr_text = ocr_space_file(crop_path)
        all_ocr_text.append(ocr_text)  # Collect all OCR text

        # Field-specific logic
        if label.upper() in ["AADHAAR_NUMBER", "AADHAR_NUMBER", "AADHAAR_NO"]:
            aadhaar_num = extract_aadhaar_number(ocr_text)
            if aadhaar_num:
                details["aadhaar_number"] = f"{aadhaar_num[0:4]} {aadhaar_num[4:8]} {aadhaar_num[8:12]}"
        elif label.upper() == "ADDRESS":
            details["address"] = clean_address(ocr_text)
        elif label.upper() == "NAME":
            details["name"] = ocr_text.strip()
        elif label.upper() in ["DOB", "DATE_OF_BIRTH"]:
            dob = extract_dob(ocr_text)
            if dob:
                details["dob"] = dob
        elif label.upper() in ["GENDER", "SEX"]:
            gender = extract_gender(ocr_text)
            if gender:
                details["gender"] = gender

        output_data["fields"].append({
            "label": label,
            "ocr_text": ocr_text
        })

    # Fallback: If DOB not found, search in all OCR text
    if not details["dob"]:
        for text in all_ocr_text:
            dob = extract_dob(text)
            if dob:
                details["dob"] = dob
                break
    
    # Fallback: If gender not found, search in all OCR text
    if not details["gender"]:
        for text in all_ocr_text:
            gender = extract_gender(text)
            if gender:
                details["gender"] = gender
                break
    
    # Additional fallback: OCR the entire image if DOB or gender still not found
    if not details["dob"] or not details["gender"]:
        full_image_path = os.path.join(CROPPED_FOLDER, f"full_image_{uuid.uuid4()}.jpg")
        cv2.imwrite(full_image_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        full_ocr_text = ocr_space_file(full_image_path)
        
        if not details["dob"]:
            dob = extract_dob(full_ocr_text)
            if dob:
                details["dob"] = dob
        
        if not details["gender"]:
            gender = extract_gender(full_ocr_text)
            if gender:
                details["gender"] = gender

    # Verify Aadhaar using Verhoeff algorithm
    verhoeff_valid = False
    if details["aadhaar_number"]:
        verhoeff_valid = verhoeff_checksum(details["aadhaar_number"])

    # Save full result
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return jsonify({
        "details": details,
        "annotated_image": annotated_path,
        "verhoeff_valid": verhoeff_valid
    })


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/api/test-verhoeff', methods=['POST'])
def test_verhoeff():
    """Test endpoint to verify Verhoeff algorithm"""
    data = request.get_json()
    aadhaar = data.get('aadhaar', '')
    
    is_valid = verhoeff_checksum(aadhaar)
    
    # Also calculate what the check digit should be
    if len(aadhaar.replace(" ", "")) == 12:
        first_11 = aadhaar.replace(" ", "")[:11]
        expected_check_digit = generate_verhoeff_digit(first_11)
        actual_check_digit = aadhaar.replace(" ", "")[-1]
    else:
        expected_check_digit = None
        actual_check_digit = None
    
    return jsonify({
        "aadhaar": aadhaar,
        "is_valid": is_valid,
        "actual_check_digit": actual_check_digit,
        "expected_check_digit": str(expected_check_digit) if expected_check_digit else None
    })


if __name__ == '__main__':
    app.run(debug=True)