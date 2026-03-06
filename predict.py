# -*- coding: utf-8 -*-
"""
Parkinson's Disease Prediction Module
Proper UTF-8 encoding for emojis
"""

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import os

# Load your trained model
MODEL_PATH = os.path.join("models", "parkinson_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please check the path!")

model = load_model(MODEL_PATH)
print(f"✅ Model loaded successfully from {MODEL_PATH}")

# Initialize face detection cascade
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)


def is_valid_spiral_image(file_path):
    """
    Improved validation for spiral drawings - more lenient for actual spirals.
    Returns: (is_valid, error_message)
    """
    try:
        img = cv2.imread(file_path)
        if img is None:
            return False, "Unable to read the image file"
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. CRITICAL: Detect faces (reject selfies/photos of people)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        if len(faces) > 0:
            return False, "⚠️ Please upload only a spiral drawing, not a photograph of a person"
        
        # 2. Check aspect ratio (spirals can be various shapes)
        aspect_ratio = w / h
        if aspect_ratio < 0.3 or aspect_ratio > 3.5:
            return False, "⚠️ Image aspect ratio is unusual. Please upload a proper spiral drawing"
        
        # 3. Edge detection - be more lenient
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 20, 80)  # Lower thresholds to catch lighter spirals
        edge_density = np.sum(edges > 0) / edges.size
        
        # Too few edges = completely blank
        if edge_density < 0.001:
            return False, "⚠️ Image appears to be blank or has no visible drawing"
        
        # Only reject if EXTREMELY dense (like very complex photos)
        if edge_density > 0.7:
            return False, "⚠️ Image is too complex. Please upload a simple spiral drawing"
        
        # 4. Check for text/documents - be more selective
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Only count very small components (typical of text)
        text_like_components = sum(1 for i in range(1, num_labels) if 5 < stats[i, cv2.CC_STAT_AREA] < 200)
        
        # Only reject if there are MANY small text-like components
        if text_like_components > 150:
            return False, "⚠️ This appears to be a text document. Please upload a hand-drawn spiral"
        
        # 5. Check contours - be more lenient
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return False, "⚠️ No drawing detected in the image"
        
        # Only reject if there are MANY distinct objects (like a busy photo)
        if len(contours) > 80:
            return False, "⚠️ Image contains too many separate objects. Please upload a single spiral drawing"
        
        # 6. Check if image is completely blank or solid
        largest_contour = max(contours, key=cv2.contourArea)
        contour_area = cv2.contourArea(largest_contour)
        image_area = h * w
        area_ratio = contour_area / image_area
        
        if area_ratio < 0.002:
            return False, "⚠️ Drawing is too faint or small"
        
        if area_ratio > 0.98:
            return False, "⚠️ Image appears to be completely filled"
        
        # 7. Color saturation check (reject colorful photos) - be more lenient
        if len(img.shape) == 3:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            avg_saturation = np.mean(saturation)
            
            # Only reject if VERY colorful (like nature photos)
            if avg_saturation > 100:
                return False, "⚠️ Image appears to be a colorful photograph. Please upload a spiral drawing"
        
        # 8. Check for straight lines (graphs/charts have many)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=80, maxLineGap=10)
        if lines is not None and len(lines) > 30:
            return False, "⚠️ This appears to be a graph or chart. Please upload a hand-drawn spiral"
        
        # 9. Brightness check - reject if too dark or too bright (likely bad photo)
        avg_brightness = np.mean(gray)
        if avg_brightness < 20 or avg_brightness > 250:
            return False, "⚠️ Image is too dark or too bright. Please upload a clear photo of your spiral drawing"
        
        # If all checks pass - it's likely a valid spiral
        return True, "Valid spiral image"
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}"


def prepare_image(file_path, target_size=(128, 128)):
    """
    Prepares image for model prediction.
    """
    try:
        image = Image.open(file_path).convert("RGB")
        image = image.resize(target_size)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def get_treatment_suggestions():
    """
    Returns treatment suggestions for Parkinson's disease.
    """
    return {
        "medications": [
            "Levodopa (most effective medication)",
            "Dopamine agonists (Pramipexole, Ropinirole)",
            "MAO-B inhibitors (Selegiline, Rasagiline)",
            "COMT inhibitors (Entacapone)"
        ],
        "therapies": [
            "Physical therapy to improve mobility and balance",
            "Occupational therapy for daily activities",
            "Speech therapy for communication difficulties",
            "Regular exercise (walking, swimming, tai chi)"
        ],
        "lifestyle": [
            "Maintain a healthy diet rich in fruits and vegetables",
            "Stay physically active with regular exercise",
            "Get adequate sleep and rest",
            "Join support groups for emotional support"
        ],
        "important_note": "⚠️ IMPORTANT: This is a screening tool only. Please consult a qualified neurologist or movement disorder specialist for proper diagnosis and treatment plan. Early detection and treatment can significantly improve quality of life."
    }


def predict_result(file_path):
    """
    Predicts whether the spiral drawing indicates Parkinson's disease.
    Returns: (prediction_label, confidence_percentage, treatment_suggestions)
    """
    try:
        # Step 1: Validate if it's a spiral image
        is_valid, validation_message = is_valid_spiral_image(file_path)
        
        if not is_valid:
            return validation_message, 0.0, None
        
        # Step 2: Prepare and predict
        prepared_image = prepare_image(file_path)
        prediction = model.predict(prepared_image, verbose=0)
        pred_value = float(prediction[0][0])
        
        # Step 3: Interpret results
        treatment_suggestions = None
        
        if pred_value >= 0.5:
            result = "⚠️ The patient has Parkinson's Disease"
            confidence = pred_value * 100
            treatment_suggestions = get_treatment_suggestions()
        else:
            result = "✅ The patient does NOT have Parkinson's Disease"
            confidence = (1 - pred_value) * 100
        
        return result, round(confidence, 2), treatment_suggestions
    
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return "❌ Error during prediction. Please try again.", 0.0, None


# Test validation system
def test_validation():
    print("\n" + "="*60)
    print("Spiral Validation System - Optimized for Real Spirals")
    print("="*60)
    print("✅ Face detection: Active (reject selfies)")
    print("✅ Edge analysis: Lenient (accept light/shaky spirals)")
    print("✅ Text detection: Active (reject documents)")
    print("✅ Graph detection: Active (reject charts)")
    print("✅ Photo detection: Active (reject nature/indoor photos)")
    print("="*60 + "\n")

test_validation()