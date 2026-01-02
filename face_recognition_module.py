"""
Facial recognition module for detecting faces and matching with known people.
"""

import numpy as np
from PIL import Image
import io
import base64
from typing import List, Dict, Any, Optional, Tuple

# Try to import face_recognition, provide fallback if not available
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition not available. Facial recognition features disabled.")


def base64_to_numpy(base64_string: str) -> np.ndarray:
    """Convert base64 string to numpy array for face_recognition."""
    img_data = base64.b64decode(base64_string)
    pil_image = Image.open(io.BytesIO(img_data))
    
    # Convert to RGB
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    return np.array(pil_image)


def detect_faces(image_base64: str) -> List[Dict[str, Any]]:
    """
    Detect faces in an image.
    
    Args:
        image_base64: Base64 encoded image
        
    Returns:
        List of face data with locations and encodings
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return []
    
    try:
        image = base64_to_numpy(image_base64)
        
        # Find face locations
        face_locations = face_recognition.face_locations(image, model='hog')
        
        if not face_locations:
            return []
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        faces = []
        for i, (location, encoding) in enumerate(zip(face_locations, face_encodings)):
            top, right, bottom, left = location
            faces.append({
                'face_id': i,
                'location': {
                    'top': top,
                    'right': right,
                    'bottom': bottom,
                    'left': left
                },
                'encoding': encoding.tolist(),
                'name': None  # To be filled in by user
            })
        
        return faces
    
    except Exception as e:
        print(f"Error detecting faces: {e}")
        return []


def recognize_faces(
    image_base64: str,
    known_faces: Dict[str, List[List[float]]],
    tolerance: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Detect and recognize faces in an image.
    
    Args:
        image_base64: Base64 encoded image
        known_faces: Dictionary mapping names to lists of face encodings
        tolerance: How strict the matching should be (lower = stricter)
        
    Returns:
        List of face data with locations, encodings, and recognized names
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return []
    
    try:
        image = base64_to_numpy(image_base64)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(image, model='hog')
        
        if not face_locations:
            return []
        
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Prepare known face data
        known_encodings = []
        known_names = []
        for name, encodings in known_faces.items():
            for enc in encodings:
                known_encodings.append(np.array(enc))
                known_names.append(name)
        
        faces = []
        for i, (location, encoding) in enumerate(zip(face_locations, face_encodings)):
            top, right, bottom, left = location
            
            recognized_name = None
            
            if known_encodings:
                # Compare to known faces
                matches = face_recognition.compare_faces(
                    known_encodings, encoding, tolerance=tolerance
                )
                
                if True in matches:
                    # Get face distances for all matches
                    face_distances = face_recognition.face_distance(
                        known_encodings, encoding
                    )
                    best_match_idx = np.argmin(face_distances)
                    
                    if matches[best_match_idx]:
                        recognized_name = known_names[best_match_idx]
            
            faces.append({
                'face_id': i,
                'location': {
                    'top': top,
                    'right': right,
                    'bottom': bottom,
                    'left': left
                },
                'encoding': encoding.tolist(),
                'name': recognized_name
            })
        
        return faces
    
    except Exception as e:
        print(f"Error recognizing faces: {e}")
        return []


def extract_face_thumbnail(
    image_base64: str,
    face_location: Dict[str, int],
    padding: int = 20,
    size: Tuple[int, int] = (100, 100)
) -> Optional[str]:
    """
    Extract a thumbnail of a specific face from an image.
    
    Args:
        image_base64: Base64 encoded image
        face_location: Dictionary with top, right, bottom, left
        padding: Pixels to add around the face
        size: Output thumbnail size
        
    Returns:
        Base64 encoded face thumbnail or None
    """
    try:
        img_data = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(img_data))
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        width, height = pil_image.size
        
        top = max(0, face_location['top'] - padding)
        right = min(width, face_location['right'] + padding)
        bottom = min(height, face_location['bottom'] + padding)
        left = max(0, face_location['left'] - padding)
        
        face_image = pil_image.crop((left, top, right, bottom))
        face_image = face_image.resize(size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        face_image.save(buffer, format='JPEG', quality=90)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    except Exception as e:
        print(f"Error extracting face thumbnail: {e}")
        return None


def get_face_encoding_from_image(image_base64: str) -> Optional[List[float]]:
    """
    Get a single face encoding from an image (expects one face).
    
    Useful for adding a new known face from a clear portrait.
    
    Args:
        image_base64: Base64 encoded image
        
    Returns:
        Face encoding as list of floats or None
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return None
    
    try:
        image = base64_to_numpy(image_base64)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            return encodings[0].tolist()
        return None
    
    except Exception as e:
        print(f"Error getting face encoding: {e}")
        return None


def compare_faces(
    encoding1: List[float],
    encoding2: List[float],
    tolerance: float = 0.6
) -> Tuple[bool, float]:
    """
    Compare two face encodings.
    
    Args:
        encoding1: First face encoding
        encoding2: Second face encoding
        tolerance: Matching tolerance
        
    Returns:
        Tuple of (is_match, distance)
    """
    if not FACE_RECOGNITION_AVAILABLE:
        return False, 1.0
    
    try:
        enc1 = np.array(encoding1)
        enc2 = np.array(encoding2)
        
        distance = face_recognition.face_distance([enc1], enc2)[0]
        is_match = distance <= tolerance
        
        return is_match, float(distance)
    
    except Exception as e:
        print(f"Error comparing faces: {e}")
        return False, 1.0


def is_available() -> bool:
    """Check if facial recognition is available."""
    return FACE_RECOGNITION_AVAILABLE
