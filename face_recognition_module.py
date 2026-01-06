"""
Face detection module using MediaPipe.
Detects faces in photos - users manually tag names for search.
"""

import numpy as np
from PIL import Image
import io
import base64
from typing import List, Dict, Any, Optional, Tuple

# Try to import mediapipe, provide fallback if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not available. Face detection features disabled.")


def base64_to_numpy(base64_string: str) -> np.ndarray:
    """Convert base64 string to numpy array."""
    img_data = base64.b64decode(base64_string)
    pil_image = Image.open(io.BytesIO(img_data))

    # Convert to RGB
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    return np.array(pil_image)


def detect_faces(image_base64: str) -> List[Dict[str, Any]]:
    """
    Detect faces in an image using MediaPipe.

    Args:
        image_base64: Base64 encoded image

    Returns:
        List of face data with locations (no encodings - users tag manually)
    """
    if not MEDIAPIPE_AVAILABLE:
        return []

    try:
        image = base64_to_numpy(image_base64)
        height, width = image.shape[:2]

        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection

        with mp_face_detection.FaceDetection(
            model_selection=1,  # 1 = full range model (better for varied distances)
            min_detection_confidence=0.5
        ) as face_detection:

            # Process the image
            results = face_detection.process(image)

            if not results.detections:
                return []

            faces = []
            for i, detection in enumerate(results.detections):
                # Get bounding box (relative coordinates)
                bbox = detection.location_data.relative_bounding_box

                # Convert to absolute pixel coordinates
                left = int(bbox.xmin * width)
                top = int(bbox.ymin * height)
                right = int((bbox.xmin + bbox.width) * width)
                bottom = int((bbox.ymin + bbox.height) * height)

                # Clamp to image bounds
                left = max(0, left)
                top = max(0, top)
                right = min(width, right)
                bottom = min(height, bottom)

                # Get detection confidence
                confidence = detection.score[0] if detection.score else 0.0

                faces.append({
                    'face_id': i,
                    'location': {
                        'top': top,
                        'right': right,
                        'bottom': bottom,
                        'left': left
                    },
                    'confidence': float(confidence),
                    'name': None  # User will fill this in
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
    Detect faces in an image.

    Note: MediaPipe doesn't do recognition/matching. This function just
    detects faces - the known_faces parameter is ignored but kept for
    API compatibility with the old face_recognition-based code.

    Args:
        image_base64: Base64 encoded image
        known_faces: Ignored (kept for compatibility)
        tolerance: Ignored (kept for compatibility)

    Returns:
        List of detected faces (without automatic name matching)
    """
    # Just detect faces - no automatic recognition with MediaPipe
    return detect_faces(image_base64)


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


def is_available() -> bool:
    """Check if face detection is available."""
    return MEDIAPIPE_AVAILABLE
