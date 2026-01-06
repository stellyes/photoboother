"""
Image processing module for auto-cropping, orientation, and thumbnail generation.
"""

import cv2
import numpy as np
from PIL import Image, ExifTags
import io
import base64
from typing import Tuple, Optional, List

from scanner import EdgeDetector, PerspectiveTransformer


def base64_to_cv2(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image."""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def cv2_to_base64(image: np.ndarray, format: str = 'JPEG') -> str:
    """Convert OpenCV image to base64 string."""
    if format.upper() == 'JPEG':
        ext = '.jpg'
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    else:
        ext = '.png'
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    
    _, buffer = cv2.imencode(ext, image, params)
    return base64.b64encode(buffer).decode('utf-8')


def pil_to_base64(image: Image.Image, format: str = 'JPEG') -> str:
    """Convert PIL image to base64 string."""
    buffer = io.BytesIO()
    if format.upper() == 'JPEG':
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=95)
    else:
        image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_pil(base64_string: str) -> Image.Image:
    """Convert base64 string to PIL image."""
    img_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_data))


def bytes_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')


def fix_orientation_from_exif(image: Image.Image) -> Image.Image:
    """
    Fix image orientation based on EXIF data.
    
    Args:
        image: PIL Image
        
    Returns:
        Rotated image if EXIF orientation found
    """
    try:
        exif = image._getexif()
        if exif is None:
            return image
        
        orientation_key = None
        for key, val in ExifTags.TAGS.items():
            if val == 'Orientation':
                orientation_key = key
                break
        
        if orientation_key is None or orientation_key not in exif:
            return image
        
        orientation = exif[orientation_key]
        
        if orientation == 2:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            return image.rotate(180, expand=True)
        elif orientation == 4:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            return image.transpose(Image.FLIP_LEFT_RIGHT).rotate(270, expand=True)
        elif orientation == 6:
            return image.rotate(270, expand=True)
        elif orientation == 7:
            return image.transpose(Image.FLIP_LEFT_RIGHT).rotate(90, expand=True)
        elif orientation == 8:
            return image.rotate(90, expand=True)
        
        return image
    except Exception:
        return image


def rotate_image(image: Image.Image, degrees: int) -> Image.Image:
    """
    Rotate image by specified degrees.
    
    Args:
        image: PIL Image
        degrees: Rotation in degrees (0, 90, 180, 270)
        
    Returns:
        Rotated image
    """
    if degrees == 0:
        return image
    # PIL rotates counter-clockwise, so negate for clockwise
    return image.rotate(-degrees, expand=True)


def create_thumbnail(image: Image.Image, max_size: Tuple[int, int] = (300, 300)) -> Image.Image:
    """
    Create a thumbnail of the image.
    
    Args:
        image: PIL Image
        max_size: Maximum dimensions
        
    Returns:
        Thumbnail image
    """
    thumbnail = image.copy()
    thumbnail.thumbnail(max_size, Image.Resampling.LANCZOS)
    return thumbnail


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in: top-left, top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Sum and difference to find corners
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply perspective transform to get a top-down view of the document.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute width
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute height
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Compute perspective transform and apply
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def auto_crop_photo(
    image_bytes: bytes,
    min_area_ratio: float = 0.05,
    padding: int = 5
) -> Tuple[Optional[bytes], bool]:
    """
    Automatically crop a photo from a high-contrast background.

    Uses the EdgeDetector from python-photo-scanner for robust multi-method
    boundary detection including Canny, adaptive threshold, color segmentation,
    saturation-based detection, and GrabCut.

    Args:
        image_bytes: Raw image bytes
        min_area_ratio: Minimum contour area as ratio of image area
        padding: Padding to add around detected photo

    Returns:
        Tuple of (cropped image bytes or None, whether crop was applied)
    """
    # Load image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return None, False

    original = image.copy()
    height, width = image.shape[:2]
    image_area = height * width

    # Use EdgeDetector from python-photo-scanner for robust detection
    detector = EdgeDetector(
        min_area_ratio=min_area_ratio,
        max_area_ratio=0.95,
        contour_epsilon=0.02
    )

    # Detect photo boundary using multiple methods
    boundary = detector.detect(image)

    if boundary is None:
        # No boundary detected, return original
        return image_bytes, False

    # Use PerspectiveTransformer for the perspective correction
    transformer = PerspectiveTransformer()
    cropped_image = transformer.transform(original, boundary)

    # Verify the crop is meaningfully different from original
    crop_h, crop_w = cropped_image.shape[:2]
    crop_area = crop_h * crop_w

    if crop_area < image_area * 0.90:  # At least 10% smaller
        _, buffer = cv2.imencode('.jpg', cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes(), True

    # Return original if crop isn't significantly different
    return image_bytes, False


def auto_orient_photo(image: Image.Image) -> Tuple[Image.Image, int]:
    """
    Attempt to auto-orient a photo to be upright.
    
    Uses aspect ratio and edge detection heuristics.
    
    Args:
        image: PIL Image
        
    Returns:
        Tuple of (oriented image, rotation applied in degrees)
    """
    # First apply EXIF orientation
    image = fix_orientation_from_exif(image)
    
    width, height = image.size
    
    # For photo strips (tall and narrow), they should typically be vertical
    aspect = width / height
    
    # If significantly wider than tall, might need rotation
    if aspect > 1.5:
        # Photo strip is likely sideways
        # Try to detect which way to rotate using edge detection
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Check edge density in different orientations
        h, w = edges.shape
        
        # For a photo strip, we expect more horizontal edges when properly oriented
        # Sum edges in different directions
        horizontal_edges = np.sum(edges[:, w//4:3*w//4])
        vertical_edges = np.sum(edges[h//4:3*h//4, :])
        
        if horizontal_edges > vertical_edges * 1.2:
            # More horizontal structure, rotate 90 degrees
            return image.rotate(-90, expand=True), 90
        else:
            return image.rotate(90, expand=True), 270
    
    return image, 0


def process_uploaded_image(
    image_bytes: bytes,
    auto_crop: bool = True,
    auto_orient: bool = True
) -> Tuple[str, str, str, int]:
    """
    Process an uploaded image with optional auto-crop and auto-orient.
    
    Args:
        image_bytes: Raw image bytes
        auto_crop: Whether to attempt auto-cropping
        auto_orient: Whether to attempt auto-orientation
        
    Returns:
        Tuple of (processed_base64, thumbnail_base64, original_base64, orientation)
    """
    # Store original
    original_base64 = bytes_to_base64(image_bytes)
    
    # Attempt auto-crop
    processed_bytes = image_bytes
    if auto_crop:
        cropped_bytes, was_cropped = auto_crop_photo(image_bytes)
        if cropped_bytes:
            processed_bytes = cropped_bytes
    
    # Convert to PIL for further processing
    pil_image = Image.open(io.BytesIO(processed_bytes))
    
    # Fix EXIF orientation
    pil_image = fix_orientation_from_exif(pil_image)
    
    # Auto-orient
    orientation = 0
    if auto_orient:
        pil_image, orientation = auto_orient_photo(pil_image)
    
    # Convert to RGB if necessary
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Create thumbnail
    thumbnail = create_thumbnail(pil_image)
    
    # Convert to base64
    processed_base64 = pil_to_base64(pil_image)
    thumbnail_base64 = pil_to_base64(thumbnail)
    
    return processed_base64, thumbnail_base64, original_base64, orientation
