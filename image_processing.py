"""
Image processing module for auto-cropping, orientation, and thumbnail generation.
"""

import cv2
import numpy as np
from PIL import Image, ExifTags
import io
import base64
from typing import Tuple, Optional, List


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
    
    This function detects a photo (like a photo strip or printed photo)
    against a contrasting background and extracts just the photo.
    
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
    
    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    l_blurred = cv2.GaussianBlur(l_channel, (5, 5), 0)
    
    best_result = None
    best_score = 0
    
    # Determine if background is dark or light by sampling corners
    corner_size = min(50, height // 10, width // 10)
    corners = [
        gray[0:corner_size, 0:corner_size].mean(),
        gray[0:corner_size, -corner_size:].mean(),
        gray[-corner_size:, 0:corner_size].mean(),
        gray[-corner_size:, -corner_size:].mean()
    ]
    avg_corner_brightness = np.mean(corners)
    dark_background = avg_corner_brightness < 127
    
    # Method: Multiple threshold values with aggressive morphology
    for thresh_val in [60, 80, 100, 120, 140, 160, 180]:
        _, thresh = cv2.threshold(l_blurred, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Fill holes aggressively - the photo content may have dark areas
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        
        # Close to fill holes, then open to remove small noise
        filled = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel_open, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Skip if too small or too large
            if area < image_area * min_area_ratio or area > image_area * 0.92:
                continue
            
            # Get minimum area rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            rect_w, rect_h = rect[1]
            
            if rect_w == 0 or rect_h == 0:
                continue
            
            rect_area = rect_w * rect_h
            
            # Calculate how rectangular the contour is
            # Higher is better (1.0 = perfect rectangle)
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            # Calculate aspect ratio (we want reasonable ratios)
            aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
            
            # Score based on rectangularity and area
            # Prefer larger, more rectangular contours
            # But also prefer contours away from image edges
            
            # Check if contour touches image edges (likely background)
            x, y, w, h = cv2.boundingRect(contour)
            edge_margin = 20
            touches_edge = (x < edge_margin or y < edge_margin or 
                          x + w > width - edge_margin or y + h > height - edge_margin)
            
            # Calculate score
            area_ratio = area / image_area
            score = rectangularity * area_ratio
            
            # Penalize if too close to full image size or touching edges
            if area_ratio > 0.85:
                score *= 0.5
            if touches_edge and area_ratio > 0.7:
                score *= 0.7
            
            # Accept if rectangularity is reasonable (lower threshold since photos have content)
            if rectangularity > 0.45 and aspect < 10 and score > best_score:
                best_score = score
                best_result = np.array(box, dtype=np.float32)
    
    # Also try with inverted threshold for light backgrounds
    if best_result is None or best_score < 0.1:
        for thresh_val in [60, 80, 100, 120, 140, 160, 180]:
            _, thresh = cv2.threshold(l_blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
            
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            
            filled = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=3)
            filled = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel_open, iterations=2)
            
            contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < image_area * min_area_ratio or area > image_area * 0.92:
                    continue
                
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                rect_w, rect_h = rect[1]
                
                if rect_w == 0 or rect_h == 0:
                    continue
                
                rect_area = rect_w * rect_h
                rectangularity = area / rect_area if rect_area > 0 else 0
                aspect = max(rect_w, rect_h) / min(rect_w, rect_h)
                
                x, y, w, h = cv2.boundingRect(contour)
                edge_margin = 20
                touches_edge = (x < edge_margin or y < edge_margin or 
                              x + w > width - edge_margin or y + h > height - edge_margin)
                
                area_ratio = area / image_area
                score = rectangularity * area_ratio
                
                if area_ratio > 0.85:
                    score *= 0.5
                if touches_edge and area_ratio > 0.7:
                    score *= 0.7
                
                if rectangularity > 0.45 and aspect < 10 and score > best_score:
                    best_score = score
                    best_result = np.array(box, dtype=np.float32)
    
    # Apply perspective transform if we found a good result
    if best_result is not None and best_score > 0.05:
        cropped_image = four_point_transform(original, best_result)
        
        # Verify the crop is meaningfully different from original
        crop_h, crop_w = cropped_image.shape[:2]
        crop_area = crop_h * crop_w
        
        if crop_area < image_area * 0.90:  # At least 10% smaller
            _, buffer = cv2.imencode('.jpg', cropped_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return buffer.tobytes(), True
    
    # Return original if no meaningful crop detected
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
