"""Robust photo boundary detection using multiple techniques."""

from typing import Optional, List, Tuple
import cv2
import numpy as np


class EdgeDetector:
    """Detects photo boundaries using multiple edge detection strategies.

    Combines Canny edge detection with adaptive thresholding, color
    segmentation, and morphological operations for robust detection
    across varied backgrounds.
    """

    # Typical photo aspect ratios (width/height)
    PHOTO_ASPECT_RATIOS = [
        (4, 6),   # 4x6
        (5, 7),   # 5x7
        (3, 5),   # 3x5
        (8, 10),  # 8x10
        (3, 4),   # 3x4 (standard)
        (2, 3),   # 2x3
        (1, 1),   # Square
    ]

    def __init__(
        self,
        min_area_ratio: float = 0.08,
        max_area_ratio: float = 0.95,
        contour_epsilon: float = 0.02,
    ):
        """Initialize the edge detector.

        Args:
            min_area_ratio: Minimum contour area as ratio of image area.
            max_area_ratio: Maximum contour area as ratio of image area.
            contour_epsilon: Epsilon factor for contour approximation.
        """
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.contour_epsilon = contour_epsilon

    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect photo boundary using multiple methods.

        Tries several detection strategies and returns the best result.

        Args:
            image: Input image in BGR format.

        Returns:
            Array of 4 corner points or None if no boundary found.
        """
        # Downscale large images for faster processing
        scale = 1.0
        max_dim = 1500
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            proc_image = cv2.resize(image, None, fx=scale, fy=scale)
        else:
            proc_image = image

        candidates = []

        # Method 1: Multi-threshold Canny
        for low, high in [(30, 100), (50, 150), (75, 200), (20, 80)]:
            contour = self._detect_canny(proc_image, low, high)
            if contour is not None:
                candidates.append(('canny', contour))

        # Method 2: Adaptive threshold
        contour = self._detect_adaptive_threshold(proc_image)
        if contour is not None:
            candidates.append(('adaptive', contour))

        # Method 3: Color-based segmentation
        contour = self._detect_color_segmentation(proc_image)
        if contour is not None:
            candidates.append(('color', contour))

        # Method 4: Saturation-based detection
        contour = self._detect_saturation(proc_image)
        if contour is not None:
            candidates.append(('saturation', contour))

        # Method 5: Combined edge detection
        contour = self._detect_combined_edges(proc_image)
        if contour is not None:
            candidates.append(('combined', contour))

        # Method 6: GrabCut foreground segmentation
        contour = self._detect_grabcut(proc_image)
        if contour is not None:
            candidates.append(('grabcut', contour))

        if not candidates:
            return None

        # Score and select best candidate
        best_contour = self._select_best_contour(proc_image, candidates)

        # Scale back if we downscaled
        if scale != 1.0 and best_contour is not None:
            best_contour = (best_contour / scale).astype(np.float32)

        # Apply inward margin to tighten the crop
        if best_contour is not None:
            best_contour = self._shrink_contour(best_contour, image.shape, margin_percent=0.03)

        return best_contour

    def _detect_canny(
        self, image: np.ndarray, low: int, high: int
    ) -> Optional[np.ndarray]:
        """Detect using Canny edge detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)

        # Canny edge detection
        edges = cv2.Canny(enhanced, low, high)

        # Morphological closing to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        return self._find_best_quadrilateral(image, edges)

    def _detect_adaptive_threshold(
        self, image: np.ndarray
    ) -> Optional[np.ndarray]:
        """Detect using adaptive thresholding."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Get edges from threshold
        edges = cv2.Canny(thresh, 50, 150)

        return self._find_best_quadrilateral(image, edges)

    def _detect_color_segmentation(
        self, image: np.ndarray
    ) -> Optional[np.ndarray]:
        """Detect using color difference from background."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Get edge colors (likely background)
        edge_size = 20
        h, w = image.shape[:2]

        # Sample background from edges
        bg_samples = np.concatenate([
            l[:edge_size, :].flatten(),
            l[-edge_size:, :].flatten(),
            l[:, :edge_size].flatten(),
            l[:, -edge_size:].flatten(),
        ])
        bg_mean = np.mean(bg_samples)
        bg_std = np.std(bg_samples)

        # Create mask for pixels different from background
        diff = np.abs(l.astype(float) - bg_mean)
        mask = (diff > bg_std * 1.5).astype(np.uint8) * 255

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Get edges
        edges = cv2.Canny(mask, 50, 150)
        edges = cv2.dilate(edges, kernel, iterations=1)

        return self._find_best_quadrilateral(image, edges)

    def _detect_saturation(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect using saturation differences (photos often more colorful)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)

        # Photos typically have higher saturation than wood/plain backgrounds
        # Use Otsu's threshold on saturation
        _, mask = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Fill holes
        mask_filled = mask.copy()
        h, w = mask.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(mask_filled, flood_mask, (0, 0), 255)
        mask_filled = cv2.bitwise_not(mask_filled)
        mask = cv2.bitwise_or(mask, mask_filled)

        edges = cv2.Canny(mask, 50, 150)

        return self._find_best_quadrilateral(image, edges)

    def _detect_combined_edges(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Combine multiple edge detection methods."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Sobel edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel_max = sobel.max()
        if sobel_max > 0:
            sobel = (sobel / sobel_max * 255).astype(np.uint8)
        else:
            sobel = np.zeros_like(gray)

        # Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.abs(laplacian)
        lap_max = laplacian.max()
        if lap_max > 0:
            laplacian = (laplacian / lap_max * 255).astype(np.uint8)
        else:
            laplacian = np.zeros_like(gray)

        # Combine
        combined = cv2.addWeighted(sobel, 0.5, laplacian, 0.5, 0)

        # Threshold
        _, edges = cv2.threshold(combined, 30, 255, cv2.THRESH_BINARY)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        return self._find_best_quadrilateral(image, edges)

    def _detect_grabcut(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Use GrabCut for foreground segmentation."""
        h, w = image.shape[:2]

        # Initialize mask
        mask = np.zeros((h, w), np.uint8)

        # Define probable foreground rectangle (center region)
        margin = 0.1
        rect = (
            int(w * margin),
            int(h * margin),
            int(w * (1 - 2 * margin)),
            int(h * (1 - 2 * margin))
        )

        # GrabCut models
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)

            # Create binary mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)

            # Clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel)

            # Get edges
            edges = cv2.Canny(mask2, 50, 150)

            return self._find_best_quadrilateral(image, edges)
        except Exception:
            return None

    def _shrink_contour(
        self, contour: np.ndarray, image_shape: Tuple[int, int],
        margin_percent: float = 0.01
    ) -> np.ndarray:
        """Shrink contour inward by a margin to tighten the crop."""
        points = contour.astype(np.float32)

        # Calculate centroid
        centroid = points.mean(axis=0)

        # Move each point toward centroid
        shrunk = []
        for pt in points:
            direction = centroid - pt
            dist = np.linalg.norm(direction)
            if dist > 0:
                # Shrink by margin_percent of the distance to centroid
                shrink_amount = dist * margin_percent * 2
                new_pt = pt + (direction / dist) * shrink_amount
            else:
                new_pt = pt
            shrunk.append(new_pt)

        return np.array(shrunk, dtype=np.float32)

    def _find_best_quadrilateral(
        self, image: np.ndarray, edges: np.ndarray
    ) -> Optional[np.ndarray]:
        """Find the best quadrilateral contour from edge map."""
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        image_area = image.shape[0] * image.shape[1]
        min_area = image_area * self.min_area_ratio
        max_area = image_area * self.max_area_ratio

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area or area > max_area:
                continue

            # Approximate to polygon
            perimeter = cv2.arcLength(contour, True)
            epsilon = self.contour_epsilon * perimeter

            # Try different epsilon values
            for eps_mult in [0.5, 1.0, 1.5, 2.0]:
                approx = cv2.approxPolyDP(contour, epsilon * eps_mult, True)

                if len(approx) == 4:
                    # Score this quadrilateral
                    score = self._score_quadrilateral(approx, image_area, image.shape)
                    candidates.append((score, approx.reshape(4, 2)))
                    break

        if not candidates:
            # Fallback: use minimum area rectangle of largest contour
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= min_area:
                rect = cv2.minAreaRect(largest)
                box = cv2.boxPoints(rect)
                return box.astype(np.float32)
            return None

        # Return highest scoring quadrilateral
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1].astype(np.float32)

    def _score_quadrilateral(
        self, quad: np.ndarray, image_area: float,
        image_shape: Tuple[int, int] = None
    ) -> float:
        """Score a quadrilateral based on photo-likeness."""
        points = quad.reshape(4, 2)
        area = cv2.contourArea(points)

        # Penalize very large contours (likely whole image)
        area_ratio = area / image_area
        if area_ratio > 0.9:
            return 0.1  # Very low score for near-full-image contours

        # Prefer medium-sized contours (10-70% of image)
        if 0.1 <= area_ratio <= 0.7:
            area_score = 0.8 + (0.2 * (1 - abs(area_ratio - 0.4) / 0.3))
        else:
            area_score = area_ratio * 0.5

        # Convexity score
        hull = cv2.convexHull(points)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0

        # Aspect ratio score (prefer photo-like ratios)
        rect = cv2.minAreaRect(points)
        w, h = rect[1]
        if w == 0 or h == 0:
            return 0

        aspect = max(w, h) / min(w, h)
        aspect_score = self._aspect_ratio_score(aspect)

        # Angle score (prefer roughly aligned rectangles)
        angle = rect[2]
        angle_score = 1.0 - min(abs(angle), abs(90 - abs(angle))) / 45

        # Edge proximity penalty - penalize contours touching image edges
        edge_penalty = 1.0
        if image_shape is not None:
            h_img, w_img = image_shape[:2]
            margin = 0.02  # 2% margin
            min_x, min_y = points.min(axis=0)
            max_x, max_y = points.max(axis=0)

            touches_edge = (
                min_x < w_img * margin or
                min_y < h_img * margin or
                max_x > w_img * (1 - margin) or
                max_y > h_img * (1 - margin)
            )
            if touches_edge:
                edge_penalty = 0.5  # Significant penalty for edge-touching

        # Combined score
        score = (
            area_score * 0.25 +
            convexity * 0.25 +
            aspect_score * 0.25 +
            angle_score * 0.1 +
            edge_penalty * 0.15
        )

        return score

    def _aspect_ratio_score(self, aspect: float) -> float:
        """Score aspect ratio based on common photo sizes."""
        best_score = 0
        for w, h in self.PHOTO_ASPECT_RATIOS:
            target = max(w, h) / min(w, h)
            diff = abs(aspect - target)
            score = max(0, 1 - diff / 0.5)
            best_score = max(best_score, score)
        return best_score

    def _select_best_contour(
        self,
        image: np.ndarray,
        candidates: List[Tuple[str, np.ndarray]]
    ) -> Optional[np.ndarray]:
        """Select the best contour from multiple detection methods."""
        if not candidates:
            return None

        image_area = image.shape[0] * image.shape[1]

        scored = []
        for method, contour in candidates:
            score = self._score_quadrilateral(
                contour.reshape(4, 1, 2), image_area, image.shape
            )
            scored.append((score, method, contour))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][2]

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Legacy method for compatibility - returns Canny edges."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, 50, 150)

    def find_photo_contour(
        self, image: np.ndarray, edges: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Legacy method for compatibility."""
        return self.detect(image)
