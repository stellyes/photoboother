"""Perspective transformation for photo correction."""

from typing import Tuple

import cv2
import numpy as np


class PerspectiveTransformer:
    """Applies perspective transformation to correct skewed photos.

    Given 4 corner points of a detected photo, this class transforms
    the image to a flat, rectangular output with correct proportions.
    """

    def order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order points in consistent order: TL, TR, BR, BL.

        Args:
            pts: Array of 4 points (shape: 4x2).

        Returns:
            Ordered array of points: [top-left, top-right,
            bottom-right, bottom-left].
        """
        pts = pts.astype(np.float32)
        ordered = np.zeros((4, 2), dtype=np.float32)

        # Sum of coordinates: smallest = top-left, largest = bottom-right
        s = pts.sum(axis=1)
        ordered[0] = pts[np.argmin(s)]  # Top-left
        ordered[2] = pts[np.argmax(s)]  # Bottom-right

        # Difference of coordinates: smallest = top-right, largest = bottom-left
        d = np.diff(pts, axis=1).flatten()
        ordered[1] = pts[np.argmin(d)]  # Top-right
        ordered[3] = pts[np.argmax(d)]  # Bottom-left

        return ordered

    def compute_output_dimensions(
        self, pts: np.ndarray
    ) -> Tuple[int, int]:
        """Compute output dimensions preserving aspect ratio.

        Args:
            pts: Ordered array of 4 corner points.

        Returns:
            Tuple of (width, height) for the output image.
        """
        pts = self.order_points(pts)

        # Compute width as maximum of top and bottom edge lengths
        width_top = np.linalg.norm(pts[1] - pts[0])
        width_bottom = np.linalg.norm(pts[2] - pts[3])
        width = int(max(width_top, width_bottom))

        # Compute height as maximum of left and right edge lengths
        height_left = np.linalg.norm(pts[3] - pts[0])
        height_right = np.linalg.norm(pts[2] - pts[1])
        height = int(max(height_left, height_right))

        return width, height

    def transform(
        self,
        image: np.ndarray,
        pts: np.ndarray,
        output_size: Tuple[int, int] = None,
    ) -> np.ndarray:
        """Apply perspective transformation to extract the photo.

        Args:
            image: Source image in BGR format.
            pts: Array of 4 corner points defining the photo boundary.
            output_size: Optional (width, height) for output. If None,
                dimensions are computed to preserve aspect ratio.

        Returns:
            Transformed image with perspective correction applied.
        """
        # Order the points consistently
        ordered_pts = self.order_points(pts)

        # Compute output dimensions if not specified
        if output_size is None:
            width, height = self.compute_output_dimensions(ordered_pts)
        else:
            width, height = output_size

        # Ensure minimum dimensions
        width = max(width, 1)
        height = max(height, 1)

        # Define destination points (rectangle)
        dst_pts = np.array([
            [0, 0],              # Top-left
            [width - 1, 0],      # Top-right
            [width - 1, height - 1],  # Bottom-right
            [0, height - 1],     # Bottom-left
        ], dtype=np.float32)

        # Compute perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)

        # Apply the transformation
        transformed = cv2.warpPerspective(
            image, matrix, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        return transformed

    def get_transformation_matrix(
        self,
        pts: np.ndarray,
        output_size: Tuple[int, int] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Get the perspective transformation matrix without applying it.

        Useful for debugging or applying the same transformation to
        multiple images.

        Args:
            pts: Array of 4 corner points defining the photo boundary.
            output_size: Optional (width, height) for output.

        Returns:
            Tuple of (transformation matrix, output size).
        """
        ordered_pts = self.order_points(pts)

        if output_size is None:
            width, height = self.compute_output_dimensions(ordered_pts)
        else:
            width, height = output_size

        width = max(width, 1)
        height = max(height, 1)

        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)

        return matrix, (width, height)
