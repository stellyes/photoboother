"""
Interactive crop editor component for manual refinement of auto-detected photo boundaries.

Features:
- Displays original image with bounding box overlay
- Draggable corner handles for adjusting crop area
- Magnifier window that follows cursor during drag
- Batch processing support with navigation
"""

import streamlit as st
import streamlit.components.v1 as components
import base64
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


def get_crop_editor_html(
    image_base64: str,
    initial_corners: List[List[float]],
    image_width: int,
    image_height: int,
    editor_id: str = "crop_editor"
) -> str:
    """Generate HTML/JS for the interactive crop editor component."""

    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    .crop-container {{
        position: relative;
        display: inline-block;
        cursor: crosshair;
        user-select: none;
    }}
    .crop-canvas {{
        display: block;
        max-width: 100%;
        height: auto;
    }}
    .magnifier {{
        position: absolute;
        width: 120px;
        height: 120px;
        border: 3px solid #4CAF50;
        border-radius: 50%;
        pointer-events: none;
        display: none;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        z-index: 1000;
    }}
    .magnifier-canvas {{
        position: absolute;
    }}
    .corner-info {{
        position: absolute;
        background: rgba(0,0,0,0.7);
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-family: monospace;
        pointer-events: none;
        white-space: nowrap;
    }}
    .instructions {{
        background: #f0f2f6;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 14px;
        color: #333;
    }}
    .instructions strong {{
        color: #4CAF50;
    }}
</style>
</head>
<body>
<div class="instructions">
    <strong>Drag the corner handles</strong> to adjust the crop area.
    A magnifier will appear to help with precise positioning.
</div>
<div class="crop-container" id="{editor_id}_container">
    <canvas class="crop-canvas" id="{editor_id}_canvas"></canvas>
    <div class="magnifier" id="{editor_id}_magnifier">
        <canvas class="magnifier-canvas" id="{editor_id}_mag_canvas"></canvas>
    </div>
    <div class="corner-info" id="{editor_id}_info" style="display:none;"></div>
</div>

<script>
(function() {{
    const editorId = "{editor_id}";
    const container = document.getElementById(editorId + "_container");
    const canvas = document.getElementById(editorId + "_canvas");
    const ctx = canvas.getContext("2d");
    const magnifier = document.getElementById(editorId + "_magnifier");
    const magCanvas = document.getElementById(editorId + "_mag_canvas");
    const magCtx = magCanvas.getContext("2d");
    const cornerInfo = document.getElementById(editorId + "_info");

    const HANDLE_RADIUS = 12;
    const HANDLE_HIT_RADIUS = 20;
    const MAG_SIZE = 120;
    const MAG_ZOOM = 3;

    magCanvas.width = MAG_SIZE;
    magCanvas.height = MAG_SIZE;

    let originalWidth = {image_width};
    let originalHeight = {image_height};
    let displayScale = 1;

    // Corner points in original image coordinates
    let corners = {json.dumps(initial_corners)};

    let image = new Image();
    let imageLoaded = false;
    let dragging = false;
    let dragCornerIndex = -1;
    let lastSentCorners = null;

    // Load the image
    image.onload = function() {{
        imageLoaded = true;

        // Calculate display size (max 800px width while maintaining aspect ratio)
        const maxWidth = Math.min(800, container.parentElement.clientWidth - 40);
        displayScale = Math.min(1, maxWidth / originalWidth);

        canvas.width = originalWidth * displayScale;
        canvas.height = originalHeight * displayScale;

        draw();
    }};
    image.src = "data:image/jpeg;base64,{image_base64}";

    function draw() {{
        if (!imageLoaded) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw the image
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

        // Draw semi-transparent overlay outside the crop area
        ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Clear the crop area (show original image)
        ctx.save();
        ctx.beginPath();
        const scaledCorners = corners.map(c => [c[0] * displayScale, c[1] * displayScale]);
        ctx.moveTo(scaledCorners[0][0], scaledCorners[0][1]);
        for (let i = 1; i < scaledCorners.length; i++) {{
            ctx.lineTo(scaledCorners[i][0], scaledCorners[i][1]);
        }}
        ctx.closePath();
        ctx.clip();
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        ctx.restore();

        // Draw the bounding box
        ctx.strokeStyle = "#4CAF50";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(scaledCorners[0][0], scaledCorners[0][1]);
        for (let i = 1; i < scaledCorners.length; i++) {{
            ctx.lineTo(scaledCorners[i][0], scaledCorners[i][1]);
        }}
        ctx.closePath();
        ctx.stroke();

        // Draw corner handles
        const cornerLabels = ["TL", "TR", "BR", "BL"];
        for (let i = 0; i < scaledCorners.length; i++) {{
            const [x, y] = scaledCorners[i];

            // Outer circle (white)
            ctx.beginPath();
            ctx.arc(x, y, HANDLE_RADIUS, 0, Math.PI * 2);
            ctx.fillStyle = i === dragCornerIndex ? "#FFC107" : "#ffffff";
            ctx.fill();
            ctx.strokeStyle = "#4CAF50";
            ctx.lineWidth = 3;
            ctx.stroke();

            // Inner dot
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fillStyle = "#4CAF50";
            ctx.fill();

            // Label
            ctx.fillStyle = "#333";
            ctx.font = "bold 10px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(cornerLabels[i], x, y + HANDLE_RADIUS + 12);
        }}
    }}

    function getMousePos(e) {{
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {{
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        }};
    }}

    function findCornerAtPos(pos) {{
        const scaledCorners = corners.map(c => [c[0] * displayScale, c[1] * displayScale]);
        for (let i = 0; i < scaledCorners.length; i++) {{
            const dx = pos.x - scaledCorners[i][0];
            const dy = pos.y - scaledCorners[i][1];
            if (Math.sqrt(dx*dx + dy*dy) < HANDLE_HIT_RADIUS) {{
                return i;
            }}
        }}
        return -1;
    }}

    function updateMagnifier(e, cornerIdx) {{
        if (cornerIdx < 0) {{
            magnifier.style.display = "none";
            cornerInfo.style.display = "none";
            return;
        }}

        const rect = canvas.getBoundingClientRect();
        const corner = corners[cornerIdx];
        const displayX = corner[0] * displayScale;
        const displayY = corner[1] * displayScale;

        // Position magnifier near cursor but offset so it doesn't cover the corner
        const magX = e.clientX - rect.left + 30;
        const magY = e.clientY - rect.top - MAG_SIZE - 10;

        magnifier.style.display = "block";
        magnifier.style.left = magX + "px";
        magnifier.style.top = Math.max(10, magY) + "px";

        // Draw magnified view
        magCtx.clearRect(0, 0, MAG_SIZE, MAG_SIZE);

        // Calculate source area in original image coordinates
        const sourceSize = MAG_SIZE / MAG_ZOOM / displayScale;
        const sourceX = corner[0] - sourceSize / 2;
        const sourceY = corner[1] - sourceSize / 2;

        // Draw magnified image portion
        magCtx.drawImage(
            image,
            sourceX, sourceY, sourceSize, sourceSize,
            0, 0, MAG_SIZE, MAG_SIZE
        );

        // Draw crosshair in center
        magCtx.strokeStyle = "#FF5722";
        magCtx.lineWidth = 2;
        const center = MAG_SIZE / 2;
        magCtx.beginPath();
        magCtx.moveTo(center - 15, center);
        magCtx.lineTo(center + 15, center);
        magCtx.moveTo(center, center - 15);
        magCtx.lineTo(center, center + 15);
        magCtx.stroke();

        // Draw circle around center point
        magCtx.beginPath();
        magCtx.arc(center, center, 8, 0, Math.PI * 2);
        magCtx.stroke();

        // Show corner coordinates
        cornerInfo.style.display = "block";
        cornerInfo.style.left = (magX) + "px";
        cornerInfo.style.top = (Math.max(10, magY) + MAG_SIZE + 5) + "px";
        cornerInfo.textContent = `Corner ${{cornerIdx + 1}}: (${{Math.round(corner[0])}}, ${{Math.round(corner[1])}})`;
    }}

    function sendCornersToStreamlit() {{
        const cornersStr = JSON.stringify(corners);
        if (cornersStr !== lastSentCorners) {{
            lastSentCorners = cornersStr;
            // Send to Streamlit via postMessage
            window.parent.postMessage({{
                type: "streamlit:setComponentValue",
                value: corners
            }}, "*");
        }}
    }}

    canvas.addEventListener("mousedown", function(e) {{
        const pos = getMousePos(e);
        const cornerIdx = findCornerAtPos(pos);

        if (cornerIdx >= 0) {{
            dragging = true;
            dragCornerIndex = cornerIdx;
            canvas.style.cursor = "grabbing";
            draw();
            updateMagnifier(e, cornerIdx);
        }}
    }});

    canvas.addEventListener("mousemove", function(e) {{
        const pos = getMousePos(e);

        if (dragging && dragCornerIndex >= 0) {{
            // Update corner position (convert back to original coordinates)
            corners[dragCornerIndex] = [
                Math.max(0, Math.min(originalWidth, pos.x / displayScale)),
                Math.max(0, Math.min(originalHeight, pos.y / displayScale))
            ];
            draw();
            updateMagnifier(e, dragCornerIndex);
        }} else {{
            const cornerIdx = findCornerAtPos(pos);
            canvas.style.cursor = cornerIdx >= 0 ? "grab" : "crosshair";

            // Show magnifier on hover over corners
            if (cornerIdx >= 0) {{
                updateMagnifier(e, cornerIdx);
            }} else {{
                magnifier.style.display = "none";
                cornerInfo.style.display = "none";
            }}
        }}
    }});

    canvas.addEventListener("mouseup", function(e) {{
        if (dragging) {{
            dragging = false;
            dragCornerIndex = -1;
            canvas.style.cursor = "crosshair";
            draw();
            sendCornersToStreamlit();
        }}
        magnifier.style.display = "none";
        cornerInfo.style.display = "none";
    }});

    canvas.addEventListener("mouseleave", function(e) {{
        if (dragging) {{
            // Keep dragging even if mouse leaves
        }} else {{
            magnifier.style.display = "none";
            cornerInfo.style.display = "none";
        }}
    }});

    // Handle mouse up outside canvas
    document.addEventListener("mouseup", function(e) {{
        if (dragging) {{
            dragging = false;
            dragCornerIndex = -1;
            canvas.style.cursor = "crosshair";
            draw();
            sendCornersToStreamlit();
            magnifier.style.display = "none";
            cornerInfo.style.display = "none";
        }}
    }});

    // Initial draw
    if (image.complete && image.naturalHeight !== 0) {{
        image.onload();
    }}

    // Send initial corners to Streamlit
    setTimeout(sendCornersToStreamlit, 100);
}})();
</script>
</body>
</html>
"""


def render_crop_editor(
    image_base64: str,
    initial_corners: List[List[float]],
    image_width: int,
    image_height: int,
    key: str = "crop_editor"
) -> Optional[List[List[float]]]:
    """
    Render the interactive crop editor component.

    Args:
        image_base64: Base64 encoded original image
        initial_corners: Initial corner points [[x,y], [x,y], [x,y], [x,y]]
                        in order: top-left, top-right, bottom-right, bottom-left
        image_width: Original image width
        image_height: Original image height
        key: Unique key for this component instance

    Returns:
        Updated corner coordinates if changed, else None
    """
    # Generate the HTML
    html_content = get_crop_editor_html(
        image_base64=image_base64,
        initial_corners=initial_corners,
        image_width=image_width,
        image_height=image_height,
        editor_id=key
    )

    # Calculate height based on aspect ratio
    max_width = 800
    display_scale = min(1, max_width / image_width)
    display_height = int(image_height * display_scale)

    # Add extra height for instructions and padding
    component_height = display_height + 100

    # Render the component
    component_value = components.html(
        html_content,
        height=component_height,
        scrolling=False
    )

    return component_value


def order_corners(corners: List[List[float]]) -> List[List[float]]:
    """
    Order corner points consistently: top-left, top-right, bottom-right, bottom-left.

    Args:
        corners: List of 4 corner points

    Returns:
        Ordered list of corner points
    """
    corners = np.array(corners, dtype=np.float32)

    # Sum of coordinates: smallest = top-left, largest = bottom-right
    s = corners.sum(axis=1)

    # Difference: smallest = top-right, largest = bottom-left
    d = np.diff(corners, axis=1).flatten()

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = corners[np.argmin(s)]  # Top-left
    ordered[1] = corners[np.argmin(d)]  # Top-right
    ordered[2] = corners[np.argmax(s)]  # Bottom-right
    ordered[3] = corners[np.argmax(d)]  # Bottom-left

    return ordered.tolist()


def corners_to_default(width: int, height: int, margin: float = 0.05) -> List[List[float]]:
    """
    Generate default corner positions with a margin from edges.

    Args:
        width: Image width
        height: Image height
        margin: Margin as fraction of dimensions

    Returns:
        Default corner positions
    """
    mx = width * margin
    my = height * margin

    return [
        [mx, my],                    # Top-left
        [width - mx, my],            # Top-right
        [width - mx, height - my],   # Bottom-right
        [mx, height - my]            # Bottom-left
    ]
