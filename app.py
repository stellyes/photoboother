"""
Photo Storage App - Main Streamlit Application

A photo storage application with:
- Auto-cropping and orientation correction
- Tagging and search
- Favorites
- Facial recognition
"""

import streamlit as st
import uuid
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import io
import numpy as np
import cv2

from database import PhotoDatabase
from image_processing import (
    process_uploaded_image,
    base64_to_pil,
    pil_to_base64,
    rotate_image,
    base64_to_cv2,
    cv2_to_base64,
    bytes_to_base64,
    four_point_transform,
    order_points,
    create_thumbnail,
    fix_orientation_from_exif
)
from face_recognition_module import (
    detect_faces,
    extract_face_thumbnail,
    is_available as face_recognition_available
)
from crop_editor import render_crop_editor, order_corners, corners_to_default
from scanner import EdgeDetector, PerspectiveTransformer

EDGE_DETECTOR_AVAILABLE = True


# Page configuration
st.set_page_config(
    page_title="Photo Storage App",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .photo-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        background: #fafafa;
    }
    .favorite-star {
        color: gold;
        font-size: 24px;
    }
    .tag-badge {
        background: #e3f2fd;
        padding: 2px 8px;
        border-radius: 12px;
        margin: 2px;
        display: inline-block;
        font-size: 12px;
    }
    .face-box {
        border: 2px solid #4CAF50;
        padding: 5px;
        margin: 5px;
        border-radius: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'db' not in st.session_state:
        st.session_state.db = None
    if 'photos' not in st.session_state:
        st.session_state.photos = []
    if 'selected_photo' not in st.session_state:
        st.session_state.selected_photo = None
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'gallery'
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ''
    if 'filter_tag' not in st.session_state:
        st.session_state.filter_tag = None
    if 'filter_person' not in st.session_state:
        st.session_state.filter_person = None
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    if 'use_local_storage' not in st.session_state:
        st.session_state.use_local_storage = True
    # Staging state for crop review workflow
    if 'staged_photos' not in st.session_state:
        st.session_state.staged_photos = []  # Photos pending crop review
    if 'crop_review_index' not in st.session_state:
        st.session_state.crop_review_index = 0  # Current photo in review
    if 'crop_review_mode' not in st.session_state:
        st.session_state.crop_review_mode = False  # Whether in crop review mode


def has_aws_secrets() -> bool:
    """Check if AWS secrets are configured."""
    try:
        return (
            "aws" in st.secrets
            and st.secrets.aws.get("access_key_id")
            and st.secrets.aws.get("secret_access_key")
        )
    except Exception:
        return False


def get_aws_config() -> dict:
    """Get AWS configuration from secrets or environment."""
    import os

    if has_aws_secrets():
        return {
            'table_name': st.secrets.aws.get("table_name", "PhotoStorageApp"),
            'bucket_name': st.secrets.aws.get("s3_bucket", None),
            'region_name': st.secrets.aws.get("region", "us-west-1"),
            'aws_access_key_id': st.secrets.aws.access_key_id,
            'aws_secret_access_key': st.secrets.aws.secret_access_key
        }
    else:
        return {
            'table_name': os.environ.get('DYNAMODB_TABLE', 'PhotoStorageApp'),
            'bucket_name': os.environ.get('S3_BUCKET', None),
            'region_name': os.environ.get('AWS_REGION', 'us-west-1'),
            'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY')
        }


def connect_database():
    """Connect to DynamoDB + S3 or use local storage."""
    # Auto-connect using secrets/env vars if available and not yet connected
    config = get_aws_config()
    has_credentials = config['aws_access_key_id'] and config['aws_secret_access_key']

    if has_credentials and not st.session_state.db_connected:
        try:
            db = PhotoDatabase(
                table_name=config['table_name'],
                bucket_name=config['bucket_name'],
                region_name=config['region_name'],
                aws_access_key_id=config['aws_access_key_id'],
                aws_secret_access_key=config['aws_secret_access_key']
            )
            db.create_table_if_not_exists()
            st.session_state.db = db
            st.session_state.db_connected = True
            st.session_state.use_local_storage = False
            st.session_state.photos = db.get_all_photos()
        except Exception as e:
            st.sidebar.error(f"Auto-connect failed: {str(e)}")

    with st.sidebar.expander("⚙️ Database Settings", expanded=not st.session_state.db_connected):
        # Show connection status if using AWS
        if has_credentials and st.session_state.db_connected and not st.session_state.use_local_storage:
            st.success("Connected to AWS (DynamoDB + S3)")
            if st.session_state.db:
                st.caption(f"Table: {st.session_state.db.table_name}")
                st.caption(f"Bucket: {st.session_state.db.bucket_name}")
            if st.button("Disconnect"):
                st.session_state.db = None
                st.session_state.db_connected = False
                st.session_state.photos = []
                st.rerun()
            return

        storage_type = st.radio(
            "Storage Type",
            ["Local (Demo Mode)", "AWS (DynamoDB + S3)"],
            index=0 if st.session_state.use_local_storage else 1
        )

        if storage_type == "AWS (DynamoDB + S3)":
            st.session_state.use_local_storage = False

            # Use config as defaults
            aws_region = st.text_input("AWS Region", value=config['region_name'])
            aws_access_key = st.text_input("AWS Access Key ID", type="password")
            aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
            table_name = st.text_input("DynamoDB Table", value=config['table_name'])
            bucket_name = st.text_input("S3 Bucket (optional)", value=config['bucket_name'] or "")

            if st.button("Connect to AWS"):
                try:
                    db = PhotoDatabase(
                        table_name=table_name,
                        bucket_name=bucket_name if bucket_name else None,
                        region_name=aws_region,
                        aws_access_key_id=aws_access_key if aws_access_key else None,
                        aws_secret_access_key=aws_secret_key if aws_secret_key else None
                    )
                    db.create_table_if_not_exists()
                    st.session_state.db = db
                    st.session_state.db_connected = True
                    st.session_state.photos = db.get_all_photos()
                    st.success("Connected to AWS!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
        else:
            st.session_state.use_local_storage = True
            st.info("Using local demo mode (photos stored in session)")

            if not st.session_state.db_connected:
                st.session_state.db_connected = True
                if 'local_photos' not in st.session_state:
                    st.session_state.local_photos = []
                st.session_state.photos = st.session_state.local_photos


def get_display_image(photo: Dict[str, Any]) -> Image.Image:
    """Get the display image with correct orientation applied."""
    base64_data = photo.get('image_base64', '')
    orientation = photo.get('orientation', 0)
    
    image = base64_to_pil(base64_data)
    
    if orientation != 0:
        image = rotate_image(image, orientation)
    
    return image


def save_photo_local(photo_data: Dict[str, Any]):
    """Save photo to local storage."""
    if 'local_photos' not in st.session_state:
        st.session_state.local_photos = []
    
    # Check if updating existing
    existing_idx = None
    for i, p in enumerate(st.session_state.local_photos):
        if p['photo_id'] == photo_data['photo_id']:
            existing_idx = i
            break
    
    if existing_idx is not None:
        st.session_state.local_photos[existing_idx] = photo_data
    else:
        st.session_state.local_photos.append(photo_data)
    
    st.session_state.photos = st.session_state.local_photos


def delete_photo_local(photo_id: str):
    """Delete photo from local storage."""
    if 'local_photos' in st.session_state:
        st.session_state.local_photos = [
            p for p in st.session_state.local_photos 
            if p['photo_id'] != photo_id
        ]
        st.session_state.photos = st.session_state.local_photos


def detect_photo_boundary(image_bytes: bytes) -> Optional[np.ndarray]:
    """Detect photo boundary using EdgeDetector."""
    if not EDGE_DETECTOR_AVAILABLE:
        return None

    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return None

    detector = EdgeDetector(
        min_area_ratio=0.05,
        max_area_ratio=0.95,
        contour_epsilon=0.02
    )

    boundary = detector.detect(image)
    return boundary


def apply_crop_with_corners(
    original_base64: str,
    corners: List[List[float]]
) -> Tuple[str, str]:
    """Apply perspective transform crop using specified corners.

    Returns:
        Tuple of (cropped_base64, thumbnail_base64)
    """
    # Convert base64 to cv2 image
    image = base64_to_cv2(original_base64)

    # Apply perspective transform
    corners_array = np.array(corners, dtype=np.float32)
    transformer = PerspectiveTransformer()
    cropped = transformer.transform(image, corners_array)

    # Convert back to base64
    cropped_b64 = cv2_to_base64(cropped)

    # Create thumbnail
    pil_image = base64_to_pil(cropped_b64)
    pil_image.thumbnail((300, 300), Image.Resampling.LANCZOS)
    thumb_b64 = pil_to_base64(pil_image)

    return cropped_b64, thumb_b64


def upload_section():
    """Photo upload section with crop review workflow."""

    # Check if we're in crop review mode
    if st.session_state.crop_review_mode and st.session_state.staged_photos:
        crop_review_section()
        return

    st.subheader("📤 Upload Photos")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Choose photos to upload",
            type=['jpg', 'jpeg', 'png', 'gif', 'webp'],
            accept_multiple_files=True,
            key="photo_uploader"
        )

    with col2:
        auto_crop = st.checkbox("Auto-crop from background", value=True)
        manual_crop_review = st.checkbox(
            "Review & adjust crops manually",
            value=True,
            help="Review detected boundaries and adjust corners before saving"
        )
        auto_orient = st.checkbox("Auto-orient", value=True)
        detect_faces_opt = st.checkbox(
            "Detect faces",
            value=face_recognition_available(),
            disabled=not face_recognition_available()
        )

        if not face_recognition_available():
            st.caption("⚠️ Face recognition not available")

        if not EDGE_DETECTOR_AVAILABLE:
            st.caption("⚠️ Edge detection not available")
            auto_crop = False

    if uploaded_files:
        if st.button("📥 Process Photos", type="primary"):
            progress = st.progress(0)
            status = st.empty()

            staged_photos = []

            for i, uploaded_file in enumerate(uploaded_files):
                status.text(f"Processing {uploaded_file.name}...")

                try:
                    # Read file
                    image_bytes = uploaded_file.read()
                    original_b64 = bytes_to_base64(image_bytes)

                    # Get image dimensions
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    height, width = image.shape[:2]

                    # Detect boundary if auto-crop enabled
                    boundary = None
                    if auto_crop and EDGE_DETECTOR_AVAILABLE:
                        boundary = detect_photo_boundary(image_bytes)

                    # Convert boundary to corners list or use default
                    if boundary is not None:
                        corners = order_corners(boundary.tolist())
                    else:
                        corners = corners_to_default(width, height, margin=0.05)

                    # Stage photo for review
                    staged_photo = {
                        'filename': uploaded_file.name,
                        'original_base64': original_b64,
                        'width': width,
                        'height': height,
                        'corners': corners,
                        'auto_orient': auto_orient,
                        'detect_faces': detect_faces_opt,
                        'boundary_detected': boundary is not None
                    }

                    staged_photos.append(staged_photo)

                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")

                progress.progress((i + 1) / len(uploaded_files))

            if staged_photos:
                if manual_crop_review:
                    # Enter crop review mode
                    st.session_state.staged_photos = staged_photos
                    st.session_state.crop_review_index = 0
                    st.session_state.crop_review_mode = True
                    status.text("✅ Ready for crop review!")
                    st.rerun()
                else:
                    # Skip review, save directly
                    status.text("Saving photos...")
                    save_staged_photos(staged_photos)
                    status.text("✅ Upload complete!")
                    st.rerun()


def crop_review_section():
    """Interactive crop review section for staged photos."""
    staged = st.session_state.staged_photos
    idx = st.session_state.crop_review_index
    total = len(staged)

    if not staged or idx >= total:
        st.session_state.crop_review_mode = False
        st.rerun()
        return

    current = staged[idx]

    # Header with navigation
    st.subheader("✂️ Review & Adjust Crop")

    # Navigation and info
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if idx > 0:
            if st.button("← Previous", key="prev_photo"):
                st.session_state.crop_review_index -= 1
                st.rerun()

    with col2:
        st.markdown(f"**Photo {idx + 1} of {total}**: `{current['filename']}`")
        if current.get('boundary_detected'):
            st.caption("✅ Boundary auto-detected")
        else:
            st.caption("⚠️ No boundary detected - using default crop area")

    with col3:
        if idx < total - 1:
            if st.button("Next →", key="next_photo"):
                st.session_state.crop_review_index += 1
                st.rerun()

    st.divider()

    # Render the crop editor
    editor_key = f"crop_editor_{idx}_{current['filename']}"

    # Use session state to track corner updates for this photo
    corners_key = f"corners_{idx}"
    if corners_key not in st.session_state:
        st.session_state[corners_key] = current['corners']

    # Render interactive editor
    render_crop_editor(
        image_base64=current['original_base64'],
        initial_corners=st.session_state[corners_key],
        image_width=current['width'],
        image_height=current['height'],
        key=editor_key
    )

    # Corner adjustment inputs (alternative to dragging)
    with st.expander("Fine-tune corner coordinates"):
        corner_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        corners = st.session_state[corners_key]

        cols = st.columns(4)
        updated_corners = []

        for i, (label, col) in enumerate(zip(corner_labels, cols)):
            with col:
                st.caption(label)
                x = st.number_input(
                    "X", value=float(corners[i][0]),
                    min_value=0.0, max_value=float(current['width']),
                    key=f"corner_{idx}_{i}_x"
                )
                y = st.number_input(
                    "Y", value=float(corners[i][1]),
                    min_value=0.0, max_value=float(current['height']),
                    key=f"corner_{idx}_{i}_y"
                )
                updated_corners.append([x, y])

        if updated_corners != corners:
            st.session_state[corners_key] = updated_corners
            staged[idx]['corners'] = updated_corners

    st.divider()

    # Action buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔄 Reset to Auto", key="reset_crop"):
            if current.get('boundary_detected'):
                # Re-detect boundary
                image_bytes = base64.b64decode(current['original_base64'])
                boundary = detect_photo_boundary(image_bytes)
                if boundary is not None:
                    new_corners = order_corners(boundary.tolist())
                    st.session_state[corners_key] = new_corners
                    staged[idx]['corners'] = new_corners
                    st.rerun()
            else:
                new_corners = corners_to_default(
                    current['width'], current['height'], margin=0.05
                )
                st.session_state[corners_key] = new_corners
                staged[idx]['corners'] = new_corners
                st.rerun()

    with col2:
        if st.button("📐 Full Image", key="full_image"):
            new_corners = [
                [0, 0],
                [current['width'], 0],
                [current['width'], current['height']],
                [0, current['height']]
            ]
            st.session_state[corners_key] = new_corners
            staged[idx]['corners'] = new_corners
            st.rerun()

    with col3:
        if st.button("🗑️ Skip This Photo", key="skip_photo"):
            staged.pop(idx)
            if idx >= len(staged):
                st.session_state.crop_review_index = max(0, len(staged) - 1)
            if not staged:
                st.session_state.crop_review_mode = False
            st.rerun()

    with col4:
        save_label = "💾 Save All Photos" if total > 1 else "💾 Save Photo"
        if st.button(save_label, type="primary", key="save_all"):
            with st.spinner("Saving photos..."):
                # Update corners from session state before saving
                for i, photo in enumerate(staged):
                    ck = f"corners_{i}"
                    if ck in st.session_state:
                        photo['corners'] = st.session_state[ck]

                save_staged_photos(staged)

                # Clear staging state
                st.session_state.staged_photos = []
                st.session_state.crop_review_index = 0
                st.session_state.crop_review_mode = False

                # Clear corner session states
                for i in range(total):
                    ck = f"corners_{i}"
                    if ck in st.session_state:
                        del st.session_state[ck]

            st.success(f"✅ Saved {total} photo(s)!")
            st.rerun()

    # Cancel button
    st.divider()
    if st.button("❌ Cancel Upload", key="cancel_upload"):
        st.session_state.staged_photos = []
        st.session_state.crop_review_index = 0
        st.session_state.crop_review_mode = False
        st.rerun()


def save_staged_photos(staged_photos: List[Dict[str, Any]]):
    """Save all staged photos after crop review."""
    for staged in staged_photos:
        try:
            # Apply crop with adjusted corners
            cropped_b64, thumb_b64 = apply_crop_with_corners(
                staged['original_base64'],
                staged['corners']
            )

            # Apply orientation correction if enabled
            orientation = 0
            if staged.get('auto_orient'):
                pil_image = base64_to_pil(cropped_b64)
                pil_image = fix_orientation_from_exif(pil_image)

                # Simple aspect ratio based orientation
                width, height = pil_image.size
                if width / height > 1.5:
                    # Likely sideways, rotate
                    pil_image = pil_image.rotate(-90, expand=True)
                    orientation = 90

                cropped_b64 = pil_to_base64(pil_image)
                thumb_b64 = pil_to_base64(create_thumbnail(pil_image))

            # Detect faces if enabled
            faces = []
            if staged.get('detect_faces') and face_recognition_available():
                faces = detect_faces(cropped_b64)

            # Extract face_tags from detected faces
            face_tags = sorted(list(set(
                f.get('name') for f in faces if f.get('name')
            )))

            # Create photo record
            photo_id = str(uuid.uuid4())
            timestamp = datetime.utcnow().isoformat()

            photo_data = {
                'photo_id': photo_id,
                'image_base64': cropped_b64,
                'thumbnail_base64': thumb_b64,
                'original_base64': staged['original_base64'],
                'filename': staged['filename'],
                'orientation': orientation,
                'tags': [],
                'face_tags': face_tags,
                'is_favorite': False,
                'faces': faces,
                'created_at': timestamp,
                'updated_at': timestamp
            }

            # Save to storage
            if st.session_state.use_local_storage:
                save_photo_local(photo_data)
            elif st.session_state.db:
                st.session_state.db.save_photo(
                    photo_id=photo_id,
                    image_base64=cropped_b64,
                    filename=staged['filename'],
                    orientation=orientation,
                    tags=[],
                    face_tags=face_tags,
                    is_favorite=False,
                    faces=faces,
                    thumbnail_base64=thumb_b64,
                    original_base64=staged['original_base64']
                )
                st.session_state.photos = st.session_state.db.get_all_photos()

        except Exception as e:
            st.error(f"Error saving {staged['filename']}: {str(e)}")


def gallery_view(photos: List[Dict[str, Any]], key_suffix: str = ""):
    """Display photos in a gallery grid."""
    if not photos:
        st.info("No photos to display. Upload some photos to get started!")
        return

    # Number of columns
    cols_per_row = st.slider("Columns", 2, 6, 4, key=f"gallery_cols_{key_suffix}")
    
    # Create grid
    for i in range(0, len(photos), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(photos):
                photo = photos[i + j]
                
                with col:
                    # Display thumbnail or full image
                    if 'thumbnail_base64' in photo and photo['thumbnail_base64']:
                        img = base64_to_pil(photo['thumbnail_base64'])
                    else:
                        img = get_display_image(photo)
                        img.thumbnail((300, 300))
                    
                    st.image(img, use_container_width=True)
                    
                    # Photo info
                    star = "⭐" if photo.get('is_favorite') else "☆"
                    face_count = len(photo.get('faces', []))
                    face_icon = f"👤{face_count}" if face_count else ""

                    st.caption(f"{star} {face_icon}")

                    # Face tags (people in photo)
                    face_tags = photo.get('face_tags', [])
                    if face_tags:
                        st.caption(" ".join([f"👤`{ft}`" for ft in face_tags[:2]]))

                    # User tags
                    tags = photo.get('tags', [])
                    if tags:
                        st.caption(" ".join([f"`{t}`" for t in tags[:3]]))
                    
                    # View button
                    if st.button("View", key=f"view_{photo['photo_id']}_{key_suffix}"):
                        st.session_state.selected_photo = photo['photo_id']
                        st.session_state.view_mode = 'detail'
                        st.rerun()


def photo_detail_view(photo: Dict[str, Any]):
    """Display detailed view of a single photo."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display image with current orientation
        img = get_display_image(photo)
        st.image(img, use_container_width=True)
        
        # Orientation controls
        st.write("**Orientation**")
        orient_cols = st.columns(5)
        
        with orient_cols[0]:
            if st.button("↶ -90°"):
                new_orient = (photo.get('orientation', 0) - 90) % 360
                update_photo_field(photo['photo_id'], 'orientation', new_orient)
        
        with orient_cols[1]:
            if st.button("↷ +90°"):
                new_orient = (photo.get('orientation', 0) + 90) % 360
                update_photo_field(photo['photo_id'], 'orientation', new_orient)
        
        with orient_cols[2]:
            if st.button("180°"):
                new_orient = (photo.get('orientation', 0) + 180) % 360
                update_photo_field(photo['photo_id'], 'orientation', new_orient)
        
        with orient_cols[3]:
            if st.button("Reset"):
                update_photo_field(photo['photo_id'], 'orientation', 0)
        
        with orient_cols[4]:
            current = photo.get('orientation', 0)
            st.caption(f"Current: {current}°")
    
    with col2:
        # Back button
        if st.button("← Back to Gallery"):
            st.session_state.selected_photo = None
            st.session_state.view_mode = 'gallery'
            st.rerun()
        
        st.divider()
        
        # Favorite toggle
        is_fav = photo.get('is_favorite', False)
        fav_label = "⭐ Remove from Favorites" if is_fav else "☆ Add to Favorites"
        if st.button(fav_label):
            update_photo_field(photo['photo_id'], 'is_favorite', not is_fav)
        
        st.divider()

        # Face Tags section (people in photo - from face recognition)
        st.write("**People in Photo**")
        current_face_tags = photo.get('face_tags', [])

        if current_face_tags:
            face_tag_cols = st.columns(3)
            for i, face_tag in enumerate(current_face_tags):
                with face_tag_cols[i % 3]:
                    st.markdown(f"👤 `{face_tag}`")
        else:
            st.caption("No people identified yet. Name faces below to add them here.")

        st.divider()

        # User Tags section
        st.write("**Tags**")
        current_tags = photo.get('tags', [])

        # Display current tags
        if current_tags:
            tag_cols = st.columns(3)
            for i, tag in enumerate(current_tags):
                with tag_cols[i % 3]:
                    if st.button(f"❌ {tag}", key=f"remove_tag_{tag}_{photo['photo_id']}"):
                        new_tags = [t for t in current_tags if t != tag]
                        update_photo_field(photo['photo_id'], 'tags', new_tags)
        else:
            st.caption("No tags yet")

        # Add new tag
        new_tag = st.text_input("Add tag", key=f"new_tag_{photo['photo_id']}")
        if st.button("Add Tag") and new_tag:
            if new_tag not in current_tags:
                new_tags = current_tags + [new_tag]
                update_photo_field(photo['photo_id'], 'tags', new_tags)

        st.divider()
        
        # Faces section
        st.write("**Faces**")
        faces = photo.get('faces', [])
        
        if faces:
            for face in faces:
                face_col1, face_col2 = st.columns([1, 2])
                
                with face_col1:
                    # Extract and display face thumbnail
                    face_thumb = extract_face_thumbnail(
                        photo['image_base64'],
                        face['location']
                    )
                    if face_thumb:
                        st.image(
                            base64_to_pil(face_thumb),
                            width=80
                        )
                
                with face_col2:
                    current_name = face.get('name', '')
                    new_name = st.text_input(
                        "Name",
                        value=current_name or '',
                        key=f"face_name_{photo['photo_id']}_{face['face_id']}"
                    )
                    
                    if new_name != current_name:
                        if st.button("Save", key=f"save_face_{photo['photo_id']}_{face['face_id']}"):
                            # Update face name
                            updated_faces = []
                            for f in faces:
                                if f['face_id'] == face['face_id']:
                                    f['name'] = new_name
                                updated_faces.append(f)
                            update_photo_field(photo['photo_id'], 'faces', updated_faces)
        else:
            st.caption("No faces detected")

            if st.button("🔍 Detect Faces Now"):
                if face_recognition_available():
                    new_faces = detect_faces(photo['image_base64'])

                    if new_faces:
                        update_photo_field(photo['photo_id'], 'faces', new_faces)
                        st.success(f"Detected {len(new_faces)} face(s)! Add names below.")
                    else:
                        st.info("No faces detected in this photo.")
                else:
                    st.warning("Face detection not available.")
        
        st.divider()
        
        # Delete button
        if st.button("🗑️ Delete Photo", type="secondary"):
            if st.session_state.use_local_storage:
                delete_photo_local(photo['photo_id'])
            elif st.session_state.db:
                st.session_state.db.delete_photo(photo['photo_id'])
                st.session_state.photos = st.session_state.db.get_all_photos()
            
            st.session_state.selected_photo = None
            st.session_state.view_mode = 'gallery'
            st.rerun()
        
        st.divider()
        
        # Photo info
        st.write("**Info**")
        st.caption(f"Filename: {photo.get('filename', 'Unknown')}")
        st.caption(f"Created: {photo.get('created_at', 'Unknown')[:10]}")


def update_photo_field(photo_id: str, field: str, value: Any):
    """Update a single field of a photo."""
    if st.session_state.use_local_storage:
        for photo in st.session_state.local_photos:
            if photo['photo_id'] == photo_id:
                photo[field] = value
                photo['updated_at'] = datetime.utcnow().isoformat()
                # Auto-sync face_tags when faces are updated
                if field == 'faces':
                    photo['face_tags'] = sorted(list(set(
                        f.get('name') for f in value if f.get('name')
                    )))
                break
        st.session_state.photos = st.session_state.local_photos
    elif st.session_state.db:
        kwargs = {field: value}
        # Note: database.update_photo auto-syncs face_tags when faces are updated
        st.session_state.db.update_photo(photo_id, **kwargs)
        st.session_state.photos = st.session_state.db.get_all_photos()

    st.rerun()


def favorites_view():
    """Display favorited photos."""
    if st.session_state.use_local_storage:
        favorites = [p for p in st.session_state.photos if p.get('is_favorite')]
    elif st.session_state.db:
        favorites = st.session_state.db.get_favorites()
    else:
        favorites = []
    
    if not favorites:
        st.info("No favorites yet! Click the ☆ on a photo to add it to favorites.")
        return
    
    st.subheader(f"⭐ Favorites ({len(favorites)})")
    gallery_view(favorites, key_suffix="favorites")


def search_view():
    """Search and filter photos."""
    st.subheader("🔍 Search Photos")

    col1, col2 = st.columns(2)

    with col1:
        # Search by face tag (people from face recognition)
        all_face_tags = get_all_face_tags()
        selected_face_tag = st.selectbox(
            "👤 Filter by person (face recognition)",
            ["All"] + all_face_tags
        )

    with col2:
        # Search by user tag
        all_tags = get_all_tags()
        selected_tag = st.selectbox(
            "🏷️ Filter by tag",
            ["All"] + all_tags
        )

    col3, col4 = st.columns(2)

    with col3:
        # Text search for face tags
        search_face_text = st.text_input("Search people", "")

    with col4:
        # Text search for tags
        search_tag_text = st.text_input("Search tags", "")

    # Apply filters
    filtered_photos = st.session_state.photos.copy()

    if selected_face_tag != "All":
        filtered_photos = [
            p for p in filtered_photos
            if selected_face_tag in p.get('face_tags', [])
        ]

    if selected_tag != "All":
        filtered_photos = [
            p for p in filtered_photos
            if selected_tag in p.get('tags', [])
        ]

    if search_face_text:
        search_lower = search_face_text.lower()
        filtered_photos = [
            p for p in filtered_photos
            if any(search_lower in ft.lower() for ft in p.get('face_tags', []))
        ]

    if search_tag_text:
        search_lower = search_tag_text.lower()
        filtered_photos = [
            p for p in filtered_photos
            if any(search_lower in t.lower() for t in p.get('tags', []))
        ]

    st.write(f"Showing {len(filtered_photos)} of {len(st.session_state.photos)} photos")

    gallery_view(filtered_photos, key_suffix="search")


def get_all_tags() -> List[str]:
    """Get all unique user tags."""
    all_tags = set()
    for photo in st.session_state.photos:
        all_tags.update(photo.get('tags', []))
    return sorted(list(all_tags))


def get_all_face_tags() -> List[str]:
    """Get all unique face tags (person names from face recognition)."""
    all_face_tags = set()
    for photo in st.session_state.photos:
        all_face_tags.update(photo.get('face_tags', []))
    return sorted(list(all_face_tags))


def get_all_people() -> List[str]:
    """Get all unique people names."""
    all_names = set()
    for photo in st.session_state.photos:
        for face in photo.get('faces', []):
            name = face.get('name')
            if name:
                all_names.add(name)
    return sorted(list(all_names))


def people_view():
    """View and manage known people."""
    st.subheader("👥 People")
    
    all_people = get_all_people()
    
    if not all_people:
        st.info("No named people yet. Add names to faces in your photos to see them here.")
        return
    
    # Display each person with their photos
    for person_name in all_people:
        with st.expander(f"👤 {person_name}"):
            person_photos = [
                p for p in st.session_state.photos
                if any(f.get('name') == person_name for f in p.get('faces', []))
            ]
            
            st.write(f"{len(person_photos)} photo(s)")
            
            # Show thumbnails
            cols = st.columns(min(len(person_photos), 5))
            for i, photo in enumerate(person_photos[:5]):
                with cols[i]:
                    if 'thumbnail_base64' in photo:
                        st.image(base64_to_pil(photo['thumbnail_base64']), width=100)
                    
                    if st.button("View", key=f"person_view_{person_name}_{photo['photo_id']}"):
                        st.session_state.selected_photo = photo['photo_id']
                        st.session_state.view_mode = 'detail'
                        st.rerun()


def main():
    """Main application."""
    init_session_state()
    
    # Sidebar
    st.sidebar.title("📸 Photo Storage")
    
    # Database connection
    connect_database()
    
    if not st.session_state.db_connected:
        st.warning("Please connect to storage to continue.")
        return
    
    # Stats
    st.sidebar.divider()
    total_photos = len(st.session_state.photos)
    total_favorites = len([p for p in st.session_state.photos if p.get('is_favorite')])
    total_people = len(get_all_people())
    
    st.sidebar.metric("Total Photos", total_photos)
    st.sidebar.metric("Favorites", total_favorites)
    st.sidebar.metric("People", total_people)
    
    # Face recognition status
    st.sidebar.divider()
    if face_recognition_available():
        st.sidebar.success("✅ Face Recognition Available")
    else:
        st.sidebar.warning("⚠️ Face Recognition Unavailable")
        st.sidebar.caption("Install face_recognition package to enable")
    
    # Main content
    st.title("📸 Photo Storage App")
    
    # Check if viewing a specific photo
    if st.session_state.view_mode == 'detail' and st.session_state.selected_photo:
        photo = next(
            (p for p in st.session_state.photos 
             if p['photo_id'] == st.session_state.selected_photo),
            None
        )
        if photo:
            photo_detail_view(photo)
            return
    
    # Tab navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 Upload", 
        "🖼️ Gallery", 
        "⭐ Favorites", 
        "🔍 Search",
        "👥 People"
    ])
    
    with tab1:
        upload_section()
    
    with tab2:
        gallery_view(st.session_state.photos, key_suffix="gallery")
    
    with tab3:
        favorites_view()
    
    with tab4:
        search_view()
    
    with tab5:
        people_view()


if __name__ == "__main__":
    main()
