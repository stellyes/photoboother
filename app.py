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
from typing import List, Dict, Any, Optional
from PIL import Image
import io

from database import PhotoDatabase
from image_processing import (
    process_uploaded_image,
    base64_to_pil,
    pil_to_base64,
    rotate_image
)
from face_recognition_module import (
    detect_faces,
    recognize_faces,
    extract_face_thumbnail,
    is_available as face_recognition_available
)


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


def connect_database():
    """Connect to DynamoDB or use local storage."""
    with st.sidebar.expander("⚙️ Database Settings", expanded=not st.session_state.db_connected):
        storage_type = st.radio(
            "Storage Type",
            ["Local (Demo Mode)", "AWS DynamoDB"],
            index=0 if st.session_state.use_local_storage else 1
        )
        
        if storage_type == "AWS DynamoDB":
            st.session_state.use_local_storage = False
            
            aws_region = st.text_input("AWS Region", value="us-east-1")
            aws_access_key = st.text_input("AWS Access Key ID", type="password")
            aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
            table_name = st.text_input("Table Name", value="PhotoStorageApp")
            
            if st.button("Connect to DynamoDB"):
                try:
                    db = PhotoDatabase(
                        table_name=table_name,
                        region_name=aws_region,
                        aws_access_key_id=aws_access_key if aws_access_key else None,
                        aws_secret_access_key=aws_secret_key if aws_secret_key else None
                    )
                    db.create_table_if_not_exists()
                    st.session_state.db = db
                    st.session_state.db_connected = True
                    st.session_state.photos = db.get_all_photos()
                    st.success("✅ Connected to DynamoDB!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
        else:
            st.session_state.use_local_storage = True
            st.info("📁 Using local demo mode (photos stored in session)")
            
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


def upload_section():
    """Photo upload section."""
    st.subheader("📤 Upload Photos")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose photos to upload",
            type=['jpg', 'jpeg', 'png', 'gif', 'webp'],
            accept_multiple_files=True
        )
    
    with col2:
        auto_crop = st.checkbox("Auto-crop from background", value=True)
        auto_orient = st.checkbox("Auto-orient", value=True)
        detect_faces_opt = st.checkbox(
            "Detect faces",
            value=face_recognition_available(),
            disabled=not face_recognition_available()
        )
        
        if not face_recognition_available():
            st.caption("⚠️ Face recognition not available")
    
    if uploaded_files:
        if st.button("📥 Process & Upload", type="primary"):
            progress = st.progress(0)
            status = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status.text(f"Processing {uploaded_file.name}...")
                
                try:
                    # Read file
                    image_bytes = uploaded_file.read()
                    
                    # Process image
                    processed_b64, thumb_b64, original_b64, orientation = process_uploaded_image(
                        image_bytes,
                        auto_crop=auto_crop,
                        auto_orient=auto_orient
                    )
                    
                    # Detect faces if enabled
                    faces = []
                    if detect_faces_opt and face_recognition_available():
                        # Get known faces for recognition
                        known_faces = {}
                        if st.session_state.db and not st.session_state.use_local_storage:
                            known_faces = st.session_state.db.get_all_known_faces()
                        else:
                            # Build from local photos
                            for photo in st.session_state.photos:
                                for face in photo.get('faces', []):
                                    name = face.get('name')
                                    encoding = face.get('encoding')
                                    if name and encoding:
                                        if name not in known_faces:
                                            known_faces[name] = []
                                        known_faces[name].append(encoding)
                        
                        if known_faces:
                            faces = recognize_faces(processed_b64, known_faces)
                        else:
                            faces = detect_faces(processed_b64)
                    
                    # Create photo record
                    photo_id = str(uuid.uuid4())
                    timestamp = datetime.utcnow().isoformat()
                    
                    photo_data = {
                        'photo_id': photo_id,
                        'image_base64': processed_b64,
                        'thumbnail_base64': thumb_b64,
                        'original_base64': original_b64,
                        'filename': uploaded_file.name,
                        'orientation': orientation,
                        'tags': [],
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
                            image_base64=processed_b64,
                            filename=uploaded_file.name,
                            orientation=orientation,
                            tags=[],
                            is_favorite=False,
                            faces=faces,
                            thumbnail_base64=thumb_b64,
                            original_base64=original_b64
                        )
                        st.session_state.photos = st.session_state.db.get_all_photos()
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                progress.progress((i + 1) / len(uploaded_files))
            
            status.text("✅ Upload complete!")
            st.rerun()


def gallery_view(photos: List[Dict[str, Any]]):
    """Display photos in a gallery grid."""
    if not photos:
        st.info("No photos to display. Upload some photos to get started!")
        return
    
    # Number of columns
    cols_per_row = st.slider("Columns", 2, 6, 4, key="gallery_cols")
    
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
                    
                    # Tags
                    tags = photo.get('tags', [])
                    if tags:
                        st.caption(" ".join([f"`{t}`" for t in tags[:3]]))
                    
                    # View button
                    if st.button("View", key=f"view_{photo['photo_id']}"):
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
        
        # Tags section
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
                    # Get known faces
                    known_faces = get_known_faces()
                    
                    if known_faces:
                        new_faces = recognize_faces(photo['image_base64'], known_faces)
                    else:
                        new_faces = detect_faces(photo['image_base64'])
                    
                    if new_faces:
                        update_photo_field(photo['photo_id'], 'faces', new_faces)
                        st.success(f"Detected {len(new_faces)} face(s)!")
                    else:
                        st.info("No faces detected in this photo.")
                else:
                    st.warning("Face recognition not available.")
        
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


def get_known_faces() -> Dict[str, List[List[float]]]:
    """Get all known faces from storage."""
    known_faces = {}
    
    if st.session_state.use_local_storage:
        for photo in st.session_state.photos:
            for face in photo.get('faces', []):
                name = face.get('name')
                encoding = face.get('encoding')
                if name and encoding:
                    if name not in known_faces:
                        known_faces[name] = []
                    known_faces[name].append(encoding)
    elif st.session_state.db:
        known_faces = st.session_state.db.get_all_known_faces()
    
    return known_faces


def update_photo_field(photo_id: str, field: str, value: Any):
    """Update a single field of a photo."""
    if st.session_state.use_local_storage:
        for photo in st.session_state.local_photos:
            if photo['photo_id'] == photo_id:
                photo[field] = value
                photo['updated_at'] = datetime.utcnow().isoformat()
                break
        st.session_state.photos = st.session_state.local_photos
    elif st.session_state.db:
        kwargs = {field: value}
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
    gallery_view(favorites)


def search_view():
    """Search and filter photos."""
    st.subheader("🔍 Search Photos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Search by tag
        all_tags = get_all_tags()
        selected_tag = st.selectbox(
            "Filter by tag",
            ["All"] + all_tags
        )
    
    with col2:
        # Search by person
        all_people = get_all_people()
        selected_person = st.selectbox(
            "Filter by person",
            ["All"] + all_people
        )
    
    with col3:
        # Text search
        search_text = st.text_input("Search tags", "")
    
    # Apply filters
    filtered_photos = st.session_state.photos.copy()
    
    if selected_tag != "All":
        filtered_photos = [
            p for p in filtered_photos 
            if selected_tag in p.get('tags', [])
        ]
    
    if selected_person != "All":
        filtered_photos = [
            p for p in filtered_photos
            if any(f.get('name') == selected_person for f in p.get('faces', []))
        ]
    
    if search_text:
        search_lower = search_text.lower()
        filtered_photos = [
            p for p in filtered_photos
            if any(search_lower in t.lower() for t in p.get('tags', []))
        ]
    
    st.write(f"Showing {len(filtered_photos)} of {len(st.session_state.photos)} photos")
    
    gallery_view(filtered_photos)


def get_all_tags() -> List[str]:
    """Get all unique tags."""
    all_tags = set()
    for photo in st.session_state.photos:
        all_tags.update(photo.get('tags', []))
    return sorted(list(all_tags))


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
        gallery_view(st.session_state.photos)
    
    with tab3:
        favorites_view()
    
    with tab4:
        search_view()
    
    with tab5:
        people_view()


if __name__ == "__main__":
    main()
