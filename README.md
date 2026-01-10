# Photo Storage App v2.1

A feature-rich photo storage application built with Python and Streamlit, featuring advanced edge detection, interactive crop refinement, facial recognition, and cloud storage support.

## What's New in v2.1

### Universal Photo Format Support
- **Any aspect ratio**: Now detects photo strips, panoramas, square photos, and non-standard formats
- **Rectangularity scoring**: Uses geometric angle analysis (90° corners) instead of preset ratios
- **Better photo strip detection**: Significantly improved detection for tall/narrow photo booth strips

### Advanced Edge Detection
- **Multi-method boundary detection**: Uses 6 different algorithms (Canny, adaptive threshold, color segmentation, saturation-based, Sobel+Laplacian, GrabCut) for robust photo detection
- **Smart scoring**: Automatically selects the best detected boundary based on geometric quality metrics
- **Works on varied backgrounds**: Handles wood, fabric, colored surfaces, and more

### Interactive Crop Refinement
- **Visual crop editor**: See detected boundaries overlaid on original image
- **Draggable corner handles**: Adjust crop area by dragging corner dots
- **Magnifier window**: 3x zoom follows your cursor for precise positioning
- **Batch processing**: Review and adjust multiple photos with Previous/Next navigation
- **Fine-tune controls**: Manual coordinate input for pixel-perfect adjustments

### Face Tags (Separate from User Tags)
- **Automatic face tagging**: Names assigned to faces are stored separately as `face_tags`
- **Independent search**: Filter by people (face recognition) or by user tags separately
- **Auto-sync**: Face tags update automatically when you name faces

## Features

### Photo Upload & Processing
- **Auto-crop**: Intelligent boundary detection for photos on any background
- **Manual refinement**: Interactive editor to adjust detected boundaries
- **Auto-orient**: EXIF-based and aspect ratio analysis orientation correction
- **Batch upload**: Process multiple photos at once with individual review

### Tagging System
- **User tags**: Add custom descriptive tags to any photo
- **Face tags**: Automatic tags from named faces (kept separate)
- **Dual search**: Filter by either tag type independently

### Favorites
- Star your favorite photos
- Quick access via Favorites tab

### Facial Recognition
- Automatic face detection in uploaded photos
- Name faces to create a person database
- Automatic recognition of known people in new photos
- Browse photos by person in the People tab

### Storage Options
- **Demo Mode**: Photos stored in browser session (no setup required)
- **AWS DynamoDB + S3**: Persistent cloud storage with S3 for images

## Installation

### Prerequisites
- Python 3.9+
- pip

### Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd photobooth-archive

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Face Recognition Setup (Optional)

Face recognition requires additional system dependencies:

**macOS:**
```bash
brew install cmake
pip install face-recognition
```

**Ubuntu/Debian:**
```bash
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install libx11-dev libgtk-3-dev
pip install face-recognition
```

**Windows:**
- Install Visual Studio Build Tools
- Install cmake
- `pip install face-recognition`

The app works without face recognition - you'll see a warning that the feature is unavailable.

## Usage Guide

### Uploading Photos

1. Go to the **Upload** tab
2. Drag & drop or select photos
3. Configure options:
   - **Auto-crop**: Detect photo boundaries automatically
   - **Review & adjust crops**: Enable interactive crop editor
   - **Auto-orient**: Fix rotated photos
   - **Detect faces**: Find faces in photos
4. Click **Process Photos**

### Crop Review (New in v2)

When "Review & adjust crops" is enabled:

1. **View detected boundary**: Green box shows detected photo edges
2. **Drag corners**: Click and drag the corner dots to adjust
3. **Use magnifier**: Hover over corners to see zoomed view
4. **Navigate batch**: Use Previous/Next buttons for multiple photos
5. **Quick actions**:
   - **Reset to Auto**: Re-detect boundary
   - **Full Image**: Use entire image (no crop)
   - **Skip**: Remove photo from batch
6. Click **Save All Photos** when done

### Managing Photos

- **Gallery**: View all photos in a grid
- **View details**: Click any photo for full view
- **Rotate**: Use orientation buttons
- **Tag**: Add user tags in detail view
- **People**: View face tags (auto-populated from named faces)
- **Favorite**: Click star to add to favorites

### Searching

- **By Person**: Filter by face tags (people identified via face recognition)
- **By Tag**: Filter by user-added tags
- **Text Search**: Type to filter either category

## File Structure

```
photobooth-archive/
├── app.py                      # Main Streamlit application
├── database.py                 # DynamoDB + S3 integration
├── image_processing.py         # Image processing utilities (auto-crop, orientation)
├── face_recognition_module.py  # Face detection and recognition
├── crop_editor.py              # Interactive crop editor component
├── scanner/                    # Photo boundary detection module
│   ├── __init__.py             # Exports EdgeDetector, PerspectiveTransformer
│   ├── detector.py             # Multi-method boundary detection (6 algorithms)
│   └── transformer.py          # Perspective correction transformation
├── requirements.txt            # Python dependencies
├── .streamlit/                 # Streamlit configuration
├── raw/                        # Sample raw photos (for testing)
├── processed/                  # Sample processed photos (for testing)
└── README.md                   # This file
```

## How Auto-Crop Works

The edge detection uses multiple strategies to find photo boundaries:

### Detection Methods
1. **Multi-threshold Canny**: Edge detection at various sensitivity levels (20-200)
2. **Adaptive Threshold**: Local threshold adaptation for uneven lighting
3. **Color Segmentation**: LAB color space background differentiation
4. **Saturation Detection**: Photos typically more colorful than backgrounds
5. **Combined Edges**: Sobel + Laplacian edge combination
6. **GrabCut**: Foreground/background segmentation

### Scoring Algorithm
Each method produces candidate boundaries, which are scored on:
- **Area ratio** (30%): Prefer contours that are 15-75% of image area
- **Rectangularity** (30%): How close corners are to 90° angles
- **Convexity** (25%): Prefer convex shapes
- **Edge proximity** (15%): Penalize shapes touching image borders

**Note**: Unlike v2.0, there are no preset aspect ratios. The detector now works with any photo format - standard prints, photo strips, panoramas, squares, or any custom size.

The highest-scoring boundary is selected and can be manually refined in the crop editor.

## AWS Setup

### Environment Variables

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
export DYNAMODB_TABLE=PhotoStorageApp
export S3_BUCKET=your-photo-bucket
```

### Required IAM Permissions

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "dynamodb:PutItem",
                "dynamodb:GetItem",
                "dynamodb:Scan",
                "dynamodb:UpdateItem",
                "dynamodb:DeleteItem",
                "dynamodb:CreateTable",
                "dynamodb:DescribeTable"
            ],
            "Resource": "arn:aws:dynamodb:*:*:table/PhotoStorageApp"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "s3:CreateBucket"
            ],
            "Resource": [
                "arn:aws:s3:::your-photo-bucket",
                "arn:aws:s3:::your-photo-bucket/*"
            ]
        }
    ]
}
```

## Streamlit Cloud Deployment

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Add secrets in Streamlit Cloud dashboard:

```toml
[aws]
access_key_id = "your_key"
secret_access_key = "your_secret"
region = "us-east-1"
table_name = "PhotoStorageApp"
s3_bucket = "your-bucket"
```

## Troubleshooting

### "Face recognition not available"
- Install the `face_recognition` package and its dependencies
- On some systems, you may need to install dlib first

### Photos not cropping correctly
- Enable "Review & adjust crops" to manually refine boundaries
- Try different backgrounds with more contrast
- Use the magnifier for precise corner placement

### DynamoDB connection issues
- Verify your AWS credentials are correct
- Check your IAM permissions include both DynamoDB and S3
- Ensure the region is correct

### App running slowly
- Face recognition can be slow on large images
- Large batches may take time to process
- Consider reducing image size before upload

## License

MIT License - feel free to modify and use as needed!

## Changelog

### v2.1
- **Removed aspect ratio constraints**: Now detects any photo format (strips, panoramas, etc.)
- **Added rectangularity scoring**: Measures corner angles instead of matching preset ratios
- **Improved photo strip detection**: Better handling of tall/narrow formats
- **Updated scoring weights**: Rebalanced for geometric quality over format matching
- **Refined contour shrinking**: Tighter crops with 2% margin adjustment

### v2.0
- Added multi-method edge detection (6 algorithms)
- Added interactive crop editor with draggable corners
- Added magnifier for precise crop adjustment
- Added batch upload with review workflow
- Added face_tags separate from user tags
- Added dual search (by person / by tag)
- Integrated scanner module directly into app
- Improved boundary scoring algorithm

### v1.0
- Initial release with basic auto-crop
- Face recognition
- DynamoDB storage
- Tagging and favorites
