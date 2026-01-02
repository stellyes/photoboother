# Photo Storage App 📸

A feature-rich photo storage application built with Python and Streamlit, with DynamoDB backend support and facial recognition capabilities.

## Features

### 📤 Photo Upload & Processing
- **Auto-crop**: Automatically detects and crops photos from high-contrast backgrounds (perfect for scanned photo strips, photos on desks, etc.)
- **Auto-orient**: Uses EXIF data and image analysis to properly orient photos
- **Manual orientation**: Rotate photos 90°, 180°, or 270° after upload

### 🏷️ Tagging System
- Add custom tags to any photo
- Search photos by tag
- Filter gallery by tag

### ⭐ Favorites
- Star your favorite photos
- Quick access via Favorites tab

### 👤 Facial Recognition
- Automatic face detection in uploaded photos
- Name faces to create a person database
- Automatic recognition of known people in new photos
- Browse photos by person

### 💾 Storage Options
- **Demo Mode**: Photos stored in browser session (no setup required)
- **AWS DynamoDB**: Persistent cloud storage with photos stored as base64

## Installation

### Prerequisites
- Python 3.9+
- pip

### Basic Setup

```bash
# Clone or download the app
cd photo_app

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

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

The app works without face recognition - you'll just see a warning that the feature is unavailable.

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Guide

### Storage Setup
1. On first launch, choose your storage type:
   - **Local (Demo Mode)**: No setup needed, photos persist only during session
   - **AWS DynamoDB**: Enter your AWS credentials and table name

### Uploading Photos
1. Go to the **Upload** tab
2. Drag & drop or select photos
3. Configure options:
   - **Auto-crop**: Enable for photos against backgrounds (like the photo strip example)
   - **Auto-orient**: Enable to fix rotated photos
   - **Detect faces**: Enable to find faces in photos
4. Click "Process & Upload"

### Managing Photos
- **Gallery**: View all photos in a grid
- **View details**: Click any photo to see full size
- **Rotate**: Use orientation buttons to rotate
- **Tag**: Add/remove tags in detail view
- **Favorite**: Click ☆ to add to favorites

### Facial Recognition
1. Upload a photo with faces
2. Click on a photo to view details
3. Named faces appear in the Faces section
4. Enter a name for each face and click Save
5. That person will be auto-recognized in future uploads

### Searching
- **By Tag**: Select from dropdown
- **By Person**: Select from dropdown
- **Text Search**: Type to filter tags

## AWS DynamoDB Setup

### Creating the Table
The app automatically creates the table if it doesn't exist, but you can also create it manually:

```bash
aws dynamodb create-table \
    --table-name PhotoStorageApp \
    --attribute-definitions AttributeName=photo_id,AttributeType=S \
    --key-schema AttributeName=photo_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST
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
        }
    ]
}
```

### Environment Variables (Alternative to UI)
Instead of entering credentials in the UI, you can set environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

## File Structure

```
photo_app/
├── app.py                    # Main Streamlit application
├── database.py               # DynamoDB integration
├── image_processing.py       # Auto-crop, orientation, thumbnails
├── face_recognition_module.py # Face detection and recognition
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## How Auto-Crop Works

The auto-crop feature uses computer vision to detect photos against contrasting backgrounds:

1. Converts image to grayscale
2. Applies edge detection (Canny)
3. Finds contours in the image
4. Looks for quadrilateral shapes (4-sided)
5. Applies perspective transform to straighten
6. Returns the cropped, straightened photo

This works best with:
- Photos on solid-color backgrounds
- High contrast between photo and background
- Reasonably flat photos (not overly curved)

## Troubleshooting

### "Face recognition not available"
- Install the `face_recognition` package and its dependencies
- On some systems, you may need to install dlib first

### Photos not cropping correctly
- Ensure there's good contrast between photo and background
- Try with a darker or lighter background
- Manual cropping can be done in an image editor before upload

### DynamoDB connection issues
- Verify your AWS credentials are correct
- Check your IAM permissions
- Ensure the region is correct
- Try creating the table manually first

### App running slowly
- Face recognition can be slow on large images
- Thumbnails are created to speed up gallery view
- Consider uploading smaller images

## License

MIT License - feel free to modify and use as needed!
