# PhotoGPT - Intelligent Photo Retrieval System

PhotoGPT is an AI-powered photo search and retrieval system that helps you find photos of specific people from large event photo collections. It uses advanced face recognition and semantic search capabilities to match selfies with event photos.

## Features

- **Face Recognition Search**: Register a person's selfie and find all photos containing them from event photo collections
- **Semantic Photo Search**: Describe what you're looking for in natural language (activities, scenes, objects, locations)
- **Smart Matching**: Combines face detection with visual similarity scoring
- **Easy Registration**: Simple web interface to register new people with their selfies
- **Batch Processing**: Efficiently index large collections of event photos offline

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Git (optional, for cloning the repository)
- At least 4GB RAM recommended
- GPU optional but recommended for faster processing

### Step 1: Clone or Download the Repository

```bash
git clone <your-repository-url>
cd Photogpt
```

Or download and extract the ZIP file to a folder.

### Step 2: Create a Virtual Environment (Recommended)

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- Streamlit (web interface)
- PyTorch (deep learning)
- OpenCLIP (vision-language model)
- MediaPipe (face detection)
- FAISS (vector similarity search)
- Pillow (image processing)

### Step 4: Prepare Your Photo Folders

Create the required directory structure:

```
PhotoGPT/
├── data/
│   ├── event_photos/      # Put your event photos here
│   ├── selfies/           # Registered selfies (auto-created)
│   ├── temp_uploads/      # Temporary uploads (auto-created)
│   └── embeddings/        # Index files (auto-created)
```

Add your event photos to the `data/event_photos/` folder.

### Step 5: Build the Photo Index

Before searching, you need to index your event photos:

```bash
python src/offline_indexing.py --event-photos-dir data/event_photos --output-dir data/embeddings
```

**Options:**
- `--mode face` (default): Index faces only - enables person search
- `--mode full_image`: Index full images - enables semantic search
- `--batch-size 32`: Adjust batch size based on your GPU memory



The indexing process will:
1. Detect faces in all event photos
2. Generate embeddings for each face/image
3. Build a FAISS index for fast similarity search
4. Save metadata for photo retrieval

### Step 6: Launch the Web Interface

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

### 1. Register a New Person

1. Go to the **"Register New Person"** tab
2. Enter the person's name
3. Upload a clear selfie (JPG/PNG/HEIC)
4. Click "Register Person"

### 2. Search by Name

1. Go to the **"Search by Name"** tab
2. Enter the registered person's name
3. Adjust similarity threshold if needed (default: 0.65)
4. Click "Search"
5. View matching photos from event collections

### 3. Semantic Search

1. Go to the **"Semantic Search"** tab
2. Describe what you're looking for (e.g., "people dancing", "beach sunset", "group photo")
3. Adjust similarity threshold if needed
4. Click "Search"
5. Browse results

## Requirements

Main dependencies:
- Python 3.8+
- streamlit
- torch
- open-clip-torch
- mediapipe
- faiss-cpu (or faiss-gpu)
- pillow
- pillow-heif
- numpy


