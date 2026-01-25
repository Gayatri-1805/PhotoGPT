# PhotoGPT Complete Workflow 🎯

## Overview: Find People by Activity in 3 Simple Steps

This guide walks you through the complete workflow for finding photos using identity + activity queries like **"Gayatri sitting on a swing"**.

---

## 📊 The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: Index All Photos                     │
│  Process entire dataset → Generate visual embeddings            │
│  One-time setup: Run offline indexing                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                STEP 2: Register Identities                       │
│  Upload reference photos → Tag with names (e.g., "Gayatri")     │
│  Creates face embeddings for each person                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 3: Query & Retrieve                      │
│  Natural language query: "Gayatri sitting on a swing"           │
│  System matches: Identity (Gayatri) + Activity (sitting/swing)  │
│  Returns: Ranked photos matching both criteria                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Index All Photos (One-Time Setup)

### What It Does
- Processes every photo in your dataset
- Detects faces and extracts face embeddings (512-D vectors)
- Stores embeddings in FAISS index for fast search
- Creates metadata mapping embeddings to image files

### How to Run

```bash
python src/offline_indexing.py \
  --event-photos-dir data/event_photos \
  --output-dir data/embeddings \
  --mode face
```

**Parameters:**
- `--event-photos-dir`: Folder containing all your event photos
- `--output-dir`: Where to save the index files
- `--mode`: Use `face` for person + activity search

**Time:** ~1-2 seconds per photo (CPU), faster on GPU

**Output Files:**
- `data/embeddings/faiss.index` - Vector search index
- `data/embeddings/metadata.json` - Photo metadata

**Example:**
```bash
# Index 1000 photos from a wedding
python src/offline_indexing.py \
  --event-photos-dir data/event_photos/wedding_2024 \
  --output-dir data/embeddings \
  --mode face

# Output:
# ✓ Processed 1000 images
# ✓ Detected 3500 faces across 1000 images
# ✓ Built FAISS index with 3500 face embeddings
```

---

## Step 2: Register Identities

### What It Does
- User uploads a reference photo (selfie or portrait)
- System detects the face and creates a face embedding
- Identity is tagged with a name (e.g., "Gayatri", "John", "Sarah")
- Embedding is stored in person profiles database

### How to Use (Web UI)

1. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

2. **Navigate to "👤 Register New Person" tab**

3. **Upload Reference Photo:**
   - Click "Upload Your Selfie"
   - Choose a clear photo with one face
   - Best practices:
     - Front-facing
     - Good lighting
     - Neutral expression works best

4. **Enter Name:**
   - Type the person's name: "Gayatri"
   - Click "✅ Register Me"

5. **Confirmation:**
   - System confirms face detected
   - Profile saved to `data/embeddings/person_profiles.json`

**What Gets Stored:**
```json
{
  "Gayatri": {
    "name": "Gayatri",
    "selfie_path": "data/selfies/Gayatri_photo.jpg",
    "embedding": [0.123, 0.456, ...],  // 512-D vector
    "registration_date": "2026-01-25"
  }
}
```

**Repeat for All People:**
- Register everyone you want to search for
- Each person needs one reference photo
- Can register as many people as needed

---

## Step 3: Query with Identity + Activity

### What It Does
- Combines two searches into one intelligent query
- **Stage 1:** Find all photos containing the specified person
- **Stage 2:** Filter by activity description
- Returns photos ranked by combined relevance

### How to Use (Web UI)

1. **Navigate to "🔍 Search by Name" tab**

2. **Select Search Mode:**
   - Choose **"🎬 Specific Activity"**

3. **Enter Person Name:**
   - Type or select: "Gayatri"

4. **Describe the Activity:**
   - Use quick buttons: "running", "eating", "dancing", etc.
   - OR type custom: "sitting on a swing"

5. **Click "🔍 Search"**

### Behind the Scenes

```python
# System executes this workflow:

# 1. Retrieve Gayatri's face embedding
gayatri_embedding = person_profiles["Gayatri"]["embedding"]

# 2. Find all photos with Gayatri's face
candidate_photos = faiss_index.search(gayatri_embedding, threshold=0.5)
# Result: 150 photos containing Gayatri

# 3. Encode activity description
activity_embedding = clip_encode_text("sitting on a swing")

# 4. Score each candidate photo for activity match
for photo in candidate_photos:
    photo_embedding = clip_encode_image(photo)
    activity_score = similarity(photo_embedding, activity_embedding)
    combined_score = 0.4 * face_score + 0.6 * activity_score

# 5. Filter and rank
final_results = filter_and_sort(scored_photos, threshold=0.25)
# Result: 12 photos of Gayatri on a swing, sorted by relevance
```

### Scoring System

Each result has two scores:

**Face Similarity (40% weight):**
- How well the face matches Gayatri's reference photo
- Range: 0-1 (higher = better match)

**Activity Similarity (60% weight):**
- How well the full image matches "sitting on a swing"
- Range: 0-1 (higher = better match)

**Combined Score:**
- Weighted average of both scores
- Results sorted by combined score (best first)

### Example Queries

| Query | What It Finds |
|-------|---------------|
| `Gayatri sitting on a swing` | Gayatri in photos with swings |
| `John playing basketball` | John with basketball/court |
| `Sarah eating cake` | Sarah with food/cake |
| `Mike running` | Mike in running poses |
| `Emily dancing` | Emily in dance/party scenes |
| `David giving presentation` | David at podium/presenting |

---

## Advanced Usage

### Adjusting Thresholds

**In Sidebar:**
- **Similarity Threshold** (0.3 - 0.8)
  - Controls face matching strictness
  - Higher = fewer but more accurate matches
  - Recommended: 0.5

**In Code (activity threshold):**
```python
# src/online_query.py, line ~383
activity_threshold: float = 0.25  # Lower = more results
```

### Multi-Word Activity Descriptions

The system understands complex activities:

**Simple:** "sitting"
**Better:** "sitting on swing"  
**Best:** "sitting on a swing in garden"

More context = better results!

### Batch Queries

You can search for multiple people doing the same activity:

1. Search: "Gayatri dancing" → Save results
2. Search: "John dancing" → Save results
3. Compare who danced more!

---

## Troubleshooting

### "No photos found containing [person]"

**Causes:**
- Person not in indexed photos
- Face threshold too high
- Poor quality reference photo

**Solutions:**
1. Lower similarity threshold in sidebar
2. Re-register with a clearer reference photo
3. Verify person appears in event photos

### "Found person but none match activity"

**Causes:**
- Activity description doesn't match visuals
- Activity threshold too high
- Activity not clearly visible

**Solutions:**
1. Try simpler activity: "sitting on swing" → "sitting"
2. Try synonyms: "running" → "jogging" or "racing"
3. Lower activity threshold (requires code change)

### Getting Too Many/Few Results

**Too Many Results:**
- Increase similarity threshold (sidebar)
- Use more specific activity descriptions
- Increase activity threshold in code

**Too Few Results:**
- Decrease similarity threshold (sidebar)
- Simplify activity description
- Use broader terms

---

## Performance & Limitations

### Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Indexing | 1-2s/photo | One-time, run offline |
| Registration | <1s | One-time per person |
| Query (face) | <1s | Fast FAISS search |
| Query (face+activity) | 2-5s | Analyzes each candidate |

For 1000 photos with 100 candidates: ~2-3 seconds total

### What Works Well

✅ Clear, visible activities (running, eating, dancing)  
✅ Distinct poses and actions  
✅ Well-lit photos with clear faces  
✅ Activities with visual props (swing, ball, food)  

### Limitations

⚠️ Subtle activities harder to detect (thinking, waiting)  
⚠️ Requires face mode indexing (not full_image mode)  
⚠️ Activity must be visually apparent  
⚠️ Best with 5+ pixels face size  

---

## Complete Example Walkthrough

### Scenario: Find "Gayatri sitting on a swing" at a park event

**Step 1: Index Photos** (10 minutes for 500 photos)
```bash
python src/offline_indexing.py \
  --event-photos-dir data/event_photos/park_event \
  --output-dir data/embeddings \
  --mode face
```

**Step 2: Register Gayatri** (30 seconds)
1. Launch: `streamlit run app.py`
2. Go to "Register New Person"
3. Upload Gayatri's clear photo
4. Enter name: "Gayatri"
5. Click "Register Me"
6. ✅ Success!

**Step 3: Search** (5 seconds)
1. Go to "Search by Name"
2. Select "🎬 Specific Activity"
3. Enter "Gayatri"
4. Type activity: "sitting on a swing"
5. Click "🔍 Search"
6. Results: 8 photos of Gayatri on swings, ranked by relevance

**Step 4: Download**
- View each result with scores
- Download individual photos
- Or download all 8 as ZIP

**Total Time:** ~11 minutes (mostly one-time indexing)

---

## API Usage (Programmatic)

For developers who want to use this programmatically:

```python
from src.online_query import PhotoRetriever
from src.person_manager import PersonManager

# Initialize
retriever = PhotoRetriever('data/embeddings/faiss.index', 
                           'data/embeddings/metadata.json')
person_mgr = PersonManager('data/embeddings/person_profiles.json')

# Get person's embedding
gayatri = person_mgr.get_profile('Gayatri')

# Search for activity
results = retriever.find_person_doing_activity(
    person_embedding=gayatri['embedding'],
    activity_description='sitting on a swing',
    person_name='Gayatri',
    face_threshold=0.5,
    activity_threshold=0.25
)

# Process results
for photo in results['matches']:
    print(f"Photo: {photo['image_path']}")
    print(f"Face Score: {photo['face_similarity']:.2%}")
    print(f"Activity Score: {photo['activity_similarity']:.2%}")
    print(f"Combined: {photo['combined_score']:.2%}")
```

---

## Summary

This three-step workflow enables powerful identity + activity search:

1. **Index once** → Process all photos offline
2. **Register identities** → Tag reference photos with names
3. **Query naturally** → "Name + Activity" finds exactly what you want

The system intelligently combines face recognition with visual understanding to find photos matching both who you're looking for and what they're doing.

Perfect for:
- Event photography management
- Sports team photo organization
- Family photo albums
- Conference/wedding photo discovery

**Happy searching! 📸**
