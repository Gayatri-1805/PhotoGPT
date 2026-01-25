# PhotoGPT - Quick Reference Card 📋

## One Query, Two Filters: Identity + Activity

```
┌─────────────────────────────────────────────────────────────┐
│  Query: "Gayatri sitting on a swing"                        │
└─────────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────────────────────┐
        │   Split into Two Components   │
        └───────────────────────────────┘
                ↓               ↓
        ┌────────────┐   ┌──────────────────┐
        │  "Gayatri" │   │ "sitting on swing"│
        │  (Identity)│   │   (Activity)      │
        └────────────┘   └──────────────────┘
                ↓               ↓
        ┌────────────┐   ┌──────────────────┐
        │Face Search │   │ Visual Analysis  │
        │  in Index  │   │  of Candidates   │
        └────────────┘   └──────────────────┘
                ↓               ↓
        ┌────────────┐   ┌──────────────────┐
        │ 150 photos │   │  Filter & Score  │
        │with Gayatri│   │  each for swing  │
        └────────────┘   └──────────────────┘
                        ↓
                ┌──────────────────┐
                │  Combine Scores  │
                │ 40% Face + 60% Act│
                └──────────────────┘
                        ↓
                ┌──────────────────┐
                │   12 Results     │
                │  Ranked Best→Worst│
                └──────────────────┘
```

## 3-Step Setup

### Step 1: Index Photos (One-Time)
```bash
python src/offline_indexing.py \
  --event-photos-dir data/event_photos \
  --output-dir data/embeddings \
  --mode face
```
**Output:** FAISS index with face embeddings from all photos

### Step 2: Register Identities
```
Web UI → "👤 Register New Person"
→ Upload Gayatri's photo
→ Enter name: "Gayatri"
→ Click "Register Me"
```
**Output:** Gayatri's face embedding saved to profiles

### Step 3: Query with Activity
```
Web UI → "🔍 Search by Name"
→ Select "🎬 Specific Activity"
→ Enter name: "Gayatri"
→ Enter activity: "sitting on a swing"
→ Click "🔍 Search"
```
**Output:** Photos of Gayatri on swings, ranked by score

## Scoring System

Each result has two scores:

| Score Type | Weight | What It Measures |
|------------|--------|------------------|
| **Face Similarity** | 40% | How well face matches Gayatri's reference photo |
| **Activity Similarity** | 60% | How well image matches "sitting on swing" description |
| **Combined Score** | 100% | Weighted average (final ranking) |

**Example Result:**
```
Photo: IMG_5234.jpg
├─ Face Score: 0.87 (87%)
├─ Activity Score: 0.72 (72%)
└─ Combined: 0.78 (78%) ← Ranked by this
```

## Query Examples

| Query Type | Example | Finds |
|------------|---------|-------|
| **Simple Action** | "John running" | John in running poses |
| **With Object** | "Sarah eating cake" | Sarah with food/cake |
| **With Location** | "Mike dancing at party" | Mike dancing (party context) |
| **Detailed** | "Gayatri sitting on swing" | Gayatri on/near swings |
| **Sports** | "David playing basketball" | David with ball/court |

## Common Activities That Work Well

✅ **Physical Actions**
- running, walking, jumping
- sitting, standing, lying
- swimming, dancing, playing

✅ **Social Activities**
- talking, laughing, hugging
- eating, drinking
- presenting, performing

✅ **With Props/Objects**
- "playing basketball" (ball visible)
- "reading book" (book visible)
- "eating food" (food visible)
- "using phone" (phone visible)

⚠️ **Harder to Detect**
- Emotions (thinking, feeling)
- Abstract states (waiting, planning)
- Subtle actions (whispering, glancing)

## Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Indexing Speed** | 1-2 sec/photo | One-time, offline |
| **Query Speed** | 2-5 seconds | For 100 candidate photos |
| **Accuracy** | 70-90% | For clear, visible activities |
| **Storage** | ~2KB/face | Embedding only |

## Thresholds

### Face Threshold (adjustable in sidebar)
- **0.6-0.8**: High precision, fewer matches
- **0.5** ⭐: Balanced (recommended)
- **0.3-0.4**: More matches, lower precision

### Activity Threshold (in code)
- Default: **0.25**
- Lower: More results
- Higher: Stricter activity matching

## Troubleshooting

### "No photos found containing [person]"
→ Lower face threshold in sidebar  
→ Re-register with clearer photo  
→ Check person is in indexed photos

### "Found person but no activity matches"
→ Try simpler activity ("sitting" vs "sitting on swing")  
→ Use synonyms ("running" vs "jogging")  
→ Activity may not be visually clear

### Too many/few results?
→ Adjust similarity threshold  
→ Make query more/less specific  
→ Check activity is visible in photos

## File Structure

```
PhotoGPT/
├── data/
│   ├── event_photos/          ← Your photos here
│   ├── embeddings/
│   │   ├── faiss.index        ← Face embeddings (Step 1)
│   │   ├── metadata.json      ← Image metadata
│   │   └── person_profiles.json ← Identities (Step 2)
│   └── selfies/               ← Reference photos
├── src/
│   ├── offline_indexing.py    ← Step 1 script
│   └── online_query.py        ← Step 3 logic
└── app.py                     ← Web interface
```

## Tech Stack

- **Face Detection**: MediaPipe
- **Visual Embeddings**: CLIP ViT-B/32
- **Text Encoding**: CLIP Text Encoder
- **Vector Search**: FAISS
- **Web UI**: Streamlit

## API Quick Reference

```python
from src.online_query import PhotoRetriever
from src.person_manager import PersonManager

# Load systems
retriever = PhotoRetriever(index_path, metadata_path)
person_mgr = PersonManager(profiles_path)

# Get person
profile = person_mgr.get_profile('Gayatri')

# Search with activity
results = retriever.find_person_doing_activity(
    person_embedding=profile['embedding'],
    activity_description='sitting on a swing',
    person_name='Gayatri',
    face_threshold=0.5,      # Face matching
    activity_threshold=0.25   # Activity matching
)

# Access results
for photo in results['matches']:
    print(photo['image_path'])
    print(f"Face: {photo['face_similarity']:.0%}")
    print(f"Activity: {photo['activity_similarity']:.0%}")
```

## Best Practices

### For Better Results
✅ Use high-quality reference photos (registration)  
✅ Describe visible, physical activities  
✅ Include objects/props in query ("swing", "ball")  
✅ Use simple, descriptive language  
✅ Index photos in 'face' mode

### To Avoid
❌ Vague activities ("doing something")  
❌ Abstract concepts ("being happy")  
❌ Activities without visual cues  
❌ Mixing too many concepts in one query

---

**Quick Start:** `streamlit run app.py` → Register person → Search with activity! 🚀
