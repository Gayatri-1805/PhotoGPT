# Quick Start Guide - PhotoGPT Semantic Search 🚀

## Get Started in 5 Minutes!

### Step 1: Install Dependencies (1 min)

```bash
cd Photogpt
pip install -r requirements.txt
```

### Step 2: Add Your Photos (1 min)

Place your event photos in the `data/event_photos/` folder:

```bash
# Create the directory if it doesn't exist
mkdir -p data/event_photos

# Copy your photos
cp /path/to/your/photos/*.jpg data/event_photos/
```

### Step 3: Build the Search Index (2-3 min)

For **semantic search** (describe photos in natural language):

```bash
python src/offline_indexing.py \
  --event-photos-dir data/event_photos \
  --output-dir data/embeddings \
  --mode full_image
```

This will process all photos and create a searchable index.

**Time estimate**: ~1-2 seconds per photo on CPU, faster on GPU

### Step 4: Launch the App (< 1 min)

```bash
streamlit run app.py
```

The app opens automatically at: `http://localhost:8501`

### Step 5: Search! 🔍

1. Go to the **"🎨 Semantic Search"** tab
2. Type what you're looking for:
   - "people dancing"
   - "sunset beach"
   - "group photo indoors"
3. Click **Search**
4. Download individual photos or all as ZIP!

## Example Searches to Try

**Activities:**
```
people running
someone eating
dancing
playing sports
```

**Scenes:**
```
sunset
beach scene
indoor party
outdoor event
mountain landscape
```

**Combinations:**
```
sunset beach group photo
people dancing at party
outdoor wedding ceremony
```

## Switching to Face Search Mode

Want to find specific people instead? Rebuild the index:

```bash
python src/offline_indexing.py \
  --event-photos-dir data/event_photos \
  --output-dir data/embeddings \
  --mode face
```

Then use the **"👤 Register New Person"** and **"🔍 Search by Name"** tabs.

## Adjusting Search Sensitivity

In the sidebar, adjust the **Similarity Threshold**:

- **Lower (0.20-0.25)**: More results, exploratory
- **Medium (0.25-0.30)**: Balanced ⭐ **Recommended**
- **Higher (0.30-0.35)**: Fewer, more precise results

## Troubleshooting

**No results?**
- Try lowering the threshold
- Simplify your query
- Make sure photos are in `data/event_photos/`

**App won't start?**
- Run: `pip install -r requirements.txt`
- Check Python version: `python --version` (need 3.8+)

**Indexing failed?**
- Verify photos are valid images (.jpg, .png)
- Check you have enough disk space
- See error messages for details

## What's Next?

📖 Read the [Semantic Search Guide](SEMANTIC_SEARCH_GUIDE.md) for advanced tips

📚 Check the [README](README.md) for full documentation

🎯 Try different queries and discover what works best for your photos!

---

**Happy searching! 📸**
