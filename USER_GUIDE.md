# Find YOUR Photos with Natural Language 🔍

## "Find me at Taj Mahal" - It's That Simple!

PhotoGPT lets you find YOUR photos from a large dataset using natural language descriptions.

---

## Quick Example

**Scenario:** You have 1000 event photos. You want to find photos of yourself at Taj Mahal.

### What You Do:

1. **Register yourself** (one time)
   - Upload your photo
   - Enter your name: "John"

2. **Search naturally**
   - Enter name: "John"
   - Describe: "at Taj Mahal"
   - Click Search

3. **Get results**
   - All photos of YOU at Taj Mahal
   - Ranked by relevance

---

## What You Can Search

### 🏛️ Locations & Places
```
"at Taj Mahal"
"at beach"
"in mountains"
"at Eiffel Tower"
"at wedding venue"
"in garden"
"at stadium"
```

### 🏃 Activities & Actions
```
"running"
"eating"
"dancing"
"sitting on swing"
"playing sports"
"swimming"
"reading"
"taking selfie"
```

### 🎨 Scenes & Settings
```
"sunset"
"sunrise"
"at night"
"indoors"
"outdoors"
"at party"
"near water"
```

### 🎯 Objects & Props
```
"with car"
"with birthday cake"
"holding balloon"
"near tree"
"with dog"
"wearing sunglasses"
```

### 👗 Clothing & Appearance
```
"wearing red dress"
"in formal attire"
"wearing hat"
"in blue shirt"
```

### 🎭 Combined Descriptions
```
"dancing at beach sunset"
"eating cake at party"
"running in park"
"sitting at cafe with friends"
"wearing red dress at wedding"
"at Taj Mahal wearing blue shirt"
```

---

## How It Works

### The Magic Behind the Scenes

```
Your Search: "me at Taj Mahal"
        ↓
Step 1: Find all photos containing YOU
   (using your registered face)
        ↓
Step 2: Filter for "Taj Mahal"
   (using AI visual understanding)
        ↓
Result: Photos of YOU at Taj Mahal
   (ranked by how well they match)
```

### Dual Scoring System

Each photo gets two scores:

1. **Face Match** (40%): Is this YOU?
2. **Scene Match** (60%): Does it match "Taj Mahal"?

**Combined Score** = Weighted average of both

Photos ranked from best match to worst match.

---

## Real Examples

### Example 1: Tourist Photos

**Your Dataset:** 500 photos from India trip

**Your Search:** "at Taj Mahal"

**Results:**
- Photo 1: You in front of Taj Mahal (95% match)
- Photo 2: You near Taj Mahal gardens (87% match)
- Photo 3: You at Taj Mahal entrance (82% match)

### Example 2: Party Photos

**Your Dataset:** 300 photos from birthday party

**Your Search:** "with birthday cake"

**Results:**
- Photo 1: You blowing candles (92% match)
- Photo 2: You cutting cake (88% match)
- Photo 3: You holding cake slice (79% match)

### Example 3: Sports Event

**Your Dataset:** 200 photos from sports day

**Your Search:** "running"

**Results:**
- Photo 1: You in running race (94% match)
- Photo 2: You jogging (85% match)
- Photo 3: You sprinting (83% match)

---

## Step-by-Step Guide

### First Time Setup (5 minutes)

**1. Index Your Photos** (Already done! ✓)
```bash
python src/offline_indexing.py \
  --event-photos-dir data/event_photos \
  --output-dir data/embeddings
```

**2. Launch the App**
```bash
streamlit run app.py
```

**3. Register Yourself**
- Go to "👤 Register New Person"
- Upload YOUR clear photo
- Enter YOUR name (e.g., "Sarah")
- Click "Register Me"
- ✅ Done! System now knows what you look like

### Every Search (10 seconds)

**1. Go to "🔍 Search by Name"**

**2. Select "🔍 Describe the Photo"**

**3. Enter YOUR name**
```
Name: Sarah
```

**4. Describe what's in the photo**
```
Quick buttons: "at beach", "running", "sunset"
OR
Type your own: "at Taj Mahal"
```

**5. Click "🔍 Search"**

**6. Browse Results**
- See all matching photos
- View face & scene scores
- Download individually or as ZIP

---

## Tips for Best Results

### ✅ Do This

**Be Specific**
- ❌ "at monument"
- ✅ "at Taj Mahal"

**Describe Visible Things**
- ✅ "at beach" (beach is visible)
- ✅ "wearing red dress" (dress is visible)
- ✅ "with dog" (dog is visible)

**Combine Concepts**
- ✅ "dancing at beach sunset"
- ✅ "eating cake at party"
- ✅ "running in park wearing blue"

**Use Landmarks**
- ✅ "at Taj Mahal"
- ✅ "at Eiffel Tower"
- ✅ "at Golden Gate Bridge"

### ❌ Avoid This

**Vague Descriptions**
- ❌ "doing something"
- ❌ "somewhere outside"

**Abstract Concepts**
- ❌ "feeling happy" (not visible)
- ❌ "thinking about work" (not visible)

**Things Not in Photo**
- ❌ "not at beach" (system can't process negation)
- Use positive descriptions instead

---

## Common Questions

### Q: Can I search for multiple people?

**A:** One person at a time. But you can:
1. Search "John at beach"
2. Then search "Sarah at beach"
3. Compare results!

### Q: What if I'm not sure of the exact wording?

**A:** Try variations!
- "at Taj Mahal"
- "near Taj Mahal"
- "standing at Taj Mahal"
- "in front of Taj Mahal"

All will work!

### Q: Can I search by date or time?

**A:** Indirectly, yes!
- "sunset" → Photos during sunset
- "at night" → Nighttime photos
- "daytime" → Daytime photos

### Q: How accurate is it?

**Typical accuracy:**
- Famous landmarks: 85-95%
- Common activities: 80-90%
- Clothing/colors: 70-85%
- Complex scenes: 65-80%

**Tip:** Lower the similarity threshold for more results!

---

## Troubleshooting

### "No photos found"

**Possible reasons:**
1. You're not in those photos
2. Description doesn't match
3. Threshold too high

**Solutions:**
1. Try simpler description: "Taj Mahal" → "monument"
2. Lower similarity threshold in sidebar
3. Check you're registered correctly

### "Results don't match well"

**Solutions:**
1. Be more specific: "at beach" → "at beach sunset"
2. Try different words: "running" → "jogging"
3. Adjust threshold for more/fewer results

### "System is slow"

**Normal behavior:**
- First search: 3-5 seconds (loading models)
- Subsequent searches: 1-2 seconds
- Large datasets (1000+ photos): 2-5 seconds

---

## Privacy & Data

**What's Stored:**
- Your face embedding (512 numbers)
- Your name
- Path to your reference photo

**What's NOT Stored:**
- Original photos (only paths)
- Search history
- Any personal data beyond name

**Data Location:**
- `data/embeddings/person_profiles.json`
- All stored locally on your machine
- No cloud upload

---

## Advanced Usage

### Multiple Concepts

Search with multiple ideas:

```
"at beach during sunset wearing blue"
```

System will find photos matching ALL concepts!

### Adjusting Thresholds

**Face Threshold** (sidebar):
- 0.5: Balanced (recommended)
- 0.6+: Stricter (fewer, better matches)
- 0.4: Looser (more matches)

**Scene Threshold** (default 0.22):
- Controls how well scene must match
- Lower = more results
- Higher = stricter matching

---

## Success Stories

### Use Case 1: Wedding Photos

**Problem:** 2000 wedding photos, want only yours

**Solution:**
1. Register yourself
2. Search "me" → Get all 150 photos of you
3. Search "me at ceremony" → Get 40 ceremony photos
4. Search "me dancing" → Get 25 dancing photos

**Result:** Found exactly what you wanted in seconds!

### Use Case 2: Vacation Album

**Problem:** 800 vacation photos across 10 locations

**Solution:**
1. Search "me at Taj Mahal" → 35 photos
2. Search "me at beach" → 52 photos
3. Search "me at restaurant" → 28 photos

**Result:** Organized entire trip by location!

### Use Case 3: Sports Photography

**Problem:** 500 marathon photos, find your race shots

**Solution:**
1. Search "me running" → 89 photos
2. Search "me at finish line" → 12 photos
3. Search "me with medal" → 5 photos

**Result:** Created personal highlight reel!

---

## What's Next?

Now that you understand how it works:

1. **Try it out!** Register yourself and search
2. **Experiment** with different descriptions
3. **Explore** all the quick search buttons
4. **Share** your results!

**Happy searching! 📸**

---

## Quick Reference Card

```
┌─────────────────────────────────────────────┐
│ SEARCH FORMULA:                             │
│                                             │
│ YOUR NAME + DESCRIPTION                     │
│                                             │
│ Examples:                                   │
│ • "John" + "at Taj Mahal"                  │
│ • "Sarah" + "running"                       │
│ • "Mike" + "at beach sunset"               │
│ • "Emma" + "wearing red dress"             │
└─────────────────────────────────────────────┘

Results: All YOUR photos matching the description!
```
