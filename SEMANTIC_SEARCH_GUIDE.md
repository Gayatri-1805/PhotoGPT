# Semantic Photo Search Guide 🎨

## What is Semantic Search?

Semantic Search allows you to find photos by describing what's in them using natural language. Unlike traditional keyword tagging, this uses AI to understand the **meaning** and **visual content** of images automatically.

## How It Works

PhotoGPT uses **CLIP (Contrastive Language-Image Pre-training)**, a neural network trained by OpenAI that understands both images and text in the same embedding space.

```
Your Text Query → CLIP Text Encoder → 512-D Vector
                                           ↓
                                    Similarity Match
                                           ↓
All Photos → CLIP Image Encoder → 512-D Vectors
```

The system finds photos whose visual embeddings are most similar to your text description's embedding.

## Setup Requirements

### 1. Build Index in Full Image Mode

```bash
python src/offline_indexing.py \
  --event-photos-dir data/event_photos \
  --output-dir data/embeddings \
  --mode full_image
```

**Why full_image mode?**
- Analyzes entire image composition
- Captures scenes, activities, and context
- Better for semantic understanding vs. just faces

### 2. Launch the App

```bash
streamlit run app.py
```

## Query Examples by Category

### 🏃 Activities & Actions

**What works well:**
- "people running"
- "someone eating"
- "dancing"
- "playing sports"
- "swimming"
- "reading a book"
- "taking a photo"
- "giving a speech"

**Tips:**
- Be specific about the action
- Mention "people" or "person" for clarity
- Plural vs singular matters

### 🌅 Scenes & Environments

**What works well:**
- "sunset"
- "beach scene"
- "indoor party"
- "outdoor event"
- "mountain landscape"
- "city skyline"
- "forest path"
- "restaurant interior"

**Tips:**
- Describe the setting or atmosphere
- Time of day can help: "sunset", "night", "daytime"
- Location type: "indoor", "outdoor", "urban", "nature"

### 🎭 Events & Occasions

**What works well:**
- "wedding ceremony"
- "birthday party"
- "graduation"
- "conference presentation"
- "concert performance"
- "sports game"
- "family gathering"

**Tips:**
- Event type + key activity
- Include setting if relevant

### 👥 Groups & People

**What works well:**
- "group photo"
- "crowd of people"
- "two people talking"
- "selfie"
- "people smiling"
- "formal portrait"
- "candid moment"

**Tips:**
- Number of people helps
- Specify pose/arrangement
- Emotional expressions

### 🎨 Visual Attributes

**What works well:**
- "colorful decorations"
- "formal attire"
- "casual clothing"
- "bright lighting"
- "dark background"
- "blurry motion"
- "close-up shot"

**Tips:**
- Visual properties and composition
- Lighting and color
- Photography style

### 🍕 Objects & Items

**What works well:**
- "food on table"
- "birthday cake"
- "musical instruments"
- "flowers"
- "sports equipment"
- "electronic devices"
- "vehicles"

**Tips:**
- Be specific about the object
- Mention context: "on table", "in hand"

## Advanced Query Techniques

### 1. Combining Concepts

**Simple:**
- "sunset"

**Better:**
- "sunset beach"

**Best:**
- "sunset beach group photo"

Combining multiple concepts creates more specific searches.

### 2. Adding Context

Instead of: "dancing"
Try: "people dancing at night party"

Instead of: "food"
Try: "people eating food outdoors"

### 3. Using Negation (doesn't work directly)

CLIP doesn't support negation well. Instead:
- ❌ "not raining"
- ✅ "sunny day"

### 4. Adjusting Threshold

In the sidebar, adjust the **Similarity Threshold**:

- **0.20-0.25**: Very broad results, exploratory
- **0.25-0.30**: Balanced for semantic search (recommended)
- **0.30-0.35**: More precise, fewer results
- **0.35+**: Very strict, may miss relevant photos

## What Works Best vs. Struggles

### ✅ Works Great

1. **Visible Activities**
   - Running, swimming, dancing
   - Eating, drinking, talking
   
2. **Clear Scenes**
   - Sunset, beach, mountains
   - Indoor vs outdoor
   - Day vs night

3. **Obvious Objects**
   - Food, vehicles, animals
   - Large, prominent items

4. **General Compositions**
   - Group photos vs solo
   - Close-up vs wide shot

### ⚠️ Challenging

1. **Subtle Emotions**
   - "sad person" (may struggle)
   - Better: "smiling" or "laughing"

2. **Abstract Concepts**
   - "love", "happiness", "celebration"
   - Better: describe visible elements

3. **Text in Images**
   - "sign that says..."
   - CLIP has limited OCR ability

4. **Specific People**
   - "photo of John"
   - Use face search for this!

5. **Fine-grained Details**
   - "red car" vs "blue car" (color works, but specific shades are hard)
   - "Nike shoes" (brand recognition limited)

## Optimization Tips

### 1. Start Broad, Then Narrow

```
Step 1: "outdoor event"           → 50 results
Step 2: "outdoor event sunset"    → 20 results
Step 3: "outdoor event sunset dancing" → 5 results
```

### 2. Use Multiple Phrasings

If "people running" doesn't work well, try:
- "runners"
- "jogging"
- "athletic activity"
- "exercise outdoors"

### 3. Think Visually

Ask yourself: "What would I **see** in this photo?"
- Not: "birthday party"
- Better: "birthday cake with candles people celebrating"

### 4. Leverage CLIP's Training

CLIP was trained on internet images with captions. Phrases that commonly appear together work better:
- "sunset beach" (common pairing)
- "group selfie" (common phrase)
- "wedding ceremony" (common concept)

## Troubleshooting

### "No Results Found"

1. **Lower the threshold** (try 0.20-0.25)
2. **Simplify your query** (start with one concept)
3. **Try synonyms** (e.g., "ocean" vs "sea", "photo" vs "picture")
4. **Verify photos are indexed** (check data/event_photos/)

### "Results Don't Match My Query"

1. **Add more context** ("running" → "people running race")
2. **Be more specific** ("party" → "indoor birthday party")
3. **Adjust threshold higher** (0.30-0.35)
4. **Try different wording** (CLIP interprets literally)

### "Only Getting Partial Matches"

This is normal! CLIP finds **visual similarity**, not exact matches. Photos with some elements of your query will appear.

**Solution:** Use higher threshold or combine more specific terms.

## Example Workflows

### Workflow 1: Finding Action Shots

```
Query: "playing basketball"
Results: 12 photos

Refine: "playing basketball outdoors"
Results: 7 photos

Best: "playing basketball outdoor court daytime"
Results: 4 highly relevant photos
```

### Workflow 2: Finding Scenic Moments

```
Query: "landscape"
Results: 30 photos

Refine: "mountain landscape"
Results: 15 photos

Best: "mountain landscape sunset sky"
Results: 5 stunning shots
```

### Workflow 3: Finding Group Photos

```
Query: "people"
Results: 100+ photos

Refine: "group of people"
Results: 40 photos

Best: "group photo formal setting"
Results: 10 perfect group shots
```

## Performance Expectations

### Typical Accuracy
- **High (80%+)**: Common activities, clear scenes
- **Medium (50-80%)**: Specific combinations, nuanced queries  
- **Variable (<50%)**: Abstract concepts, fine details

### Speed
- **Encoding**: ~50-100ms per query
- **Search**: <500ms for 10,000 photos
- **Total**: Usually <1 second

## Combining with Face Search

Use **both modes** for best results:

1. **Semantic Search**: Find photos with specific activities/scenes
2. **Face Search**: Find which of those photos contain specific people

Example:
1. Semantic: "sunset beach" → 20 photos
2. Face: Search for "John" within those 20
3. Result: Sunset beach photos with John

## Learn More

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [OpenCLIP GitHub](https://github.com/mlfoundations/open_clip)
- [Understanding CLIP](https://openai.com/blog/clip/)

---

**Remember**: Semantic search is exploratory. Experiment with different queries to discover what works best for your photo collection!
