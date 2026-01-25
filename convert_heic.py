from PIL import Image
import pillow_heif
import os

pillow_heif.register_heif_opener()

INPUT_DIR = "data/event_photos_heic"
OUTPUT_DIR = "data/event_photos"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith((".heic", ".heif")):
        heic_path = os.path.join(INPUT_DIR, filename)
        png_path = os.path.join(
            OUTPUT_DIR,
            os.path.splitext(filename)[0] + ".png"
        )

        image = Image.open(heic_path)
        image.save(png_path, format="PNG")

print("HEIC → PNG conversion completed.")
