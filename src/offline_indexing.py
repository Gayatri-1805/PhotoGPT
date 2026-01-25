"""
Offline pipeline: Process event photos, extract face embeddings using CLIP, and build FAISS index.
This script should be run once to index all event photos.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face_utils import FaceProcessor
from src.faiss_utils import FaissIndexManager


def process_event_photos(
    event_photos_dir: str,
    output_dir: str,
    mode: str = 'full_image',
    image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')
) -> None:
    """
    Process all event photos and create FAISS index.
    
    Pipeline:
    1. Load all images from event_photos_dir
    2. Extract CLIP embeddings based on mode:
       - 'full_image': Embed entire images (for text search)
       - 'face': Detect faces and embed face regions (for face search)
    3. Normalize embeddings (L2 normalization)
    4. Build FAISS index
    5. Save index and metadata
    
    Args:
        event_photos_dir: Directory containing event photos
        output_dir: Directory to save index and metadata
        mode: Embedding mode ('full_image' or 'face')
        image_extensions: Tuple of valid image extensions
    """
    
    print("=" * 80)
    print(f"OFFLINE INDEXING PIPELINE - {mode.upper()} MODE")
    print("=" * 80)
    
    # Initialize face processor
    face_processor = FaceProcessor(det_size=(640, 640))
    
    # Get all image files
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(event_photos_dir).rglob(f'*{ext}'))
    
    image_paths = [str(p) for p in image_paths]
    
    if len(image_paths) == 0:
        print(f"❌ No images found in {event_photos_dir}")
        return
    
    print(f"\n📸 Found {len(image_paths)} event photos")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔧 Mode: {mode}\n")
    
    # Process all images
    all_embeddings = []
    all_metadata = []
    item_id = 0
    
    if mode == 'full_image':
        print("Processing event photos (full image embeddings)...")
        for img_path in tqdm(image_paths, desc="Extracting embeddings"):
            try:
                # Get full image embedding
                embedding = face_processor.get_full_image_embedding(img_path)
                
                all_embeddings.append(embedding)
                
                metadata = {
                    'item_id': item_id,
                    'image_path': img_path,
                    'mode': 'full_image'
                }
                all_metadata.append(metadata)
                item_id += 1
            
            except Exception as e:
                print(f"\n⚠ Error processing {os.path.basename(img_path)}: {str(e)}")
                continue
        
        print(f"\n✓ Processed {len(all_embeddings)} images")
    
    else:  # face mode
        print("Processing event photos (face detection + embeddings)...")
        for img_path in tqdm(image_paths, desc="Extracting faces"):
            try:
                # Detect all faces in the image
                faces = face_processor.detect_faces(img_path)
                
                # Store each detected face
                for face in faces:
                    all_embeddings.append(face['embedding'])
                    
                    metadata = {
                        'item_id': item_id,
                        'image_path': img_path,
                        'bbox': face['bbox'],
                        'det_score': face['det_score'],
                        'mode': 'face'
                    }
                    all_metadata.append(metadata)
                    item_id += 1
            
            except Exception as e:
                print(f"\n⚠ Error processing {os.path.basename(img_path)}: {str(e)}")
                continue
        
        print(f"\n✓ Detected {len(all_embeddings)} faces across {len(image_paths)} images")
        print(f"  Average: {len(all_embeddings)/len(image_paths):.2f} faces per image")
    
    if len(all_embeddings) == 0:
        print(f"\n❌ No items processed. Please check your event photos.")
        return
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(all_embeddings, dtype='float32')
    
    print(f"\n📊 Embedding shape: {embeddings_array.shape}")
    print(f"  Dimension: {embeddings_array.shape[1]}")
    
    # Create FAISS index
    print("\nBuilding FAISS index...")
    index_manager = FaissIndexManager(embedding_dim=512)
    index_manager.create_index(embeddings_array)
    
    # Save index and metadata
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(output_dir, 'faiss.index')
    metadata_path = os.path.join(output_dir, 'metadata.json')
    
    index_manager.save_index(index_path)
    index_manager.save_metadata(all_metadata, metadata_path)
    
    print("\n" + "=" * 80)
    print("✅ INDEXING COMPLETE!")
    print("=" * 80)
    print(f"📊 Statistics:")
    print(f"  Total event photos: {len(image_paths)}")
    print(f"  Total items indexed: {len(all_embeddings)}")
    print(f"  Mode: {mode}")
    print(f"  Index file: {index_path}")
    print(f"  Metadata file: {metadata_path}")
    print("=" * 80)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Offline indexing: Process event photos and build embeddings index'
    )
    parser.add_argument(
        '--event-photos-dir',
        type=str,
        default='data/event_photos',
        help='Directory containing event photos (default: data/event_photos)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/embeddings',
        help='Directory to save index and metadata (default: data/embeddings)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='full_image',
        choices=['full_image', 'face'],
        help='Indexing mode: full_image (for text search) or face (for face search)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.event_photos_dir):
        print(f"❌ Event photos directory not found: {args.event_photos_dir}")
        print(f"Please create the directory and add your event photos.")
        return
    
    # Run indexing pipeline
    process_event_photos(args.event_photos_dir, args.output_dir, args.mode)


if __name__ == '__main__':
    main()