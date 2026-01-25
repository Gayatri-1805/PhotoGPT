"""
Online pipeline: Query face matching using selfie.
Searches FAISS index for similar faces and returns matching event photos.
"""

import os
import numpy as np
from typing import List, Dict, Optional
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face_utils import FaceProcessor
from src.faiss_utils import FaissIndexManager


class PhotoRetriever:
    """Handles querying and retrieving event photos based on selfie."""
    
    def __init__(
        self,
        index_path: str = 'data/embeddings/faiss.index',
        metadata_path: str = 'data/embeddings/metadata.json'
    ):
        """
        Initialize photo retriever with pre-built index.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
        """
        print("Initializing Photo Retriever...")
        
        # Initialize face processor
        self.face_processor = FaceProcessor(det_size=(640, 640))
        
        # Initialize and load FAISS index
        self.index_manager = FaissIndexManager(embedding_dim=512)
        
        try:
            self.index_manager.load_index(index_path)
            self.index_manager.load_metadata(metadata_path)
            print("✓ Photo Retriever ready!")
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("Please run offline indexing first.")
            raise
    
    def find_photos(
        self,
        selfie_path: str = None,
        text_query: str = None,
        similarity_threshold: float = 0.5,
        max_results: int = 100
    ) -> Dict:
        """
        Find all event photos using either a selfie or text description.
        
        Pipeline:
        1. Extract embedding from selfie OR encode text query
        2. Search FAISS index
        3. Compute cosine similarity
        4. Filter by threshold
        5. Group results by image (if face mode)
        6. Return matched photos
        
        Args:
            selfie_path: Path to user's selfie (for face search mode)
            text_query: Natural language description of the photo (for text search mode)
            similarity_threshold: Minimum cosine similarity (0-1)
                                 Higher = stricter matching
                                 Recommended: 0.2-0.4 for text, 0.4-0.6 for faces
            max_results: Maximum number of matches to consider
            
        Returns:
            Dictionary containing:
            - success: bool
            - message: str
            - query_info: dict with query metadata
            - matches: list of matched photos with metadata
            - total_photos: int
        """
        result = {
            'success': False,
            'message': '',
            'query_info': {},
            'matches': [],
            'total_photos': 0
        }
        
        # Validate inputs
        if not selfie_path and not text_query:
            result['message'] = "Please provide either a selfie or a text description."
            return result
        
        if selfie_path and text_query:
            result['message'] = "Please provide only one: selfie OR text description."
            return result
        
        # Get query embedding
        if text_query:
            # Text search mode
            print(f"\n🔍 Processing text query: '{text_query}'")
            query_embedding = self.face_processor.encode_text(text_query)
            print("✓ Text encoded")
            
            result['query_info'] = {
                'query_type': 'text',
                'query_text': text_query,
                'embedding_dim': query_embedding.shape[0],
                'similarity_threshold': similarity_threshold
            }
        else:
            # Face search mode
            print(f"\n🔍 Processing selfie: {os.path.basename(selfie_path)}")
            query_embedding = self.face_processor.extract_single_face_embedding(selfie_path)
            
            if query_embedding is None:
                result['message'] = "No face detected in selfie or multiple faces found. Please upload a clear selfie with only one face."
                return result
            
            print("✓ Face detected in selfie")
            
            result['query_info'] = {
                'query_type': 'face',
                'selfie_path': selfie_path,
                'embedding_dim': query_embedding.shape[0],
                'similarity_threshold': similarity_threshold
            }
        
        # Search and filter
        print(f"🔎 Searching index with threshold: {similarity_threshold}")
        
        matches = self.index_manager.search_with_threshold(
            query_embedding,
            similarity_threshold=similarity_threshold,
            max_results=max_results
        )
        
        if len(matches) == 0:
            result['message'] = f"No confident matches found (threshold: {similarity_threshold}). Try lowering the threshold."
            return result
        
        print(f"✓ Found {len(matches)} matches")
        
        # Check metadata mode to determine how to process results
        if len(self.index_manager.metadata) > 0:
            mode = self.index_manager.metadata[0].get('mode', 'face')
        else:
            mode = 'face'
        
        if mode == 'full_image':
            # Full image mode - each match is a complete image
            matched_photos = []
            for match in matches:
                metadata = match['metadata']
                photo_info = {
                    'image_path': metadata['image_path'],
                    'similarity': match['similarity'],
                    'max_similarity': match['similarity'],
                    'avg_similarity': match['similarity'],
                    'num_matches': 1
                }
                matched_photos.append(photo_info)
        else:
            # Face mode - group by image and aggregate
            photo_groups = {}
            
            for match in matches:
                metadata = match['metadata']
                image_path = metadata['image_path']
                
                if image_path not in photo_groups:
                    photo_groups[image_path] = {
                        'image_path': image_path,
                        'faces': [],
                        'max_similarity': 0.0,
                        'avg_similarity': 0.0
                    }
                
                face_info = {
                    'bbox': metadata.get('bbox', [0, 0, 0, 0]),
                    'similarity': match['similarity'],
                    'det_score': metadata.get('det_score', 1.0)
                }
                
                photo_groups[image_path]['faces'].append(face_info)
                
                # Track max similarity for this photo
                if match['similarity'] > photo_groups[image_path]['max_similarity']:
                    photo_groups[image_path]['max_similarity'] = match['similarity']
            
            # Calculate average similarity and prepare final results
            matched_photos = []
            for photo_info in photo_groups.values():
                similarities = [f['similarity'] for f in photo_info['faces']]
                photo_info['avg_similarity'] = sum(similarities) / len(similarities)
                photo_info['num_matches'] = len(photo_info['faces'])
                matched_photos.append(photo_info)
        
        # Sort by max similarity (best match first)
        matched_photos.sort(key=lambda x: x['max_similarity'], reverse=True)
        
        # Prepare result
        result['success'] = True
        result['matches'] = matched_photos
        result['total_photos'] = len(matched_photos)
        result['message'] = f"Found {len(matched_photos)} photos matching your query"
        
        print(f"✅ {result['message']}")
        
        return result
    
    def find_photos_by_embedding(
        self,
        query_embedding: np.ndarray,
        similarity_threshold: float = 0.5,
        max_results: int = 100,
        person_name: str = None
    ) -> Dict:
        """
        Find all event photos using a pre-computed embedding (e.g., from registered person).
        
        This method matches the registered person's face embedding against all faces
        detected in the indexed event photos.
        
        Args:
            query_embedding: Pre-computed face embedding from registered selfie
            similarity_threshold: Minimum cosine similarity (0-1)
            max_results: Maximum number of face matches to consider
            person_name: Name of the person (for display)
            
        Returns:
            Dictionary containing search results with matched event photos
        """
        result = {
            'success': False,
            'message': '',
            'query_info': {},
            'matches': [],
            'total_photos': 0
        }
        
        print(f"\n{'='*80}")
        print(f"🔍 SEARCHING EVENT PHOTOS FOR: {person_name if person_name else 'PERSON'}")
        print(f"{'='*80}")
        print(f"Total indexed faces from event photos: {self.index_manager.index.ntotal}")
        print(f"Similarity threshold: {similarity_threshold}")
        
        result['query_info'] = {
            'query_type': 'registered_person',
            'person_name': person_name,
            'embedding_dim': query_embedding.shape[0],
            'similarity_threshold': similarity_threshold
        }
        
        # Search FAISS index for similar faces from event photos
        print(f"\n🔎 Matching face against all event photo faces...")
        
        matches = self.index_manager.search_with_threshold(
            query_embedding,
            similarity_threshold=similarity_threshold,
            max_results=max_results
        )
        
        if len(matches) == 0:
            print(f"❌ No matching faces found in event photos")
            result['message'] = f"No photos found for {person_name if person_name else 'this person'}. Try lowering the threshold."
            return result
        
        print(f"✓ Found {len(matches)} matching face(s) in event photos")
        
        # Group matches by image path (event photo)
        photo_groups = {}
        
        for match in matches:
            metadata = match['metadata']
            image_path = metadata['image_path']
            
            if image_path not in photo_groups:
                photo_groups[image_path] = {
                    'image_path': image_path,
                    'faces': [],
                    'max_similarity': 0.0,
                    'avg_similarity': 0.0
                }
            
            face_info = {
                'bbox': metadata.get('bbox', [0, 0, 0, 0]),
                'similarity': match['similarity'],
                'det_score': metadata.get('det_score', 1.0)
            }
            
            photo_groups[image_path]['faces'].append(face_info)
            
            if match['similarity'] > photo_groups[image_path]['max_similarity']:
                photo_groups[image_path]['max_similarity'] = match['similarity']
        
        print(f"✓ Grouped into {len(photo_groups)} unique event photos")
        
        # Calculate stats for each event photo
        matched_photos = []
        for photo_info in photo_groups.values():
            similarities = [f['similarity'] for f in photo_info['faces']]
            photo_info['avg_similarity'] = sum(similarities) / len(similarities)
            photo_info['num_matches'] = len(photo_info['faces'])
            matched_photos.append(photo_info)
            
            print(f"  📸 {os.path.basename(photo_info['image_path'])}: {photo_info['num_matches']} face(s), max similarity: {photo_info['max_similarity']:.3f}")
        
        # Sort by max similarity (best matches first)
        matched_photos.sort(key=lambda x: x['max_similarity'], reverse=True)
        
        # Prepare result
        result['success'] = True
        result['matches'] = matched_photos
        result['total_photos'] = len(matched_photos)
        result['message'] = f"Found {len(matched_photos)} event photo(s) containing {person_name if person_name else 'this person'}"
        
        print(f"\n{'='*80}")
        print(f"✅ {result['message']}")
        print(f"{'='*80}\n")
        
        return result
    
    def get_match_summary(self, result: Dict) -> str:
        """
        Generate a human-readable summary of match results.
        
        Args:
            result: Result dictionary from find_photos()
            
        Returns:
            Formatted summary string
        """
        if not result['success']:
            return f"❌ {result['message']}"
        
        summary = f"✅ {result['message']}\n"
        summary += f"📊 Statistics:\n"
        summary += f"  Total photos matched: {result['total_photos']}\n"
        
        if result['total_photos'] > 0:
            all_similarities = []
            for photo in result['matches']:
                all_similarities.append(photo['max_similarity'])
            
            summary += f"  Best match similarity: {max(all_similarities):.3f}\n"
            summary += f"  Average similarity: {sum(all_similarities)/len(all_similarities):.3f}\n"
            summary += f"  Similarity threshold used: {result['query_info']['similarity_threshold']}\n"
        
        return summary


def demo_query(
    selfie_path: str,
    index_path: str = 'data/embeddings/faiss.index',
    metadata_path: str = 'data/embeddings/metadata.json',
    similarity_threshold: float = 0.5
):
    """
    Demo function to test photo retrieval.
    
    Args:
        selfie_path: Path to selfie image
        index_path: Path to FAISS index
        metadata_path: Path to metadata JSON
        similarity_threshold: Similarity threshold for matching
    """
    print("=" * 80)
    print("ONLINE QUERY PIPELINE - Demo")
    print("=" * 80)
    
    # Initialize retriever
    retriever = PhotoRetriever(index_path, metadata_path)
    
    # Find photos
    result = retriever.find_photos(selfie_path, similarity_threshold=similarity_threshold)
    
    # Print summary
    print("\n" + "=" * 80)
    print(retriever.get_match_summary(result))
    print("=" * 80)
    
    # Print top 5 matches
    if result['success'] and result['total_photos'] > 0:
        print(f"\nTop 5 matched photos:")
        for i, photo in enumerate(result['matches'][:5], 1):
            print(f"\n{i}. {os.path.basename(photo['image_path'])}")
            print(f"   Max similarity: {photo['max_similarity']:.3f}")
            print(f"   Faces detected: {photo['num_matches']}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Query event photos using a selfie')
    parser.add_argument('--selfie', type=str, required=True, help='Path to selfie image')
    parser.add_argument('--threshold', type=float, default=0.5, help='Similarity threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.selfie):
        print(f"❌ Selfie not found: {args.selfie}")
    else:
        demo_query(args.selfie, similarity_threshold=args.threshold)