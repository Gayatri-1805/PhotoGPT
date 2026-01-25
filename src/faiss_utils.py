"""
FAISS index creation, saving, loading, and searching utilities.
"""

import faiss
import numpy as np
import json
import os
from typing import List, Tuple, Dict


class FaissIndexManager:
    """Manages FAISS index for face embeddings."""
    
    def __init__(self, embedding_dim: int = 512):
        """
        Initialize FAISS index manager.
        
        Args:
            embedding_dim: Dimension of face embeddings (512 for CLIP ViT-B-32)
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []
    
    def create_index(self, embeddings: np.ndarray) -> None:
        """
        Create a FAISS index from embeddings.
        Uses IndexFlatL2 for exact search with L2 distance.
        Since embeddings are normalized, L2 distance is equivalent to cosine similarity.
        
        Args:
            embeddings: Array of shape (n_faces, embedding_dim)
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {embeddings.shape[1]}")
        
        # Create flat L2 index (exact search)
        # For normalized vectors: L2 distance = 2(1 - cosine_similarity)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add embeddings to index
        self.index.add(embeddings.astype('float32'))
        
        print(f"✓ FAISS index created with {self.index.ntotal} embeddings")
    
    def save_index(self, index_path: str) -> None:
        """
        Save FAISS index to disk.
        
        Args:
            index_path: Path to save the index file
        """
        if self.index is None:
            raise ValueError("No index to save. Create index first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, index_path)
        print(f"✓ FAISS index saved to {index_path}")
    
    def load_index(self, index_path: str) -> None:
        """
        Load FAISS index from disk.
        
        Args:
            index_path: Path to the index file
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        print(f"✓ FAISS index loaded from {index_path}")
        print(f"  Total embeddings: {self.index.ntotal}")
    
    def save_metadata(self, metadata: List[Dict], metadata_path: str) -> None:
        """
        Save metadata (image paths, bboxes, face IDs) to JSON.
        
        Args:
            metadata: List of metadata dictionaries
            metadata_path: Path to save metadata JSON
        """
        self.metadata = metadata
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Save as JSON
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata saved to {metadata_path}")
        print(f"  Total face records: {len(metadata)}")
    
    def load_metadata(self, metadata_path: str) -> List[Dict]:
        """
        Load metadata from JSON.
        
        Args:
            metadata_path: Path to metadata JSON file
            
        Returns:
            List of metadata dictionaries
        """
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print(f"✓ Metadata loaded from {metadata_path}")
        print(f"  Total face records: {len(self.metadata)}")
        
        return self.metadata
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors in the index.
        
        Args:
            query_embedding: Query face embedding (512-D CLIP embedding)
            k: Number of nearest neighbors to retrieve
            
        Returns:
            distances: L2 distances to k nearest neighbors
            indices: Indices of k nearest neighbors in the index
        """
        if self.index is None:
            raise ValueError("No index loaded. Load or create index first.")
        
        # Reshape query to (1, embedding_dim)
        query = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query, k)
        
        return distances[0], indices[0]
    
    def search_with_threshold(
        self, 
        query_embedding: np.ndarray, 
        similarity_threshold: float = 0.5,
        max_results: int = 100
    ) -> List[Dict]:
        """
        Search and filter results by similarity threshold.
        
        Args:
            query_embedding: Query face embedding
            similarity_threshold: Minimum cosine similarity (0-1)
            max_results: Maximum number of results to consider
            
        Returns:
            List of matching results with metadata
        """
        # Search for top max_results
        distances, indices = self.search(query_embedding, k=max_results)
        
        results = []
        for dist, idx in zip(distances, indices):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            # Convert L2 distance to cosine similarity
            # For normalized vectors: cosine_sim = 1 - (L2_dist^2 / 2)
            cosine_similarity = 1 - (dist / 2)
            
            # Filter by threshold
            if cosine_similarity >= similarity_threshold:
                result = {
                    'face_id': int(idx),
                    'similarity': float(cosine_similarity),
                    'distance': float(dist),
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else None
                }
                results.append(result)
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results