"""
Person profile management - stores selfie embeddings with names.
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face_utils import FaceProcessor


class PersonManager:
    """Manages person profiles (name + selfie embedding)."""
    
    def __init__(self, profiles_path: str = 'data/embeddings/person_profiles.json'):
        """
        Initialize person manager.
        
        Args:
            profiles_path: Path to JSON file storing person profiles
        """
        self.profiles_path = profiles_path
        self.profiles = {}  # name -> {embedding, selfie_path}
        self.face_processor = None
        self.load_profiles()
    
    def _init_face_processor(self):
        """Lazy initialization of face processor."""
        if self.face_processor is None:
            self.face_processor = FaceProcessor(det_size=(640, 640))
    
    def load_profiles(self) -> None:
        """Load person profiles from disk."""
        if os.path.exists(self.profiles_path):
            with open(self.profiles_path, 'r') as f:
                data = json.load(f)
                # Convert embeddings back to numpy arrays
                self.profiles = {}
                for name, profile in data.items():
                    self.profiles[name] = {
                        'embedding': np.array(profile['embedding']),
                        'selfie_path': profile['selfie_path']
                    }
            print(f"✓ Loaded {len(self.profiles)} person profiles")
        else:
            self.profiles = {}
            print("No existing profiles found")
    
    def save_profiles(self) -> None:
        """Save person profiles to disk."""
        os.makedirs(os.path.dirname(self.profiles_path), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        data = {}
        for name, profile in self.profiles.items():
            data[name] = {
                'embedding': profile['embedding'].tolist(),
                'selfie_path': profile['selfie_path']
            }
        
        with open(self.profiles_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved {len(self.profiles)} person profiles")
    
    def register_person(self, name: str, selfie_path: str) -> Dict:
        """
        Register a new person with their selfie.
        
        Args:
            name: Person's name
            selfie_path: Path to their selfie image
            
        Returns:
            Result dictionary with success status and message
        """
        result = {
            'success': False,
            'message': ''
        }
        
        name = name.strip()
        if not name:
            result['message'] = "Please enter a name"
            return result
        
        if not os.path.exists(selfie_path):
            result['message'] = f"Selfie file not found: {selfie_path}"
            return result
        
        # Initialize face processor
        self._init_face_processor()
        
        # Extract face embedding from selfie
        try:
            embedding = self.face_processor.extract_single_face_embedding(selfie_path)
            
            if embedding is None:
                result['message'] = "No face detected in selfie. Please upload a clear selfie with one face."
                return result
            
            # Store profile
            self.profiles[name] = {
                'embedding': embedding,
                'selfie_path': selfie_path
            }
            
            self.save_profiles()
            
            result['success'] = True
            result['message'] = f"✅ Successfully registered {name}!"
            
        except Exception as e:
            result['message'] = f"Error processing selfie: {str(e)}"
        
        return result
    
    def get_person_embedding(self, name: str) -> Optional[np.ndarray]:
        """
        Get the face embedding for a person by name.
        
        Args:
            name: Person's name (case-insensitive)
            
        Returns:
            Face embedding or None if not found
        """
        # Case-insensitive search
        for stored_name, profile in self.profiles.items():
            if stored_name.lower() == name.lower():
                return profile['embedding']
        return None
    
    def get_all_names(self) -> List[str]:
        """Get list of all registered person names."""
        return sorted(list(self.profiles.keys()))
    
    def remove_person(self, name: str) -> bool:
        """
        Remove a person's profile.
        
        Args:
            name: Person's name
            
        Returns:
            True if removed, False if not found
        """
        if name in self.profiles:
            del self.profiles[name]
            self.save_profiles()
            return True
        return False
    
    def get_profile(self, name: str) -> Optional[Dict]:
        """
        Get complete profile for a person.
        
        Args:
            name: Person's name (case-insensitive)
            
        Returns:
            Profile dict or None
        """
        for stored_name, profile in self.profiles.items():
            if stored_name.lower() == name.lower():
                return {
                    'name': stored_name,
                    'embedding': profile['embedding'],
                    'selfie_path': profile['selfie_path']
                }
        return None
