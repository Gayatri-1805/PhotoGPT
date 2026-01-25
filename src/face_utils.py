"""
Face detection and embedding utilities using MediaPipe and CLIP.
Uses MediaPipe for face detection and CLIP for image embeddings.
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
import open_clip
from PIL import Image
from typing import List, Tuple, Optional
import os


class FaceProcessor:
    """Handles face detection using MediaPipe and embedding generation using CLIP."""
    
    def __init__(self, det_size=(640, 640)):
        """
        Initialize MediaPipe face detection and CLIP models.
        
        Args:
            det_size: Detection size for face detection (width, height)
        """
        print("Initializing MediaPipe and CLIP models...")
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 1 for full-range detection (better for varied distances)
            min_detection_confidence=0.5
        )
        
        # Initialize CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', 
            pretrained='laion2b_s34b_b79k'
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize tokenizer for text encoding
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
        self.det_size = det_size
        
        print(f"✓ Face detection and CLIP models loaded successfully (device: {self.device})")
    
    def detect_faces(self, image_path: str) -> List[dict]:
        """
        Detect all faces in an image and extract CLIP embeddings for each face.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries containing face info:
            - bbox: [x1, y1, x2, y2]
            - embedding: 512-D CLIP embedding of the face region
            - det_score: Detection confidence score
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        
        # Detect faces using MediaPipe
        results = self.face_detection.process(img_rgb)
        
        face_data = []
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox_rel = detection.location_data.relative_bounding_box
                
                # Convert to absolute coordinates
                x1 = int(bbox_rel.xmin * w)
                y1 = int(bbox_rel.ymin * h)
                box_w = int(bbox_rel.width * w)
                box_h = int(bbox_rel.height * h)
                x2 = x1 + box_w
                y2 = y1 + box_h
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                bbox = [x1, y1, x2, y2]
                
                # Extract face region
                face_crop = img_rgb[y1:y2, x1:x2]
                
                # Skip if face crop is too small
                if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                    continue
                
                # Get CLIP embedding for the face region
                embedding = self._get_clip_embedding(face_crop)
                
                # Get detection confidence
                det_score = detection.score[0]
                
                face_data.append({
                    'bbox': bbox,
                    'embedding': embedding,
                    'det_score': det_score
                })
        
        return face_data
    
    def _get_clip_embedding(self, image_crop: np.ndarray) -> np.ndarray:
        """
        Get CLIP embedding for an image crop.
        
        Args:
            image_crop: RGB image crop (numpy array)
            
        Returns:
            Normalized CLIP embedding (512-D for ViT-B-32)
        """
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image_crop)
        
        # Preprocess and get embedding
        with torch.no_grad():
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            embedding = self.model.encode_image(image_tensor)
            
            # Normalize embedding
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            embedding = embedding.cpu().numpy().flatten()
        
        return embedding
    
    def extract_single_face_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract embedding from a single face (e.g., selfie).
        Returns None if no face or multiple faces detected.
        
        Args:
            image_path: Path to selfie image
            
        Returns:
            512-D normalized CLIP embedding or None
        """
        faces = self.detect_faces(image_path)
        
        if len(faces) == 0:
            print(f"⚠ No face detected in {os.path.basename(image_path)}")
            return None
        
        if len(faces) > 1:
            print(f"⚠ Multiple faces detected in {os.path.basename(image_path)}, using the largest face")
            # Use the face with largest bounding box area
            faces.sort(key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]), reverse=True)
        
        return faces[0]['embedding']
    
    def get_full_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Get CLIP embedding for a full image (no face detection).
        
        Args:
            image_path: Path to the image file
            
        Returns:
            512-D normalized CLIP embedding of the entire image
        """
        # Read and convert image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get CLIP embedding for full image
        embedding = self._get_clip_embedding(img_rgb)
        
        return embedding
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text description using CLIP text encoder.
        
        Args:
            text: Natural language description of the photo
            
        Returns:
            512-D normalized CLIP text embedding
        """
        # Tokenize text
        text_tokens = self.tokenizer([text]).to(self.device)
        
        # Get text embedding
        with torch.no_grad():
            text_embedding = self.model.encode_text(text_tokens)
            
            # Normalize embedding
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            
            # Convert to numpy
            text_embedding = text_embedding.cpu().numpy().flatten()
        
        return text_embedding
    
    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """
        L2 normalize an embedding vector.
        
        Args:
            embedding: Face embedding vector
            
        Returns:
            L2 normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    @staticmethod
    def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        Since embeddings are L2 normalized, this is just dot product.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score [-1, 1], higher is more similar
        """
        return float(np.dot(embedding1, embedding2))
    
    def __del__(self):
        """Clean up MediaPipe resources."""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()