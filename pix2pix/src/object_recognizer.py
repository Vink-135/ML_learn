"""
Intelligent Object Recognition and Generation System
Recognizes drawn objects and generates realistic, properly colored images
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
from typing import Tuple, Dict, List, Optional

class ObjectRecognizer:
    def __init__(self):
        self.object_templates = self._initialize_object_templates()
        self.color_mappings = self._initialize_color_mappings()
        
    def _initialize_object_templates(self) -> Dict:
        """Initialize templates for common objects"""
        return {
            'apple': {
                'shape': 'round_with_stem',
                'typical_colors': [(220, 20, 60), (34, 139, 34), (255, 215, 0)],  # Red, Green, Yellow
                'size_ratio': (0.8, 1.0),  # width/height ratio
                'features': ['round', 'stem_top', 'smooth']
            },
            'car': {
                'shape': 'rectangular_with_wheels',
                'typical_colors': [(255, 0, 0), (0, 0, 255), (128, 128, 128), (0, 0, 0)],  # Red, Blue, Gray, Black
                'size_ratio': (2.0, 1.0),
                'features': ['rectangular', 'wheels', 'windows']
            },
            'house': {
                'shape': 'square_with_triangle',
                'typical_colors': [(139, 69, 19), (255, 255, 255), (220, 20, 60)],  # Brown, White, Red roof
                'size_ratio': (1.2, 1.0),
                'features': ['square_base', 'triangle_roof', 'door', 'windows']
            },
            'tree': {
                'shape': 'trunk_with_crown',
                'typical_colors': [(139, 69, 19), (34, 139, 34)],  # Brown trunk, Green leaves
                'size_ratio': (0.6, 1.5),
                'features': ['vertical_trunk', 'round_crown']
            },
            'flower': {
                'shape': 'petals_with_center',
                'typical_colors': [(255, 192, 203), (255, 255, 0), (255, 0, 255), (255, 165, 0)],  # Pink, Yellow, Magenta, Orange
                'size_ratio': (1.0, 1.0),
                'features': ['petals', 'center', 'stem']
            },
            'sun': {
                'shape': 'circle_with_rays',
                'typical_colors': [(255, 255, 0), (255, 215, 0)],  # Yellow, Gold
                'size_ratio': (1.0, 1.0),
                'features': ['circle', 'rays']
            },
            'cat': {
                'shape': 'oval_with_ears',
                'typical_colors': [(255, 165, 0), (128, 128, 128), (0, 0, 0), (255, 255, 255)],  # Orange, Gray, Black, White
                'size_ratio': (1.3, 1.0),
                'features': ['oval_body', 'triangular_ears', 'tail']
            },
            'fish': {
                'shape': 'oval_with_tail',
                'typical_colors': [(255, 165, 0), (0, 191, 255), (255, 20, 147)],  # Orange, Blue, Pink
                'size_ratio': (1.8, 1.0),
                'features': ['oval_body', 'tail_fin', 'fins']
            }
        }
    
    def _initialize_color_mappings(self) -> Dict:
        """Initialize natural color mappings for objects"""
        return {
            'apple': {
                'primary': (220, 20, 60),    # Red
                'secondary': (34, 139, 34),   # Green (stem/leaf)
                'highlight': (255, 182, 193), # Light red (shine)
                'shadow': (139, 0, 0)         # Dark red
            },
            'car': {
                'primary': (255, 0, 0),       # Red
                'secondary': (128, 128, 128), # Gray (wheels/details)
                'highlight': (255, 255, 255), # White (windows/lights)
                'shadow': (64, 64, 64)        # Dark gray
            },
            'house': {
                'primary': (255, 255, 255),   # White walls
                'secondary': (220, 20, 60),   # Red roof
                'highlight': (139, 69, 19),   # Brown (door/trim)
                'shadow': (105, 105, 105)     # Gray shadow
            },
            'tree': {
                'primary': (34, 139, 34),     # Green leaves
                'secondary': (139, 69, 19),   # Brown trunk
                'highlight': (144, 238, 144), # Light green
                'shadow': (0, 100, 0)         # Dark green
            },
            'flower': {
                'primary': (255, 192, 203),   # Pink petals
                'secondary': (255, 255, 0),   # Yellow center
                'highlight': (255, 255, 255), # White highlights
                'shadow': (219, 112, 147)     # Dark pink
            },
            'sun': {
                'primary': (255, 255, 0),     # Yellow
                'secondary': (255, 215, 0),   # Gold
                'highlight': (255, 255, 224), # Light yellow
                'shadow': (255, 140, 0)       # Orange
            },
            'cat': {
                'primary': (255, 165, 0),     # Orange
                'secondary': (255, 192, 203), # Pink (nose/ears)
                'highlight': (255, 255, 255), # White
                'shadow': (205, 133, 63)      # Dark orange
            },
            'fish': {
                'primary': (255, 165, 0),     # Orange
                'secondary': (0, 191, 255),   # Blue
                'highlight': (255, 255, 255), # White
                'shadow': (255, 140, 0)       # Dark orange
            }
        }
    
    def recognize_object(self, drawing_array: np.ndarray) -> Tuple[str, float]:
        """
        Recognize what object is drawn based on shape analysis
        Returns: (object_name, confidence_score)
        """
        try:
            # Ensure input is uint8
            if drawing_array.dtype != np.uint8:
                drawing_array = np.clip(drawing_array, 0, 255).astype(np.uint8)

            # Convert to grayscale for analysis
            if len(drawing_array.shape) == 3:
                gray = cv2.cvtColor(drawing_array, cv2.COLOR_RGB2GRAY) if drawing_array.shape[2] == 3 else drawing_array[:,:,0]
            else:
                gray = drawing_array

            # Ensure gray is uint8
            if gray.dtype != np.uint8:
                gray = np.clip(gray, 0, 255).astype(np.uint8)

            # Find contours
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception as e:
            print(f"Error in object recognition: {e}")
            return 'apple', 0.5  # Default fallback
        
        if not contours:
            return 'unknown', 0.0
            
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Analyze shape features
        features = self._analyze_shape_features(largest_contour, gray.shape)
        
        # Match against known objects
        best_match = 'apple'  # Default to apple
        best_score = 0.0
        
        for obj_name, obj_template in self.object_templates.items():
            score = self._calculate_match_score(features, obj_template)
            if score > best_score:
                best_score = score
                best_match = obj_name
                
        return best_match, best_score
    
    def _analyze_shape_features(self, contour, image_shape) -> Dict:
        """Analyze geometric features of the drawn shape"""
        # Calculate basic properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 1.0
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Extent
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        return {
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'solidity': solidity,
            'extent': extent,
            'area_ratio': area / (image_shape[0] * image_shape[1])
        }
    
    def _calculate_match_score(self, features: Dict, template: Dict) -> float:
        """Calculate how well the features match a template"""
        score = 0.0
        
        # Check aspect ratio
        expected_ratio = template['size_ratio'][0] / template['size_ratio'][1]
        ratio_diff = abs(features['aspect_ratio'] - expected_ratio)
        ratio_score = max(0, 1 - ratio_diff)
        score += ratio_score * 0.3
        
        # Check circularity for round objects
        if 'round' in template['features']:
            score += features['circularity'] * 0.4
        elif 'rectangular' in template['features']:
            score += (1 - features['circularity']) * 0.4
        else:
            score += 0.2  # Neutral score
            
        # Check solidity
        score += features['solidity'] * 0.3
        
        return min(score, 1.0)
    
    def get_object_colors(self, object_name: str) -> Dict:
        """Get the natural colors for a recognized object"""
        return self.color_mappings.get(object_name, self.color_mappings['apple'])
    
    def get_object_template(self, object_name: str) -> Dict:
        """Get the template for a recognized object"""
        return self.object_templates.get(object_name, self.object_templates['apple'])
