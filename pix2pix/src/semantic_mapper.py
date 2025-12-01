"""
Semantic Color Mapping System for Pix2Pix
Maps semantic colors to realistic building elements like the classic facades example
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
from typing import Dict, Tuple, List
import random

class SemanticMapper:
    def __init__(self):
        self.semantic_colors = self._initialize_semantic_colors()
        self.texture_mappings = self._initialize_texture_mappings()
        
    def _initialize_semantic_colors(self) -> Dict:
        """Initialize semantic color palette for building elements"""
        return {
            # Building structure
            'wall': (255, 165, 0),      # Orange - main wall
            'roof': (255, 0, 0),        # Red - roof
            'foundation': (139, 69, 19), # Brown - foundation/base
            
            # Openings
            'window': (0, 0, 255),      # Blue - windows
            'door': (0, 255, 0),        # Green - doors
            'balcony': (255, 255, 0),   # Yellow - balconies
            
            # Details
            'trim': (255, 255, 255),    # White - window trim, decorative elements
            'shutters': (128, 0, 128),  # Purple - shutters
            'awning': (255, 192, 203),  # Pink - awnings
            
            # Background
            'sky': (135, 206, 235),     # Sky blue - background
            'ground': (128, 128, 128),  # Gray - ground/street
            
            # Special elements
            'chimney': (139, 0, 0),     # Dark red - chimney
            'stairs': (169, 169, 169),  # Dark gray - stairs
        }
    
    def _initialize_texture_mappings(self) -> Dict:
        """Map semantic colors to realistic textures and colors"""
        return {
            'wall': {
                'colors': [(222, 184, 135), (205, 133, 63), (210, 180, 140)],  # Tan, brown, beige
                'texture': 'brick',
                'patterns': ['horizontal_lines', 'brick_pattern']
            },
            'roof': {
                'colors': [(139, 69, 19), (160, 82, 45), (128, 0, 0)],  # Brown, saddle brown, maroon
                'texture': 'tiles',
                'patterns': ['tile_pattern', 'shingle_pattern']
            },
            'window': {
                'colors': [(173, 216, 230), (135, 206, 235), (70, 130, 180)],  # Light blue, sky blue, steel blue
                'texture': 'glass',
                'patterns': ['glass_reflection', 'window_frame']
            },
            'door': {
                'colors': [(139, 69, 19), (160, 82, 45), (101, 67, 33)],  # Brown shades
                'texture': 'wood',
                'patterns': ['wood_grain', 'panel_door']
            },
            'trim': {
                'colors': [(255, 255, 255), (248, 248, 255), (245, 245, 245)],  # White shades
                'texture': 'painted',
                'patterns': ['smooth', 'decorative_molding']
            },
            'balcony': {
                'colors': [(192, 192, 192), (169, 169, 169), (128, 128, 128)],  # Gray shades
                'texture': 'metal',
                'patterns': ['railing', 'ornate_ironwork']
            },
            'foundation': {
                'colors': [(105, 105, 105), (119, 136, 153), (112, 128, 144)],  # Gray shades
                'texture': 'stone',
                'patterns': ['stone_blocks', 'rough_stone']
            }
        }
    
    def get_semantic_color(self, element_name: str) -> Tuple[int, int, int]:
        """Get the semantic color for a building element"""
        return self.semantic_colors.get(element_name, (128, 128, 128))
    
    def get_realistic_color(self, semantic_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Convert semantic color to realistic color"""
        # Find the closest semantic color
        element_name = self._find_closest_semantic_element(semantic_color)
        if element_name in self.texture_mappings:
            colors = self.texture_mappings[element_name]['colors']
            return random.choice(colors)
        return semantic_color
    
    def _find_closest_semantic_element(self, color: Tuple[int, int, int]) -> str:
        """Find the closest semantic element for a given color"""
        min_distance = float('inf')
        closest_element = 'wall'
        
        for element, semantic_color in self.semantic_colors.items():
            distance = sum((a - b) ** 2 for a, b in zip(color, semantic_color))
            if distance < min_distance:
                min_distance = distance
                closest_element = element
                
        return closest_element
    
    def convert_semantic_to_realistic(self, semantic_image: np.ndarray) -> np.ndarray:
        """Convert semantic colored image to realistic building facade"""
        height, width = semantic_image.shape[:2]
        realistic_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Process each pixel
        for y in range(height):
            for x in range(width):
                semantic_color = tuple(semantic_image[y, x, :3])
                realistic_color = self.get_realistic_color(semantic_color)
                realistic_image[y, x] = realistic_color
        
        # Apply texture and detail enhancements
        realistic_image = self._add_architectural_details(realistic_image, semantic_image)
        
        return realistic_image
    
    def _add_architectural_details(self, realistic_image: np.ndarray, semantic_image: np.ndarray) -> np.ndarray:
        """Add architectural details like window frames, brick patterns, etc."""
        img = Image.fromarray(realistic_image)
        draw = ImageDraw.Draw(img)
        
        height, width = semantic_image.shape[:2]
        
        # Add window frames and details
        for y in range(height):
            for x in range(width):
                semantic_color = tuple(semantic_image[y, x, :3])
                element = self._find_closest_semantic_element(semantic_color)
                
                if element == 'window':
                    self._add_window_details(draw, x, y, realistic_image)
                elif element == 'door':
                    self._add_door_details(draw, x, y, realistic_image)
                elif element == 'wall':
                    self._add_wall_texture(draw, x, y, realistic_image)
        
        # Apply slight blur for realism
        img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
        
        return np.array(img)
    
    def _add_window_details(self, draw, x, y, image):
        """Add window frame and glass reflection details"""
        # Add window frame (simplified)
        if x % 20 == 0 or y % 20 == 0:  # Simple grid pattern
            draw.point((x, y), fill=(255, 255, 255))  # White frame
        
        # Add glass reflection effect
        if (x + y) % 30 == 0:
            draw.point((x, y), fill=(200, 220, 255))  # Light blue reflection
    
    def _add_door_details(self, draw, x, y, image):
        """Add door panel and handle details"""
        # Add door panels (vertical lines)
        if x % 15 == 0:
            draw.point((x, y), fill=(120, 60, 30))  # Darker brown for panels
        
        # Add door handle area
        if x % 40 == 35 and y % 60 > 30:
            draw.point((x, y), fill=(255, 215, 0))  # Gold handle
    
    def _add_wall_texture(self, draw, x, y, image):
        """Add brick or stone texture to walls"""
        # Add brick pattern
        if (y % 20 == 0) or (x % 40 == 0 and (y // 20) % 2 == 0):
            draw.point((x, y), fill=(180, 140, 100))  # Mortar lines
    
    def create_building_facade(self, width: int = 256, height: int = 256, 
                             building_type: str = "classic") -> np.ndarray:
        """Create a semantic building facade template"""
        img = Image.new('RGB', (width, height), self.semantic_colors['sky'])
        draw = ImageDraw.Draw(img)
        
        if building_type == "classic":
            return self._create_classic_facade(draw, width, height)
        elif building_type == "modern":
            return self._create_modern_facade(draw, width, height)
        else:
            return self._create_simple_facade(draw, width, height)
    
    def _create_classic_facade(self, draw, width, height):
        """Create a classic building facade like the example"""
        # Main building wall
        wall_top = height // 6
        wall_bottom = height - height // 8
        draw.rectangle([20, wall_top, width-20, wall_bottom], fill=self.semantic_colors['wall'])
        
        # Roof
        draw.polygon([(10, wall_top), (width//2, wall_top-30), (width-10, wall_top)], 
                    fill=self.semantic_colors['roof'])
        
        # Foundation
        draw.rectangle([15, wall_bottom, width-15, height-10], fill=self.semantic_colors['foundation'])
        
        # Windows (3 floors, 4 windows per floor)
        window_width, window_height = 25, 35
        for floor in range(3):
            y_pos = wall_top + 20 + floor * 60
            for window in range(4):
                x_pos = 40 + window * 45
                draw.rectangle([x_pos, y_pos, x_pos + window_width, y_pos + window_height], 
                             fill=self.semantic_colors['window'])
                # Window trim
                draw.rectangle([x_pos-2, y_pos-2, x_pos + window_width+2, y_pos + window_height+2], 
                             outline=self.semantic_colors['trim'], width=2)
        
        # Main door
        door_width, door_height = 30, 50
        door_x = width // 2 - door_width // 2
        door_y = wall_bottom - door_height
        draw.rectangle([door_x, door_y, door_x + door_width, door_y + door_height], 
                      fill=self.semantic_colors['door'])
        
        # Balconies
        for floor in range(1, 3):  # Only upper floors
            y_pos = wall_top + 20 + floor * 60 + 35
            draw.rectangle([35, y_pos, width-35, y_pos + 8], fill=self.semantic_colors['balcony'])
        
        return np.array(img)
    
    def _create_modern_facade(self, draw, width, height):
        """Create a modern building facade"""
        # Simple modern design
        draw.rectangle([30, 40, width-30, height-20], fill=self.semantic_colors['wall'])
        
        # Large windows
        for floor in range(4):
            y_pos = 60 + floor * 45
            for window in range(3):
                x_pos = 50 + window * 50
                draw.rectangle([x_pos, y_pos, x_pos + 35, y_pos + 30], 
                             fill=self.semantic_colors['window'])
        
        return np.array(img)
    
    def _create_simple_facade(self, draw, width, height):
        """Create a simple house facade"""
        # House body
        draw.rectangle([40, height//2, width-40, height-30], fill=self.semantic_colors['wall'])
        
        # Triangular roof
        draw.polygon([(30, height//2), (width//2, height//3), (width-30, height//2)], 
                    fill=self.semantic_colors['roof'])
        
        # Door
        draw.rectangle([width//2-15, height-60, width//2+15, height-30], 
                      fill=self.semantic_colors['door'])
        
        # Windows
        draw.rectangle([60, height//2+20, 90, height//2+50], fill=self.semantic_colors['window'])
        draw.rectangle([width-90, height//2+20, width-60, height//2+50], fill=self.semantic_colors['window'])
        
        return np.array(img)
