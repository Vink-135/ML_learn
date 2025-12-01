"""
Realistic Object Generator
Creates detailed, properly colored images of recognized objects
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import cv2
from typing import Tuple, Dict
import math

class ObjectGenerator:
    def __init__(self):
        pass
    
    def generate_realistic_object(self, object_name: str, colors: Dict, 
                                size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """Generate a realistic image of the specified object"""
        
        generators = {
            'apple': self._generate_apple,
            'car': self._generate_car,
            'house': self._generate_house,
            'tree': self._generate_tree,
            'flower': self._generate_flower,
            'sun': self._generate_sun,
            'cat': self._generate_cat,
            'fish': self._generate_fish
        }
        
        generator_func = generators.get(object_name, self._generate_apple)
        return generator_func(colors, size)
    
    def _generate_apple(self, colors: Dict, size: Tuple[int, int]) -> np.ndarray:
        """Generate a realistic apple"""
        img = Image.new('RGB', size, (255, 255, 255))  # White background
        draw = ImageDraw.Draw(img)
        
        # Apple body (slightly oval)
        center_x, center_y = size[0] // 2, size[1] // 2 + 10
        radius_x, radius_y = size[0] // 3, size[1] // 3
        
        # Main apple body
        apple_bbox = [center_x - radius_x, center_y - radius_y, 
                     center_x + radius_x, center_y + radius_y]
        draw.ellipse(apple_bbox, fill=colors['primary'])
        
        # Apple indentation at top
        indent_y = center_y - radius_y + 10
        draw.ellipse([center_x - 15, indent_y - 10, center_x + 15, indent_y + 10], 
                    fill=colors['shadow'])
        
        # Stem
        stem_x = center_x
        stem_y = center_y - radius_y
        draw.rectangle([stem_x - 3, stem_y - 15, stem_x + 3, stem_y], 
                      fill=colors['secondary'])
        
        # Leaf
        leaf_points = [(stem_x + 5, stem_y - 10), (stem_x + 15, stem_y - 15), 
                      (stem_x + 10, stem_y - 5)]
        draw.polygon(leaf_points, fill=colors['secondary'])
        
        # Highlight for shine
        highlight_x = center_x - radius_x // 3
        highlight_y = center_y - radius_y // 3
        draw.ellipse([highlight_x - 15, highlight_y - 20, highlight_x + 15, highlight_y + 20], 
                    fill=colors['highlight'])
        
        # Apply blur for realism
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return np.array(img)
    
    def _generate_car(self, colors: Dict, size: Tuple[int, int]) -> np.ndarray:
        """Generate a realistic car"""
        img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = size[0] // 2, size[1] // 2 + 20
        car_width, car_height = size[0] // 2, size[1] // 4
        
        # Car body
        car_bbox = [center_x - car_width, center_y - car_height,
                   center_x + car_width, center_y + car_height]
        draw.rectangle(car_bbox, fill=colors['primary'])
        
        # Car roof
        roof_height = car_height // 2
        roof_bbox = [center_x - car_width + 30, center_y - car_height - roof_height,
                    center_x + car_width - 30, center_y - car_height]
        draw.rectangle(roof_bbox, fill=colors['primary'])
        
        # Windows
        window_bbox = [center_x - car_width + 35, center_y - car_height - roof_height + 5,
                      center_x + car_width - 35, center_y - car_height - 5]
        draw.rectangle(window_bbox, fill=colors['highlight'])
        
        # Wheels
        wheel_radius = 20
        wheel_y = center_y + car_height - 10
        # Left wheel
        draw.ellipse([center_x - car_width + 20, wheel_y - wheel_radius,
                     center_x - car_width + 20 + wheel_radius * 2, wheel_y + wheel_radius],
                    fill=colors['shadow'])
        # Right wheel
        draw.ellipse([center_x + car_width - 40, wheel_y - wheel_radius,
                     center_x + car_width - 40 + wheel_radius * 2, wheel_y + wheel_radius],
                    fill=colors['shadow'])
        
        return np.array(img)
    
    def _generate_house(self, colors: Dict, size: Tuple[int, int]) -> np.ndarray:
        """Generate a realistic house"""
        img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = size[0] // 2, size[1] // 2 + 30
        house_width, house_height = size[0] // 3, size[1] // 3
        
        # House base
        house_bbox = [center_x - house_width, center_y - house_height,
                     center_x + house_width, center_y + house_height]
        draw.rectangle(house_bbox, fill=colors['primary'])
        
        # Roof
        roof_points = [(center_x - house_width - 10, center_y - house_height),
                      (center_x, center_y - house_height - 40),
                      (center_x + house_width + 10, center_y - house_height)]
        draw.polygon(roof_points, fill=colors['secondary'])
        
        # Door
        door_width, door_height = 25, 50
        door_x = center_x - door_width // 2
        door_y = center_y + house_height - door_height
        draw.rectangle([door_x, door_y, door_x + door_width, door_y + door_height],
                      fill=colors['highlight'])
        
        # Windows
        window_size = 20
        # Left window
        draw.rectangle([center_x - house_width + 20, center_y - 20,
                       center_x - house_width + 20 + window_size, center_y - 20 + window_size],
                      fill=colors['highlight'])
        # Right window
        draw.rectangle([center_x + house_width - 40, center_y - 20,
                       center_x + house_width - 40 + window_size, center_y - 20 + window_size],
                      fill=colors['highlight'])
        
        return np.array(img)
    
    def _generate_tree(self, colors: Dict, size: Tuple[int, int]) -> np.ndarray:
        """Generate a realistic tree"""
        img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = size[0] // 2, size[1] // 2 + 40
        
        # Trunk
        trunk_width, trunk_height = 20, 80
        trunk_bbox = [center_x - trunk_width // 2, center_y - trunk_height // 2,
                     center_x + trunk_width // 2, center_y + trunk_height // 2]
        draw.rectangle(trunk_bbox, fill=colors['secondary'])
        
        # Tree crown (leaves)
        crown_radius = 60
        crown_y = center_y - trunk_height // 2 - 20
        draw.ellipse([center_x - crown_radius, crown_y - crown_radius,
                     center_x + crown_radius, crown_y + crown_radius],
                    fill=colors['primary'])
        
        # Add some texture to leaves
        for i in range(10):
            x = center_x + np.random.randint(-crown_radius + 10, crown_radius - 10)
            y = crown_y + np.random.randint(-crown_radius + 10, crown_radius - 10)
            draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=colors['highlight'])
        
        return np.array(img)
    
    def _generate_flower(self, colors: Dict, size: Tuple[int, int]) -> np.ndarray:
        """Generate a realistic flower"""
        img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = size[0] // 2, size[1] // 2
        
        # Stem
        draw.rectangle([center_x - 3, center_y + 20, center_x + 3, center_y + 80],
                      fill=(34, 139, 34))
        
        # Petals
        petal_length = 30
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            x1 = center_x + math.cos(rad) * 10
            y1 = center_y + math.sin(rad) * 10
            x2 = center_x + math.cos(rad) * petal_length
            y2 = center_y + math.sin(rad) * petal_length
            
            draw.ellipse([x2 - 12, y2 - 8, x2 + 12, y2 + 8], fill=colors['primary'])
        
        # Center
        draw.ellipse([center_x - 8, center_y - 8, center_x + 8, center_y + 8],
                    fill=colors['secondary'])
        
        return np.array(img)
    
    def _generate_sun(self, colors: Dict, size: Tuple[int, int]) -> np.ndarray:
        """Generate a realistic sun"""
        img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = size[0] // 2, size[1] // 2
        sun_radius = 40
        
        # Sun rays
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            x1 = center_x + math.cos(rad) * (sun_radius + 10)
            y1 = center_y + math.sin(rad) * (sun_radius + 10)
            x2 = center_x + math.cos(rad) * (sun_radius + 30)
            y2 = center_y + math.sin(rad) * (sun_radius + 30)
            draw.line([(x1, y1), (x2, y2)], fill=colors['primary'], width=4)
        
        # Sun body
        draw.ellipse([center_x - sun_radius, center_y - sun_radius,
                     center_x + sun_radius, center_y + sun_radius],
                    fill=colors['primary'])
        
        # Sun face
        # Eyes
        draw.ellipse([center_x - 15, center_y - 15, center_x - 5, center_y - 5],
                    fill=(0, 0, 0))
        draw.ellipse([center_x + 5, center_y - 15, center_x + 15, center_y - 5],
                    fill=(0, 0, 0))
        # Smile
        draw.arc([center_x - 20, center_y - 5, center_x + 20, center_y + 15],
                start=0, end=180, fill=(0, 0, 0), width=3)
        
        return np.array(img)
    
    def _generate_cat(self, colors: Dict, size: Tuple[int, int]) -> np.ndarray:
        """Generate a realistic cat"""
        img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = size[0] // 2, size[1] // 2 + 20
        
        # Cat body
        body_width, body_height = 60, 40
        draw.ellipse([center_x - body_width, center_y - body_height,
                     center_x + body_width, center_y + body_height],
                    fill=colors['primary'])
        
        # Cat head
        head_radius = 35
        head_y = center_y - body_height - 20
        draw.ellipse([center_x - head_radius, head_y - head_radius,
                     center_x + head_radius, head_y + head_radius],
                    fill=colors['primary'])
        
        # Ears
        ear_points1 = [(center_x - 25, head_y - head_radius + 10),
                      (center_x - 15, head_y - head_radius - 15),
                      (center_x - 5, head_y - head_radius + 10)]
        ear_points2 = [(center_x + 5, head_y - head_radius + 10),
                      (center_x + 15, head_y - head_radius - 15),
                      (center_x + 25, head_y - head_radius + 10)]
        draw.polygon(ear_points1, fill=colors['primary'])
        draw.polygon(ear_points2, fill=colors['primary'])
        
        # Eyes
        draw.ellipse([center_x - 15, head_y - 10, center_x - 5, head_y], fill=(0, 0, 0))
        draw.ellipse([center_x + 5, head_y - 10, center_x + 15, head_y], fill=(0, 0, 0))
        
        # Nose
        draw.polygon([(center_x - 3, head_y + 5), (center_x + 3, head_y + 5), (center_x, head_y + 10)],
                    fill=colors['secondary'])
        
        # Tail
        draw.ellipse([center_x + body_width - 10, center_y - 60,
                     center_x + body_width + 30, center_y - 20],
                    fill=colors['primary'])
        
        return np.array(img)
    
    def _generate_fish(self, colors: Dict, size: Tuple[int, int]) -> np.ndarray:
        """Generate a realistic fish"""
        img = Image.new('RGB', size, (255, 255, 255))
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = size[0] // 2, size[1] // 2
        
        # Fish body
        body_width, body_height = 80, 40
        draw.ellipse([center_x - body_width, center_y - body_height,
                     center_x + body_width // 2, center_y + body_height],
                    fill=colors['primary'])
        
        # Tail
        tail_points = [(center_x + body_width // 2, center_y - body_height // 2),
                      (center_x + body_width, center_y - body_height),
                      (center_x + body_width + 20, center_y),
                      (center_x + body_width, center_y + body_height),
                      (center_x + body_width // 2, center_y + body_height // 2)]
        draw.polygon(tail_points, fill=colors['secondary'])
        
        # Eye
        draw.ellipse([center_x - body_width + 20, center_y - 10,
                     center_x - body_width + 35, center_y + 5],
                    fill=colors['highlight'])
        draw.ellipse([center_x - body_width + 25, center_y - 5,
                     center_x - body_width + 30, center_y],
                    fill=(0, 0, 0))
        
        # Fins
        fin_points = [(center_x - body_width + 40, center_y + body_height - 10),
                     (center_x - body_width + 30, center_y + body_height + 15),
                     (center_x - body_width + 50, center_y + body_height + 5)]
        draw.polygon(fin_points, fill=colors['secondary'])
        
        return np.array(img)
