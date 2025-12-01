"""
Universal Template Generator for Any Object Type

This module creates simple drawing templates for various object categories
to help users get started with the universal Pix2Pix system.
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw

class UniversalTemplateGenerator:
    """Generate simple drawing templates for any object category"""
    
    def __init__(self):
        self.templates = {
            "🍎 Apple": self.create_apple_template,
            "🚗 Car": self.create_car_template,
            "🏠 House": self.create_house_template,
            "🐱 Cat": self.create_cat_template,
            "🌳 Tree": self.create_tree_template,
            "👤 Person": self.create_person_template
        }
    
    def create_template(self, template_name, width=400, height=400):
        """Create a template for the specified object"""
        if template_name in self.templates:
            return self.templates[template_name](width, height)
        else:
            return self.create_blank_template(width, height)
    
    def create_blank_template(self, width, height):
        """Create a blank white template"""
        return np.ones((height, width, 3), dtype=np.uint8) * 255
    
    def create_apple_template(self, width=400, height=400):
        """Create a simple apple template"""
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Apple body (red circle)
        center_x, center_y = width // 2, height // 2 + 20
        radius = min(width, height) // 3
        
        # Main apple shape (slightly oval)
        draw.ellipse([center_x - radius, center_y - radius * 0.9, 
                     center_x + radius, center_y + radius * 1.1], 
                    fill='red', outline='darkred', width=2)
        
        # Apple indent at top
        draw.ellipse([center_x - radius * 0.3, center_y - radius * 0.9, 
                     center_x + radius * 0.3, center_y - radius * 0.5], 
                    fill='white', outline='darkred', width=1)
        
        # Stem (brown)
        draw.rectangle([center_x - 3, center_y - radius * 0.9 - 15, 
                       center_x + 3, center_y - radius * 0.9], 
                      fill='brown')
        
        # Leaf (green)
        draw.ellipse([center_x + 5, center_y - radius * 0.9 - 10, 
                     center_x + 20, center_y - radius * 0.9 + 5], 
                    fill='green')
        
        return np.array(img)
    
    def create_car_template(self, width=400, height=400):
        """Create a simple car template"""
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = width // 2, height // 2
        car_width = width * 0.6
        car_height = height * 0.3
        
        # Car body (blue rectangle)
        draw.rectangle([center_x - car_width//2, center_y - car_height//2,
                       center_x + car_width//2, center_y + car_height//2],
                      fill='blue', outline='darkblue', width=2)
        
        # Car roof (smaller rectangle)
        roof_width = car_width * 0.6
        roof_height = car_height * 0.6
        draw.rectangle([center_x - roof_width//2, center_y - car_height//2 - roof_height,
                       center_x + roof_width//2, center_y - car_height//2],
                      fill='lightblue', outline='darkblue', width=2)
        
        # Wheels (black circles)
        wheel_radius = car_height * 0.3
        wheel_y = center_y + car_height//2 + wheel_radius//2
        
        # Left wheel
        draw.ellipse([center_x - car_width//3 - wheel_radius, wheel_y - wheel_radius,
                     center_x - car_width//3 + wheel_radius, wheel_y + wheel_radius],
                    fill='black', outline='gray', width=2)
        
        # Right wheel
        draw.ellipse([center_x + car_width//3 - wheel_radius, wheel_y - wheel_radius,
                     center_x + car_width//3 + wheel_radius, wheel_y + wheel_radius],
                    fill='black', outline='gray', width=2)
        
        # Windows (light blue)
        window_margin = 10
        draw.rectangle([center_x - roof_width//2 + window_margin, 
                       center_y - car_height//2 - roof_height + window_margin,
                       center_x + roof_width//2 - window_margin, 
                       center_y - car_height//2 - window_margin],
                      fill='lightcyan', outline='darkblue', width=1)
        
        return np.array(img)
    
    def create_house_template(self, width=400, height=400):
        """Create a simple house template"""
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = width // 2, height // 2
        house_width = width * 0.5
        house_height = height * 0.4
        
        # House walls (beige)
        draw.rectangle([center_x - house_width//2, center_y,
                       center_x + house_width//2, center_y + house_height],
                      fill='wheat', outline='brown', width=2)
        
        # Roof (red triangle)
        roof_points = [
            (center_x - house_width//2 - 20, center_y),
            (center_x + house_width//2 + 20, center_y),
            (center_x, center_y - house_height//2)
        ]
        draw.polygon(roof_points, fill='red', outline='darkred', width=2)
        
        # Door (brown)
        door_width = house_width * 0.2
        door_height = house_height * 0.6
        draw.rectangle([center_x - door_width//2, center_y + house_height - door_height,
                       center_x + door_width//2, center_y + house_height],
                      fill='brown', outline='darkred', width=2)
        
        # Windows (blue)
        window_size = house_width * 0.15
        # Left window
        draw.rectangle([center_x - house_width//2 + 30, center_y + 30,
                       center_x - house_width//2 + 30 + window_size, center_y + 30 + window_size],
                      fill='lightblue', outline='blue', width=2)
        
        # Right window
        draw.rectangle([center_x + house_width//2 - 30 - window_size, center_y + 30,
                       center_x + house_width//2 - 30, center_y + 30 + window_size],
                      fill='lightblue', outline='blue', width=2)
        
        return np.array(img)
    
    def create_cat_template(self, width=400, height=400):
        """Create a simple cat template"""
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = width // 2, height // 2
        
        # Cat body (gray oval)
        body_width = width * 0.3
        body_height = height * 0.4
        draw.ellipse([center_x - body_width//2, center_y,
                     center_x + body_width//2, center_y + body_height],
                    fill='lightgray', outline='gray', width=2)
        
        # Cat head (gray circle)
        head_radius = body_width * 0.6
        head_y = center_y - head_radius * 0.3
        draw.ellipse([center_x - head_radius//2, head_y - head_radius//2,
                     center_x + head_radius//2, head_y + head_radius//2],
                    fill='lightgray', outline='gray', width=2)
        
        # Ears (triangles)
        ear_size = head_radius * 0.4
        # Left ear
        draw.polygon([
            (center_x - head_radius//3, head_y - head_radius//2),
            (center_x - head_radius//3 - ear_size//2, head_y - head_radius//2 - ear_size),
            (center_x - head_radius//3 + ear_size//2, head_y - head_radius//2 - ear_size)
        ], fill='lightgray', outline='gray', width=2)
        
        # Right ear
        draw.polygon([
            (center_x + head_radius//3, head_y - head_radius//2),
            (center_x + head_radius//3 - ear_size//2, head_y - head_radius//2 - ear_size),
            (center_x + head_radius//3 + ear_size//2, head_y - head_radius//2 - ear_size)
        ], fill='lightgray', outline='gray', width=2)
        
        # Eyes (black dots)
        eye_size = 8
        draw.ellipse([center_x - 20 - eye_size//2, head_y - 10 - eye_size//2,
                     center_x - 20 + eye_size//2, head_y - 10 + eye_size//2],
                    fill='black')
        draw.ellipse([center_x + 20 - eye_size//2, head_y - 10 - eye_size//2,
                     center_x + 20 + eye_size//2, head_y - 10 + eye_size//2],
                    fill='black')
        
        # Nose (pink triangle)
        draw.polygon([
            (center_x, head_y + 5),
            (center_x - 5, head_y + 15),
            (center_x + 5, head_y + 15)
        ], fill='pink')
        
        # Tail (curved line)
        tail_points = [
            (center_x + body_width//2, center_y + body_height//2),
            (center_x + body_width//2 + 30, center_y + body_height//2 - 20),
            (center_x + body_width//2 + 40, center_y + body_height//2 - 50)
        ]
        for i in range(len(tail_points) - 1):
            draw.line([tail_points[i], tail_points[i+1]], fill='gray', width=8)
        
        return np.array(img)
    
    def create_tree_template(self, width=400, height=400):
        """Create a simple tree template"""
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = width // 2, height // 2
        
        # Tree trunk (brown rectangle)
        trunk_width = width * 0.08
        trunk_height = height * 0.3
        draw.rectangle([center_x - trunk_width//2, center_y + trunk_height//2,
                       center_x + trunk_width//2, center_y + trunk_height//2 + trunk_height],
                      fill='brown', outline='darkred', width=2)
        
        # Tree crown (green circle)
        crown_radius = width * 0.25
        draw.ellipse([center_x - crown_radius, center_y - crown_radius,
                     center_x + crown_radius, center_y + crown_radius],
                    fill='green', outline='darkgreen', width=2)
        
        return np.array(img)
    
    def create_person_template(self, width=400, height=400):
        """Create a simple person template"""
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        center_x, center_y = width // 2, height // 2
        
        # Head (peach circle)
        head_radius = width * 0.08
        head_y = center_y - height * 0.25
        draw.ellipse([center_x - head_radius, head_y - head_radius,
                     center_x + head_radius, head_y + head_radius],
                    fill='peachpuff', outline='tan', width=2)
        
        # Body (blue rectangle)
        body_width = width * 0.12
        body_height = height * 0.25
        draw.rectangle([center_x - body_width//2, head_y + head_radius,
                       center_x + body_width//2, head_y + head_radius + body_height],
                      fill='lightblue', outline='blue', width=2)
        
        # Arms (lines)
        arm_length = width * 0.1
        arm_y = head_y + head_radius + body_height * 0.3
        draw.line([center_x - body_width//2, arm_y, center_x - body_width//2 - arm_length, arm_y],
                  fill='peachpuff', width=6)
        draw.line([center_x + body_width//2, arm_y, center_x + body_width//2 + arm_length, arm_y],
                  fill='peachpuff', width=6)
        
        # Legs (lines)
        leg_length = height * 0.15
        leg_y = head_y + head_radius + body_height
        draw.line([center_x - body_width//4, leg_y, center_x - body_width//4, leg_y + leg_length],
                  fill='blue', width=6)
        draw.line([center_x + body_width//4, leg_y, center_x + body_width//4, leg_y + leg_length],
                  fill='blue', width=6)
        
        return np.array(img)

# Global instance
universal_template_generator = UniversalTemplateGenerator()
