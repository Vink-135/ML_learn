"""
Enhanced image processing module for Pix2Pix application
Provides color enhancement, background removal, and realistic image generation
"""

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
try:
    from skimage import exposure, filters
    from skimage.segmentation import flood_fill
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
import tensorflow as tf
from object_recognizer import ObjectRecognizer
from object_generator import ObjectGenerator

class ImageEnhancer:
    def __init__(self):
        self.background_removal_enabled = True
        self.object_recognizer = ObjectRecognizer()
        self.object_generator = ObjectGenerator()
        
    def enhance_colors(self, image_array):
        """Enhance colors to make them more vibrant and realistic"""
        # Convert to PIL for easier manipulation
        if isinstance(image_array, np.ndarray):
            image = Image.fromarray(image_array.astype('uint8'))
        else:
            image = image_array
            
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.3)
        
        # Enhance color saturation
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.4)
        
        # Enhance brightness slightly
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        
        # Apply slight sharpening
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        
        return np.array(image)
    
    def remove_gray_background(self, image_array, threshold=200):
        """Remove gray/uniform background and replace with white or transparent"""
        img = image_array.copy()

        # Ensure input is uint8
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Convert to grayscale to detect uniform areas
        if CV2_AVAILABLE:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            # Manual RGB to grayscale conversion
            gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

        # Create mask for gray/uniform areas
        mask = np.abs(gray - np.mean(gray)) < 30  # Detect uniform areas

        # Also detect areas that are close to gray (128)
        gray_mask = np.abs(gray - 128) < 50

        # Combine masks
        background_mask = mask | gray_mask

        # Create alpha channel version
        img_rgba = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        img_rgba[:,:,:3] = img
        img_rgba[:,:,3] = 255  # Full opacity

        # Make background transparent
        img_rgba[background_mask, 3] = 0

        # For non-transparent version, make background white
        img_white_bg = img.copy()
        img_white_bg[background_mask] = [255, 255, 255]

        return img_rgba, img_white_bg
    
    def apply_realistic_color_mapping(self, image_array):
        """Apply color mapping to make images look more realistic"""
        img = image_array.astype(np.float32) / 255.0
        
        # Apply gamma correction for more natural look
        img = np.power(img, 0.8)
        
        # Enhance specific color channels
        # Boost reds and oranges (skin tones, warm colors)
        red_boost = np.where(img[:,:,0] > img[:,:,1], 1.2, 1.0)
        img[:,:,0] *= red_boost
        
        # Enhance blues and greens (sky, nature)
        blue_green_boost = np.where((img[:,:,1] > 0.3) | (img[:,:,2] > 0.3), 1.1, 1.0)
        img[:,:,1] *= blue_green_boost
        img[:,:,2] *= blue_green_boost
        
        # Clip values and convert back
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        return img
    
    def create_intelligent_object(self, drawing_array):
        """Create a realistic object based on intelligent recognition"""
        try:
            # Recognize what object is drawn
            object_name, confidence = self.object_recognizer.recognize_object(drawing_array)

            print(f"Recognized object: {object_name} (confidence: {confidence:.2f})")

            # Get appropriate colors for the object
            colors = self.object_recognizer.get_object_colors(object_name)

            # Generate realistic object
            realistic_image = self.object_generator.generate_realistic_object(
                object_name, colors, (drawing_array.shape[1], drawing_array.shape[0])
            )

            return realistic_image, object_name, confidence

        except Exception as e:
            print(f"Error in intelligent object creation: {e}")
            # Fallback to original method
            return self.create_realistic_fallback(drawing_array), "unknown", 0.0

    def create_realistic_fallback(self, drawing_array):
        """Create a more realistic image from drawing when no trained model is available"""
        # Ensure input is uint8
        if drawing_array.dtype != np.uint8:
            drawing_array = np.clip(drawing_array, 0, 255).astype(np.uint8)

        # Convert drawing to edges
        if CV2_AVAILABLE:
            gray = cv2.cvtColor(drawing_array, cv2.COLOR_RGB2GRAY)
            # Ensure gray is uint8 for Canny
            if gray.dtype != np.uint8:
                gray = np.clip(gray, 0, 255).astype(np.uint8)
            edges = cv2.Canny(gray, 50, 150)
        else:
            # Manual edge detection
            gray = np.dot(drawing_array[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            # Simple edge detection using gradients
            gy, gx = np.gradient(gray.astype(float))
            edges = (np.sqrt(gx**2 + gy**2) > 30).astype(np.uint8) * 255

        # Create a base realistic image with gradients and textures
        height, width = gray.shape

        # Create gradient background
        y, x = np.ogrid[:height, :width]
        gradient = np.sin(x * 0.01) * np.cos(y * 0.01) * 50 + 128

        # Add some texture
        noise = np.random.normal(0, 10, (height, width))
        textured_bg = gradient + noise

        # Create colored version based on drawing
        result = np.zeros((height, width, 3), dtype=np.uint8)

        # Where there are edges, create more defined colors
        edge_mask = edges > 0

        # Assign colors based on position and drawing
        for i in range(3):  # RGB channels
            channel = textured_bg.copy()

            # Add color variation based on drawing
            if i == 0:  # Red channel
                channel[edge_mask] += 50
            elif i == 1:  # Green channel
                channel[edge_mask] += 30
            else:  # Blue channel
                channel[edge_mask] += 20

            result[:,:,i] = np.clip(channel, 0, 255)

        return result
    
    def process_generated_image(self, generated_image, original_drawing=None,
                              background_option="white", enhance_colors=True):
        """
        Complete processing pipeline for generated images

        Args:
            generated_image: The raw generated image from the model
            original_drawing: The original drawing (optional, for fallback)
            background_option: "white", "transparent", or "original"
            enhance_colors: Whether to apply color enhancement
        """

        try:
            # Ensure generated_image is uint8
            if generated_image.dtype != np.uint8:
                generated_image = np.clip(generated_image, 0, 255).astype(np.uint8)

            # If the image is mostly gray/uniform, use intelligent object generation
            if self._is_mostly_gray(generated_image):
                if original_drawing is not None:
                    # Use intelligent object recognition and generation
                    generated_image, object_name, confidence = self.create_intelligent_object(original_drawing)
                    print(f"Generated realistic {object_name} with confidence {confidence:.2f}")
                else:
                    # Create a simple colorful pattern
                    generated_image = self._create_colorful_pattern(generated_image.shape)

            # Ensure we still have uint8 after fallback
            if generated_image.dtype != np.uint8:
                generated_image = np.clip(generated_image, 0, 255).astype(np.uint8)

            # Apply color enhancements
            if enhance_colors:
                generated_image = self.apply_realistic_color_mapping(generated_image)
                generated_image = self.enhance_colors(generated_image)

            # Final data type check
            if generated_image.dtype != np.uint8:
                generated_image = np.clip(generated_image, 0, 255).astype(np.uint8)

            # Handle background
            if background_option == "transparent":
                img_rgba, _ = self.remove_gray_background(generated_image)
                return Image.fromarray(img_rgba, 'RGBA')
            elif background_option == "white":
                _, img_white = self.remove_gray_background(generated_image)
                return Image.fromarray(img_white, 'RGB')
            else:
                return Image.fromarray(generated_image, 'RGB')

        except Exception as e:
            # Fallback to simple processing if anything fails
            print(f"Error in image processing: {e}")
            if generated_image.dtype != np.uint8:
                generated_image = np.clip(generated_image, 0, 255).astype(np.uint8)
            return Image.fromarray(generated_image, 'RGB')
    
    def _is_mostly_gray(self, image_array, threshold=0.8):
        """Check if image is mostly gray/uniform"""
        # Ensure input is uint8
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        if CV2_AVAILABLE:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        std_dev = np.std(gray)
        return std_dev < 20  # Very low variation indicates uniform gray
    
    def _create_colorful_pattern(self, shape):
        """Create a colorful pattern as fallback"""
        height, width = shape[:2]
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create colorful gradient
        for i in range(height):
            for j in range(width):
                pattern[i, j, 0] = int(255 * np.sin(i * 0.02) ** 2)  # Red
                pattern[i, j, 1] = int(255 * np.sin(j * 0.02) ** 2)  # Green  
                pattern[i, j, 2] = int(255 * np.sin((i+j) * 0.01) ** 2)  # Blue
                
        return pattern
