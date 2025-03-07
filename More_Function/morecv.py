import cv2
import numpy as np

## มีอีกวิธีนึงที่ใช้ K means clustering ในการลดสี แต่มันซับซ้อนกว่า
def convert_to_pixel_art(img, pixel_size=8, color_levels=4):
    height, width = img.shape[:2]
    # Calculate new dimensions
    small_width = width // pixel_size
    small_height = height // pixel_size
    
    # Resize image to small dimensions
    small_img = cv2.resize(img, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
    
    # Reduce color palette
    color_divisor = 256 // color_levels
    small_img = (small_img // color_divisor) * color_divisor
    
    # Scale back up
    pixel_art = cv2.resize(small_img, (width, height), interpolation=cv2.INTER_NEAREST)
    
    return pixel_art

def apply_cartoon_effect(img):
    """Apply cartoon effect to image"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur
    gray = cv2.medianBlur(gray, 5)
    
    # Detect edges
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY, 9, 9)
    
    # Apply bilateral filter for color smoothing
    color = cv2.bilateralFilter(img, 9, 300, 300)
    
    # Combine edges with color image
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    
    return cartoon

def apply_sketch_effect(img, blur_size=131, intensity=256.0, invert=True):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if blur_size % 2 == 0:
        blur_size += 1
    
    blur_size = max(3, blur_size)
    
    inverted_gray_img = 255 - gray
    
    blurred = cv2.GaussianBlur(inverted_gray_img, (blur_size, blur_size), 0)
    
    inverted_blurred = 255 - blurred
    
    sketch = cv2.divide(gray, inverted_blurred, scale=intensity)
    
    if invert:
        sketch = 255 - sketch
        
    sketch_color = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    return sketch_color