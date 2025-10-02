#!/usr/bin/env python3
"""
Test script for cross road detection model.
This demonstrates how to use the cross_road_detection.py module.
"""

import os
import sys
from cross_road_detection import main

def test_cross_road_detection():
    """Test the cross road detection with sample images"""
    
    # Check if we have any sample images in the imgs directory
    imgs_dir = "imgs"
    if not os.path.exists(imgs_dir):
        print("No 'imgs' directory found. Please provide an image path.")
        return
    
    # Look for sample images
    sample_images = []
    for file in os.listdir(imgs_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            sample_images.append(os.path.join(imgs_dir, file))
    
    if not sample_images:
        print("No images found in the 'imgs' directory.")
        return
    
    print(f"Found {len(sample_images)} sample images.")
    print("Testing cross road detection on first few images...\n")
    
    # Test on first 3 images
    for i, image_path in enumerate(sample_images[:3]):
        print(f"=== Testing Image {i+1}: {os.path.basename(image_path)} ===")
        try:
            # Test with fast mode for quick results
            main(image_path, fast_mode=True, json_output=False)
            print()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            print()

if __name__ == "__main__":
    test_cross_road_detection()
