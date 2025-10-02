#!/usr/bin/env python3
"""
Test script for the updated main.py with cross road detection.
This demonstrates how the pipeline now works with early exit for images without cross roads.
"""

import os
import sys
import json
from main import main

def test_updated_pipeline():
    """Test the updated pipeline with cross road detection"""
    
    # Check if we have sample images
    imgs_dir = "imgs"
    if not os.path.exists(imgs_dir):
        print("No 'imgs' directory found. Please provide camera and map images.")
        return
    
    # Look for sample images
    camera_images = []
    map_images = []
    
    for file in os.listdir(imgs_dir):
        if file.lower().startswith('view') and file.lower().endswith(('.png', '.jpg', '.jpeg')):
            camera_images.append(os.path.join(imgs_dir, file))
        elif file.lower().startswith('map') and file.lower().endswith(('.png', '.jpg', '.jpeg')):
            map_images.append(os.path.join(imgs_dir, file))
    
    if not camera_images or not map_images:
        print("No camera (view*) or map (map*) images found in the 'imgs' directory.")
        return
    
    print(f"Found {len(camera_images)} camera images and {len(map_images)} map images.")
    print("Testing updated pipeline with cross road detection...\n")
    
    # Test on first camera and map pair
    camera_path = camera_images[0]
    map_path = map_images[0]
    
    print(f"=== Testing with Camera: {os.path.basename(camera_path)} and Map: {os.path.basename(map_path)} ===")
    
    try:
        # Test with JSON output and fast mode
        print("Running pipeline with JSON output...")
        main(
            camera_path,
            map_path,
            fast_mode=True,
            json_output=True,
            max_tiles=2  # Limit tiles for faster testing
        )
        print()
        
        # Test with verbose output
        print("Running pipeline with verbose output...")
        main(
            camera_path,
            map_path,
            fast_mode=True,
            json_output=False,
            max_tiles=2  # Limit tiles for faster testing
        )
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_updated_pipeline()
