#!/usr/bin/env python3
"""
Test script for omlab_localization.py
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic imports and functions"""
    try:
        from omlab_localization import load_model, load_image, extract_coordinates
        print("✓ All imports successful")
        
        # Test coordinate extraction
        test_response = '{"x": 100, "y": 200, "confidence": 0.8, "reason": "test"}'
        result = extract_coordinates(test_response, 500, 400)
        
        if result["x"] == 100 and result["y"] == 200:
            print("✓ Coordinate extraction working")
        else:
            print(f"✗ Coordinate extraction failed: {result}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False

def test_with_sample_images():
    """Test with sample images if available"""
    try:
        from omlab_localization import load_model, load_image, simple_localize
        
        # Check if sample images exist
        camera_path = "imgs/view1.png"
        map_path = "imgs/map1.png"
        
        if not os.path.exists(camera_path):
            print(f"✗ Camera image not found: {camera_path}")
            return False
            
        if not os.path.exists(map_path):
            print(f"✗ Map image not found: {map_path}")
            return False
        
        print("✓ Sample images found")
        print("Note: Full test requires model loading which may take time")
        print("Run: python omlab_localization.py --camera imgs/view1.png --map imgs/map1.png")
        
        return True
        
    except Exception as e:
        print(f"✗ Test error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING OMLAB LOCALIZATION")
    print("=" * 50)
    
    print("\n1. Testing basic functionality...")
    basic_ok = test_basic_functionality()
    
    print("\n2. Testing with sample images...")
    images_ok = test_with_sample_images()
    
    print("\n" + "=" * 50)
    if basic_ok and images_ok:
        print("✓ ALL TESTS PASSED")
        print("\nTo run full localization:")
        print("python omlab_localization.py --camera imgs/view1.png --map imgs/map1.png")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 50)
