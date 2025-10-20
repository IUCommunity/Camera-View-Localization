#!/usr/bin/env python3
"""
Test the coordinate extraction function with various response formats
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from omlab_localization import extract_coordinates

def test_extraction():
    """Test coordinate extraction with various formats"""
    
    test_cases = [
        # Valid JSON
        '{"x": 123, "y": 456, "confidence": 0.85, "reason": "Intersection"}',
        
        # JSON with extra text
        'The camera position is {"x": 200, "y": 300, "confidence": 0.9, "reason": "Building match"}',
        
        # Coordinate format
        'Camera position: x: 150, y: 250',
        
        # Parentheses format
        'Position is (180, 320)',
        
        # Simple numbers
        'The coordinates are 100, 200',
        
        # Malformed JSON
        '{"x": 50, "y": 75, confidence: 0.7}',  # Missing quotes around confidence
        
        # No valid coordinates
        'I cannot determine the position',
    ]
    
    map_width, map_height = 500, 400
    
    print("Testing coordinate extraction:")
    print("=" * 50)
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_text}")
        result = extract_coordinates(test_text, map_width, map_height)
        print(f"Result: x={result['x']}, y={result['y']}, conf={result['confidence']:.2f}")
        print(f"Reason: {result['reason']}")

if __name__ == "__main__":
    test_extraction()
