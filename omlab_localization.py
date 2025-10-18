#!/usr/bin/env python3
"""
Simple Geo-Localization using VLM-R1-Qwen2.5VL-3B-OVD-0321
A simplified approach for camera position localization on aerial maps.
"""

import os
import json
import argparse
import time
from typing import Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Configuration
MODEL_ID = "omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results"
DEBUG = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    """Load the vision-language model"""
    print(f"Loading model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    print("✓ Model loaded successfully")
    return processor, model

def load_image(path: str, max_size: int = 1024) -> Image.Image:
    """Load and resize image if needed"""
    img = Image.open(path).convert("RGB")
    if max_size and (img.width > max_size or img.height > max_size):
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img

def simple_localize(processor, model, camera_img: Image.Image, map_img: Image.Image) -> Dict[str, Any]:
    """
    Simple localization with basic prompting
    """
    map_width, map_height = map_img.size
    
    # Simple, direct prompt
    prompt = f"""Find the camera position on this aerial map. The camera took a street-level photo. 
    
Look at the street view image and match it to features in the aerial map.
Return the pixel coordinates where the camera was positioned.

Map size: {map_width} x {map_height} pixels
Return format: {{"x": pixel_x, "y": pixel_y, "confidence": 0.0-1.0, "reason": "explanation"}}

Coordinates must be between 0 and {map_width-1} for x, and 0 and {map_height-1} for y."""

    chat = [
        {
            "role": "system", 
            "content": "You are a geo-localization expert. Find camera positions on maps."
        },
        {
            "role": "user", 
            "content": [
                {"type": "image"},  # camera image
                {"type": "image"},  # map image  
                {"type": "text", "text": prompt},
            ]
        },
    ]
    
    try:
        print("Processing images and generating response...")
        
        chat_str = processor.apply_chat_template(chat, add_generation_prompt=True)
        inputs = processor(
            text=chat_str, 
            images=[camera_img, map_img], 
            return_tensors="pt"
        ).to(model.device)
        
        # Simple generation parameters to avoid infinite loops
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=150,  # Limit tokens to prevent infinite generation
                do_sample=False,     # Deterministic generation
                temperature=0.1,     # Low temperature for consistency
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        response = processor.decode(output[0], skip_special_tokens=True)
        print(f"Raw response: {response}")
        
        # Extract JSON from response
        result = extract_coordinates(response, map_width, map_height)
        return result
        
    except Exception as e:
        print(f"Error during localization: {e}")
        return {
            "x": map_width // 2,
            "y": map_height // 2,
            "confidence": 0.0,
            "reason": f"Error: {str(e)}"
        }

def extract_coordinates(text: str, map_width: int, map_height: int) -> Dict[str, Any]:
    """Extract coordinates from model response"""
    
    # Try to find JSON in the response
    import re
    
    # Look for JSON pattern
    json_pattern = r'\{[^{}]*"x"[^{}]*"y"[^{}]*\}'
    matches = re.findall(json_pattern, text, re.IGNORECASE)
    
    if matches:
        try:
            result = json.loads(matches[0])
            
            # Validate and clamp coordinates
            x = max(0, min(int(result.get("x", map_width // 2)), map_width - 1))
            y = max(0, min(int(result.get("y", map_height // 2)), map_height - 1))
            confidence = max(0.0, min(float(result.get("confidence", 0.5)), 1.0))
            reason = str(result.get("reason", result.get("reasoning", "No explanation provided")))
            
            return {
                "x": x,
                "y": y,
                "confidence": confidence,
                "reason": reason
            }
        except:
            pass
    
    # Fallback: try to extract numbers from text
    numbers = re.findall(r'\d+', text)
    if len(numbers) >= 2:
        try:
            x = max(0, min(int(numbers[0]), map_width - 1))
            y = max(0, min(int(numbers[1]), map_height - 1))
            return {
                "x": x,
                "y": y,
                "confidence": 0.3,
                "reason": "Extracted from text (low confidence)"
            }
        except:
            pass
    
    # Final fallback
    return {
        "x": map_width // 2,
        "y": map_height // 2,
        "confidence": 0.0,
        "reason": "Could not extract coordinates from response"
    }

def visualize_result(map_img: Image.Image, result: Dict[str, Any], camera_name: str) -> str:
    """Create simple visualization"""
    
    img = map_img.copy()
    draw = ImageDraw.Draw(img)
    
    x, y = result["x"], result["y"]
    confidence = result["confidence"]
    
    # Draw marker
    marker_size = 15
    # Outer circle (white)
    draw.ellipse(
        [x - marker_size, y - marker_size, x + marker_size, y + marker_size],
        fill=(255, 255, 255),
        outline=(0, 0, 0),
        width=3
    )
    # Inner circle (red)
    draw.ellipse(
        [x - marker_size//2, y - marker_size//2, x + marker_size//2, y + marker_size//2],
        fill=(255, 0, 0)
    )
    
    # Add confidence circle
    radius = int(20 * (1.1 - confidence))
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        outline=(255, 255, 0),
        width=2
    )
    
    # Add label
    label = f"({x}, {y})\nConf: {confidence:.2f}"
    label_x = max(10, min(x + 20, img.width - 100))
    label_y = max(10, min(y - 30, img.height - 50))
    
    # Background for text
    draw.rectangle(
        [label_x - 5, label_y - 5, label_x + 80, label_y + 30],
        fill=(0, 0, 0, 180)
    )
    draw.text((label_x, label_y), label, fill=(255, 255, 255))
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, f"simple-localization-{camera_name}.png")
    img.save(save_path)
    return save_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple geo-localization")
    parser.add_argument("--camera", required=True, help="Path to camera image")
    parser.add_argument("--map", required=True, help="Path to map image")
    parser.add_argument("--max-size", type=int, default=1024, help="Max image size")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SIMPLE GEO-LOCALIZATION")
    print("=" * 60)
    
    # Load model
    start_time = time.time()
    processor, model = load_model()
    model_time = time.time() - start_time
    print(f"Model loading time: {model_time:.2f}s")
    
    # Load images
    print(f"\nLoading images...")
    camera_img = load_image(args.camera, args.max_size)
    map_img = load_image(args.map, args.max_size)
    print(f"Camera image: {camera_img.size}")
    print(f"Map image: {map_img.size}")
    
    # Localize
    print(f"\nPerforming localization...")
    start_time = time.time()
    result = simple_localize(processor, model, camera_img, map_img)
    inference_time = time.time() - start_time
    
    print(f"Localization time: {inference_time:.2f}s")
    print(f"\nResult:")
    print(f"Position: ({result['x']}, {result['y']})")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Reason: {result['reason']}")
    
    # Visualize
    if not args.no_viz:
        camera_name = os.path.splitext(os.path.basename(args.camera))[0]
        viz_path = visualize_result(map_img, result, camera_name)
        print(f"\nVisualization saved: {viz_path}")
    
    # Output JSON
    output = {
        "success": True,
        "position": {"x": result["x"], "y": result["y"]},
        "confidence": result["confidence"],
        "reason": result["reason"],
        "timing": {
            "model_load": round(model_time, 2),
            "inference": round(inference_time, 2),
            "total": round(model_time + inference_time, 2)
        }
    }
    
    print(f"\nJSON Output:")
    print(json.dumps(output, indent=2))
    
    return output

if __name__ == "__main__":
    main()
