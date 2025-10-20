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
import re
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Configuration
MODEL_ID = "omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results"
DEBUG = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model(use_safe_dtype: bool = False, force_cpu: bool = False):
    """Load the vision-language model.
    use_safe_dtype=True forces float32 to avoid potential CUDA asserts.
    """
    print(f"Loading model: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    # Choose dtype based on device and safety flag
    if (DEVICE == "cuda") and (not use_safe_dtype) and (not force_cpu):
        dtype = torch.float16
    else:
        dtype = torch.float32
    device_map = "auto" if not force_cpu else {"": "cpu"}
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device_map,
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
    
    # Simple, direct prompt with clearer instructions and without numeric examples
    prompt = f"""Analyze the street view image and find where the camera was positioned on the aerial map.

Instructions:
1. Look at the street view (first image) - identify buildings, roads, intersections
2. Find matching features in the aerial map (second image)
3. Determine the camera position coordinates

Map dimensions: {map_width} x {map_height} pixels
Valid coordinates: x=0 to {map_width-1}, y=0 to {map_height-1}

Return ONLY valid JSON with these keys and types:
{"x": <integer>, "y": <integer>, "confidence": <float 0..1>, "reason": "<short explanation>"}

Output only the JSON object and nothing else."""

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
                max_new_tokens=200,  # Increased for better JSON responses
                do_sample=True,      # Allow some sampling for better responses
                temperature=0.3,     # Slightly higher for variety
                top_p=0.9,          # Nucleus sampling
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Prevent repetition
            )
        
        response = processor.decode(output[0], skip_special_tokens=True)
        print(f"Raw response: {response}")
        
        # Clean response - try to keep only assistant's part
        lower_resp = response.lower()
        if "assistant" in lower_resp:
            parts = re.split(r"assistant\s*:?", response, flags=re.IGNORECASE)
            if parts:
                response = parts[-1]
        
        # Extract JSON from response
        result = extract_coordinates(response, map_width, map_height)
        print(f"Extracted result: {result}")
        return result
        
    except Exception as e:
        # Keep reason user-friendly; put detailed error separately
        error_msg = str(e).split('\n', 1)[0]
        print(f"Error during localization: {error_msg}")
        return {
            "x": map_width // 2,
            "y": map_height // 2,
            "confidence": 0.0,
            "reason": "Inference failed; returned fallback position",
            "error": error_msg,
        }

def extract_coordinates(text: str, map_width: int, map_height: int) -> Dict[str, Any]:
    """Extract coordinates from model response"""
    
    import re
    
    print(f"Extracting from text: {text[:200]}...")  # Debug output
    
    # Method 1: Try to find JSON with more flexible pattern
    json_patterns = [
        r'\{[^{}]*"x"[^{}]*"y"[^{}]*\}',  # Original pattern
        r'\{[^}]*"x"[^}]*"y"[^}]*\}',     # Simpler pattern
        r'\{"x":\s*\d+[^}]*"y":\s*\d+[^}]*\}',  # More specific
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        if matches:
            # Prefer the last JSON-like block, which is likely the model's answer
            for match in reversed(matches):
                try:
                    print(f"Trying to parse JSON: {match}")  # Debug output
                    result = json.loads(match)
                    
                    # Validate and clamp coordinates
                    x = max(0, min(int(result.get("x", map_width // 2)), map_width - 1))
                    y = max(0, min(int(result.get("y", map_height // 2)), map_height - 1))
                    confidence = max(0.0, min(float(result.get("confidence", 0.5)), 1.0))
                    reason = str(result.get("reason", result.get("reasoning", "No explanation provided")))
                    
                    print(f"Successfully parsed JSON: x={x}, y={y}, conf={confidence}")  # Debug output
                    return {
                        "x": x,
                        "y": y,
                        "confidence": confidence,
                        "reason": reason
                    }
                except Exception as e:
                    print(f"JSON parsing failed: {e}")  # Debug output
                    continue
    
    # Method 2: Try to find coordinates in various formats
    coord_patterns = [
        r'x[:\s=]*(\d+)[,\s]*y[:\s=]*(\d+)',  # x:123, y:456
        r'\((\d+),\s*(\d+)\)',                # (123, 456)
        r'(\d+),\s*(\d+)',                    # 123, 456
        r'position[:\s]*(\d+)[,\s]*(\d+)',    # position: 123, 456
    ]
    
    for pattern in coord_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                x = max(0, min(int(matches[0][0]), map_width - 1))
                y = max(0, min(int(matches[0][1]), map_height - 1))
                print(f"Extracted coordinates from pattern: x={x}, y={y}")  # Debug output
                return {
                    "x": x,
                    "y": y,
                    "confidence": 0.4,
                    "reason": "Extracted from coordinate pattern"
                }
            except:
                continue
    
    # Method 3: Try to extract any two numbers
    numbers = re.findall(r'\d+', text)
    if len(numbers) >= 2:
        try:
            x = max(0, min(int(numbers[0]), map_width - 1))
            y = max(0, min(int(numbers[1]), map_height - 1))
            print(f"Extracted first two numbers: x={x}, y={y}")  # Debug output
            return {
                "x": x,
                "y": y,
                "confidence": 0.2,
                "reason": "Extracted from numbers (very low confidence)"
            }
        except:
            pass
    
    # Final fallback
    print("Using fallback coordinates")  # Debug output
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
    parser.add_argument("--safe", action="store_true", help="Use safe dtype (float32) to avoid CUDA asserts")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SIMPLE GEO-LOCALIZATION")
    print("=" * 60)
    
    # Load model
    start_time = time.time()
    processor, model = load_model(use_safe_dtype=args.safe)
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
    success = "error" not in result
    output = {
        "success": success,
        "position": {"x": result["x"], "y": result["y"]},
        "confidence": result["confidence"],
        "reason": result["reason"],
        "timing": {
            "model_load": round(model_time, 2),
            "inference": round(inference_time, 2),
            "total": round(model_time + inference_time, 2)
        }
    }
    if not success and "error" in result:
        output["error"] = result["error"]
    
    print(f"\nJSON Output:")
    print(json.dumps(output, indent=2))
    
    return output

if __name__ == "__main__":
    main()
