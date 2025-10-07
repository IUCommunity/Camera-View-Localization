# Project: Single-Shot Camera Position Localization
# ---------------------------------------------------
# This approach uses direct pixel prediction instead of bounding boxes
# and processes the entire map in one inference for speed and accuracy.

import os
import json
import re
import argparse
import time
from typing import Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# -------------------------
# Configuration
# -------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results"
DEBUG = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Model Loader
# -------------------------

def load_model():
    """Load the vision-language model"""
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return processor, model

# -------------------------
# Image Utilities
# -------------------------

def load_image(path: str, max_size: int = 2048) -> Image.Image:
    """Load and optionally resize image"""
    img = Image.open(path).convert("RGB")
    if max_size and (img.width > max_size or img.height > max_size):
        # Preserve aspect ratio
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img

# -------------------------
# JSON Extraction Helper
# -------------------------

def extract_json(resp: str) -> Dict[str, Any]:
    """Extract JSON from model response with robust parsing"""
    # Remove assistant prefix if present
    if "assistant" in resp.lower():
        parts = resp.split("assistant", 1)
        if len(parts) > 1:
            resp = parts[-1]
    
    # Try to find JSON in code blocks
    matches = re.findall(r"```(?:json)?\s*(.*?)```", resp, re.DOTALL)
    if matches:
        candidate = matches[-1].strip()
    else:
        # Find JSON object
        matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", resp, re.DOTALL)
        if not matches:
            return {}
        candidate = matches[-1]
    
    try:
        return json.loads(candidate)
    except:
        # Try to fix common issues
        candidate = candidate.strip()
        if candidate.endswith(","):
            candidate = candidate[:-1]
        # Balance braces
        open_braces = candidate.count("{")
        close_braces = candidate.count("}")
        if open_braces > close_braces:
            candidate += "}" * (open_braces - close_braces)
        try:
            return json.loads(candidate)
        except Exception as e:
            if DEBUG:
                print(f"⚠️ JSON parsing failed: {e}")
                print(f"Response: {resp[:500]}")
            return {}

# -------------------------
# Core Localization Function
# -------------------------

def localize_camera_position(
    processor,
    model,
    camera_img: Image.Image,
    map_img: Image.Image,
    *,
    num_samples: int = 5,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """
    Directly predict camera position as (x, y) pixel coordinates on the map.
    Uses multiple samples and voting/averaging for robustness.
    """
    
    map_width, map_height = map_img.size
    
    # Improved prompt for direct position prediction
    prompt = f"""You are an expert at visual localization and spatial reasoning.

TASK: Determine the exact pixel position on the aerial MAP where the ground-level CAMERA was located when the photo was taken.

INPUTS:
1. CAMERA: Ground-level street view photograph
2. MAP: Aerial/satellite view (size: {map_width}x{map_height} pixels)

ANALYSIS APPROACH:
1. Examine the CAMERA view and identify key structural features:
   - Road layout, intersections, T-junctions, crossroads
   - Building shapes, walls, fences, gates
   - Sidewalk patterns, crosswalks, road markings
   - Permanent landmarks (NOT temporary objects like cars/people)
   - Viewing direction and perspective angles

2. Analyze spatial relationships:
   - Relative positions of buildings and roads
   - Intersection geometry and configuration
   - Building footprint shapes and arrangements
   - Distance cues from perspective (closer objects appear larger)

3. Match these features to the MAP view:
   - Find the corresponding road intersection or segment
   - Identify matching building footprints and boundaries
   - Consider the viewing angle and perspective

4. Determine the camera's pixel coordinates on the map

OUTPUT FORMAT - Return ONLY valid JSON with this EXACT structure:
{{
    "x": 512,
    "y": 384,
    "confidence": 0.85,
    "reasoning": "T-junction with building on northeast corner matches map"
}}

CRITICAL REQUIREMENTS:
- "x" must be an INTEGER between 0 and {map_width-1}
- "y" must be an INTEGER between 0 and {map_height-1}
- "confidence" must be a FLOAT/NUMBER between 0.0 and 1.0 (NOT text!)
- "reasoning" is a short text string explaining the match

NO explanations outside JSON. NO markdown. JUST the JSON object."""

    predictions = []
    
    for i in range(num_samples):
        chat = [
            {
                "role": "system", 
                "content": "You are a precise visual localization system. Output only valid JSON."
            },
            {
                "role": "user", 
                "content": [
                    {"type": "image"},  # camera
                    {"type": "image"},  # map
                    {"type": "text", "text": prompt},
                ]
            },
        ]
        
        chat_str = processor.apply_chat_template(chat, add_generation_prompt=True)
        inputs = processor(
            text=chat_str, 
            images=[camera_img, map_img], 
            return_tensors="pt"
        ).to(model.device)
        
        # Generation parameters
        gen_kwargs = {
            "max_new_tokens": 300,
            "do_sample": True if temperature > 0 else False,
        }
        if temperature > 0:
            gen_kwargs.update({
                "temperature": max(0.01, temperature),
                "top_p": 0.95,
            })
        
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        
        response = processor.decode(output[0], skip_special_tokens=True)
        result = extract_json(response)
        
        if DEBUG and i == 0:
            print(f"Sample response: {response[-500:]}")
        
        # Validate result
        if isinstance(result, dict) and "x" in result and "y" in result:
            try:
                x = int(result["x"])
                y = int(result["y"])
                
                # Handle swapped confidence/reasoning fields
                conf_val = result.get("confidence", 0.5)
                reasoning_val = result.get("reasoning", "")
                
                # Check if confidence and reasoning are swapped
                if isinstance(conf_val, str) and not isinstance(reasoning_val, str):
                    # Swap them
                    conf_val, reasoning_val = reasoning_val, conf_val
                    if DEBUG:
                        print(f"  Sample {i+1}: Auto-corrected swapped confidence/reasoning fields")
                
                # Try to convert confidence to float
                if isinstance(conf_val, str):
                    # Try to extract a number from the string
                    import re
                    numbers = re.findall(r'0?\.\d+|\d+\.?\d*', conf_val)
                    if numbers:
                        conf = float(numbers[0])
                        conf = max(0.0, min(1.0, conf))  # Clamp to [0, 1]
                    else:
                        if DEBUG:
                            print(f"  Sample {i+1}: Could not extract confidence number, using 0.5")
                        conf = 0.5
                else:
                    conf = float(conf_val)
                    conf = max(0.0, min(1.0, conf))  # Clamp to [0, 1]
                
                reasoning = str(reasoning_val) if reasoning_val else ""
                
                # Clamp coordinates to valid range
                x = max(0, min(x, map_width - 1))
                y = max(0, min(y, map_height - 1))
                
                predictions.append({
                    "x": x,
                    "y": y,
                    "confidence": conf,
                    "reasoning": reasoning
                })
                
                if DEBUG:
                    print(f"  Sample {i+1}: x={x}, y={y}, conf={conf:.3f}")
                    
            except (ValueError, TypeError) as e:
                if DEBUG:
                    print(f"  Sample {i+1}: Invalid format - {e}")
                    print(f"  Raw result: {result}")
                continue
    
    # Aggregate predictions
    if not predictions:
        # Fallback to center
        return {
            "x": map_width // 2,
            "y": map_height // 2,
            "confidence": 0.0,
            "reasoning": "No valid predictions generated",
            "method": "fallback"
        }
    
    # Use weighted average based on confidence
    total_weight = sum(p["confidence"] for p in predictions)
    if total_weight > 0:
        avg_x = sum(p["x"] * p["confidence"] for p in predictions) / total_weight
        avg_y = sum(p["y"] * p["confidence"] for p in predictions) / total_weight
        avg_conf = sum(p["confidence"] for p in predictions) / len(predictions)
    else:
        # Unweighted average
        avg_x = sum(p["x"] for p in predictions) / len(predictions)
        avg_y = sum(p["y"] for p in predictions) / len(predictions)
        avg_conf = 0.5
    
    # Get best reasoning
    best_pred = max(predictions, key=lambda p: p["confidence"])
    
    return {
        "x": int(round(avg_x)),
        "y": int(round(avg_y)),
        "confidence": float(avg_conf),
        "reasoning": best_pred["reasoning"],
        "method": "weighted_average",
        "num_predictions": len(predictions),
        "predictions": predictions  # Keep individual predictions for analysis
    }

# -------------------------
# Visualization
# -------------------------

def visualize_position(
    map_img: Image.Image,
    x: int,
    y: int,
    confidence: float,
    camera_name: str,
    reasoning: str = "",
) -> str:
    """Draw the predicted camera position on the map"""
    
    img = map_img.copy()
    draw = ImageDraw.Draw(img, "RGBA")
    
    # Draw confidence radius (larger = less confident)
    radius = int(30 * (1.1 - confidence))  # 3-30 pixels
    
    # Semi-transparent uncertainty circle
    uncertainty_color = (255, 255, 0, 100)  # Yellow with alpha
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=uncertainty_color,
        outline=(255, 255, 0, 200),
        width=2
    )
    
    # Main position marker
    marker_size = 12
    # Outer white circle
    draw.ellipse(
        [x - marker_size - 2, y - marker_size - 2, 
         x + marker_size + 2, y + marker_size + 2],
        fill=(255, 255, 255),
        outline=(0, 0, 0),
        width=2
    )
    # Inner red circle
    draw.ellipse(
        [x - marker_size, y - marker_size, x + marker_size, y + marker_size],
        fill=(255, 0, 0),
        outline=(255, 255, 255),
        width=2
    )
    
    # Crosshair for precision
    line_len = 25
    draw.line([x - line_len, y, x + line_len, y], fill=(255, 255, 255), width=3)
    draw.line([x, y - line_len, x, y + line_len], fill=(255, 255, 255), width=3)
    draw.line([x - line_len, y, x + line_len, y], fill=(255, 0, 0), width=1)
    draw.line([x, y - line_len, x, y + line_len], fill=(255, 0, 0), width=1)
    
    # Add label
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    label = f"Camera: ({x}, {y})\nConfidence: {confidence:.2%}"
    label_x = max(10, min(x + 30, map_img.width - 250))
    label_y = max(10, min(y - 50, map_img.height - 100))
    
    # Draw text with background
    if font:
        try:
            bbox = draw.textbbox((label_x, label_y), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(label, font=font)
    else:
        try:
            bbox = draw.textbbox((label_x, label_y), label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = 100, 40
    
    # Background rectangle
    draw.rectangle(
        [label_x - 5, label_y - 5, label_x + tw + 10, label_y + th + 10],
        fill=(0, 0, 0, 200),
        outline=(255, 255, 255)
    )
    draw.text((label_x, label_y), label, fill=(255, 255, 255), font=font)
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, f"localization-{camera_name}.png")
    img.save(save_path)
    return save_path

# -------------------------
# Main Function
# -------------------------

def main(
    camera_path: str,
    map_path: str,
    *,
    num_samples: int = 5,
    temperature: float = 0.3,
    max_image_size: int = 2048,
    json_only: bool = False,
    no_viz: bool = False,
):
    """
    Main localization pipeline
    """
    camera_name = os.path.splitext(os.path.basename(camera_path))[0]
    
    if not json_only:
        print("=" * 60)
        print("Single-Shot Camera Localization")
        print("=" * 60)
    
    # Load model
    start_time = time.time()
    if not json_only:
        print("\n[1/4] Loading model...")
    processor, model = load_model()
    model_time = time.time() - start_time
    if not json_only:
        print(f"  ✓ Model loaded in {model_time:.2f}s")
    
    # Load images
    if not json_only:
        print("\n[2/4] Loading images...")
    camera_img = load_image(camera_path, max_image_size)
    map_img = load_image(map_path, max_image_size)
    if not json_only:
        print(f"  ✓ Camera: {camera_img.size}")
        print(f"  ✓ Map: {map_img.size}")
    
    # Localize
    if not json_only:
        print(f"\n[3/4] Localizing camera position ({num_samples} samples)...")
    inference_start = time.time()
    result = localize_camera_position(
        processor,
        model,
        camera_img,
        map_img,
        num_samples=num_samples,
        temperature=temperature,
    )
    inference_time = time.time() - inference_start
    
    if not json_only:
        print(f"  ✓ Position: ({result['x']}, {result['y']})")
        print(f"  ✓ Confidence: {result['confidence']:.2%}")
        print(f"  ✓ Reasoning: {result['reasoning']}")
        print(f"  ✓ Inference time: {inference_time:.2f}s")
    
    # Visualize
    viz_path = None
    if not no_viz:
        if not json_only:
            print("\n[4/4] Generating visualization...")
        viz_path = visualize_position(
            map_img,
            result["x"],
            result["y"],
            result["confidence"],
            camera_name,
            result["reasoning"],
        )
        if not json_only:
            print(f"  ✓ Saved: {viz_path}")
    
    total_time = time.time() - start_time
    
    # Output
    output = {
        "success": True,
        "position": {
            "x": result["x"],
            "y": result["y"]
        },
        "confidence": result["confidence"],
        "reasoning": result["reasoning"],
        "method": result["method"],
        "num_predictions": result.get("num_predictions", 1),
        "timing": {
            "model_load": round(model_time, 2),
            "inference": round(inference_time, 2),
            "total": round(total_time, 2),
        },
        "meta": {
            "model": MODEL_ID,
            "num_samples": num_samples,
            "temperature": temperature,
            "map_size": list(map_img.size),
        }
    }
    
    if viz_path:
        output["visualization"] = viz_path
    
    if json_only:
        print(json.dumps(output, indent=2))
    else:
        print("\n" + "=" * 60)
        print("RESULT:")
        print("=" * 60)
        print(json.dumps(output, indent=2))
        print("\n✓ Localization complete!")
    
    return output

# -------------------------
# Entry Point
# -------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-shot camera position localization"
    )
    parser.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Path to camera/street view image"
    )
    parser.add_argument(
        "--map",
        type=str,
        required=True,
        help="Path to map/aerial image"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of prediction samples (default: 5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=2048,
        help="Max image dimension (default: 2048)"
    )
    parser.add_argument(
        "--json",
        dest="json_only",
        action="store_true",
        help="Output JSON only"
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation"
    )
    
    args = parser.parse_args()
    
    if args.json_only:
        DEBUG = False
    
    main(
        args.camera,
        args.map,
        num_samples=args.samples,
        temperature=args.temperature,
        max_image_size=args.max_size,
        json_only=args.json_only,
        no_viz=args.no_viz,
    )

