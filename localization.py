# Project: Single-Shot Camera Position Localization
# ---------------------------------------------------
# This approach uses direct pixel prediction instead of bounding boxes
# and processes the entire map in one inference for speed and accuracy.

import os
import json
import re
import argparse
import time
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# -------------------------
# Configuration
# -------------------------
MODEL_ID = r"D:\Research\LLaMA-Factory\models_export"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results"
DEBUG = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Model Loader
# -------------------------

def load_model():
    """Load the vision-language model with optimizations"""
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,  # Use float16 for better compatibility and speed
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    # Enable optimizations if available
    try:
        # Use torch.compile for faster inference (PyTorch 2.0+)
        model = torch.compile(model, mode="reduce-overhead")
        if DEBUG:
            print("  ✓ Model compiled for optimization")
    except Exception as e:
        if DEBUG:
            print(f"  ⚠️ torch.compile not available: {e}")
    
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
# Advanced Prompting Functions
# -------------------------

def get_high_accuracy_prompt(map_width: int, map_height: int) -> str:
    """Generate optimized prompt with few-shot examples for high accuracy"""
    
    return f"""You are an expert visual localization system with specialized training in matching street-level views to aerial maps.

TASK: Find the exact pixel coordinates where the street-level camera was positioned when taking the photo.

FEW-SHOT EXAMPLES:

Example 1:
Street View: Shows T-intersection with red building on left, straight road ahead
Aerial Map: T-intersection visible, red building footprint matches
Result: Camera at intersection center, coordinates (512, 384)
Confidence: 0.92
Reasoning: "T-intersection geometry and building color/shape match perfectly"

Example 2:
Street View: Curved road with trees on both sides, building visible ahead
Aerial Map: Curved road segment with vegetation, building at end
Result: Camera positioned along curved road, coordinates (678, 234)
Confidence: 0.87
Reasoning: "Road curvature and vegetation pattern match, building placement consistent"

ANALYSIS METHODOLOGY:
1. **Landmark Matching**: Identify distinctive buildings, intersections, or structures
2. **Road Geometry**: Match road curves, intersections, and lane configurations  
3. **Spatial Relationships**: Verify relative positions of buildings, roads, and features
4. **Scale Estimation**: Use known objects to estimate camera position accuracy
5. **Perspective Validation**: Ensure viewing angle matches aerial perspective

CRITICAL REQUIREMENTS:
- Examine the camera image for permanent landmarks (buildings, roads, intersections)
- Match these landmarks to corresponding features in the aerial map
- Consider the camera's viewing direction and field of view
- Account for scale differences between street view and aerial view
- Ignore temporary objects (cars, people, shadows, weather)

OUTPUT FORMAT - Return ONLY valid JSON:
{{
    "x": <integer_pixel_x>,
    "y": <integer_pixel_y>,
    "confidence": <float_0_to_1>,
    "reasoning": "<detailed_explanation_of_match>",
    "landmarks": ["<landmark1>", "<landmark2>"],
    "viewing_direction": "<north|south|east|west|northeast|northwest|southeast|southwest>"
}}

COORDINATE CONSTRAINTS:
- x must be integer between 0 and {map_width-1}
- y must be integer between 0 and {map_height-1}
- confidence must be float between 0.0 and 1.0

NO explanations outside JSON. NO markdown formatting."""

def get_fast_mode_prompt(map_width: int, map_height: int) -> str:
    """Generate optimized prompt for fast mode"""
    
    return f"""You are a precise visual localization system. Find the camera position on the aerial map.

TASK: Determine the exact pixel coordinates where the street-level camera was located.

ANALYSIS:
1. Identify key landmarks in the street view (buildings, intersections, roads)
2. Match these to corresponding features in the aerial map
3. Determine camera position coordinates

OUTPUT FORMAT - Return ONLY valid JSON:
{{
    "x": <integer_pixel_x>,
    "y": <integer_pixel_y>,
    "confidence": <float_0_to_1>,
    "reasoning": "<brief_explanation>"
}}

Coordinates: x=[0,{map_width-1}], y=[0,{map_height-1}]. Confidence: [0.0,1.0]"""

# -------------------------
# Validation Functions
# -------------------------

def validate_prediction(result: Dict[str, Any], map_width: int, map_height: int) -> bool:
    """Validate prediction quality and format"""
    
    if not isinstance(result, dict):
        return False
    
    # Check required fields
    required_fields = ["x", "y", "confidence", "reasoning"]
    if not all(field in result for field in required_fields):
        return False
    
    try:
        x = int(result["x"])
        y = int(result["y"])
        confidence = float(result["confidence"])
        reasoning = str(result["reasoning"])
        
        # Validate coordinate bounds
        if not (0 <= x < map_width and 0 <= y < map_height):
            return False
        
        # Validate confidence range
        if not (0.0 <= confidence <= 1.0):
            return False
        
        # Validate reasoning quality (should be descriptive)
        if len(reasoning.strip()) < 5:
            return False
            
        return True
        
    except (ValueError, TypeError):
        return False

def remove_outliers(predictions: List[Dict[str, Any]], threshold: float = 2.0) -> List[Dict[str, Any]]:
    """Remove statistical outliers from predictions"""
    
    if len(predictions) < 3:
        return predictions
    
    x_coords = [p["x"] for p in predictions]
    y_coords = [p["y"] for p in predictions]
    
    # Calculate mean and standard deviation
    x_mean = sum(x_coords) / len(x_coords)
    y_mean = sum(y_coords) / len(y_coords)
    x_std = (sum((x - x_mean) ** 2 for x in x_coords) / len(x_coords)) ** 0.5
    y_std = (sum((y - y_mean) ** 2 for y in y_coords) / len(y_coords)) ** 0.5
    
    # Filter outliers
    filtered = []
    for pred in predictions:
        x_z_score = abs(pred["x"] - x_mean) / (x_std + 1e-8)
        y_z_score = abs(pred["y"] - y_mean) / (y_std + 1e-8)
        
        if x_z_score < threshold and y_z_score < threshold:
            filtered.append(pred)
    
    return filtered

def calculate_consensus(predictions: List[Dict[str, Any]]) -> float:
    """Calculate how well predictions agree with each other"""
    
    if len(predictions) < 2:
        return 1.0
    
    x_coords = [p["x"] for p in predictions]
    y_coords = [p["y"] for p in predictions]
    
    # Calculate average pairwise distance
    total_distance = 0
    count = 0
    
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            distance = ((x_coords[i] - x_coords[j]) ** 2 + (y_coords[i] - y_coords[j]) ** 2) ** 0.5
            total_distance += distance
            count += 1
    
    if count == 0:
        return 1.0
    
    avg_distance = total_distance / count
    
    # Convert distance to consensus score (0-1, higher is better)
    # Assume good consensus if average distance < 50 pixels
    consensus = max(0.0, 1.0 - (avg_distance / 50.0))
    return consensus

# -------------------------
# Core Localization Function
# -------------------------

def localize_camera_position(
    processor,
    model,
    camera_img: Image.Image,
    map_img: Image.Image,
    *,
    num_samples: int = 3,
    temperature: float = 0.1,
    fast_mode: bool = False,
    high_accuracy: bool = False,
) -> Dict[str, Any]:
    """
    Directly predict camera position as (x, y) pixel coordinates on the map.
    Uses multiple samples and advanced aggregation for robustness.
    """
    
    map_width, map_height = map_img.size
    
    # Optimize parameters based on mode
    if fast_mode:
        num_samples = 1
        temperature = 0.0
        max_tokens = 100
        prompt = get_fast_mode_prompt(map_width, map_height)
    elif high_accuracy:
        num_samples = max(2, num_samples)  # Minimum 2 for high accuracy
        temperature = 0.05
        max_tokens = 250
        prompt = get_high_accuracy_prompt(map_width, map_height)
    else:
        max_tokens = 200
        prompt = get_fast_mode_prompt(map_width, map_height)  # Use optimized prompt

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
        
        # Optimized generation parameters
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": not fast_mode,  # Deterministic in fast mode
        }
        if not fast_mode and temperature > 0:
            gen_kwargs.update({
                "temperature": max(0.01, temperature),
                "top_p": 0.9,
                "repetition_penalty": 1.1,  # Prevent repetition
                "no_repeat_ngram_size": 3,   # Prevent n-gram repetition
            })
        
        with torch.no_grad():
            output = model.generate(**inputs, **gen_kwargs)
        
        response = processor.decode(output[0], skip_special_tokens=True)
        result = extract_json(response)
        
        if DEBUG and i == 0:
            print(f"Sample response: {response[-200:] if fast_mode else response[-400:]}")
        
        # Validate result using new validation function
        if validate_prediction(result, map_width, map_height):
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
                
                # Early stopping for fast mode or high confidence
                if fast_mode or (not fast_mode and conf >= 0.9):
                    break
                    
            except (ValueError, TypeError) as e:
                if DEBUG:
                    print(f"  Sample {i+1}: Invalid format - {e}")
                    print(f"  Raw result: {result}")
                continue
        else:
            if DEBUG:
                print(f"  Sample {i+1}: Invalid prediction, skipping")
    
    # Advanced aggregation with outlier detection and consensus validation
    return aggregate_predictions_advanced(predictions, map_width, map_height)

def aggregate_predictions_advanced(predictions: List[Dict[str, Any]], map_width: int, map_height: int) -> Dict[str, Any]:
    """Advanced aggregation with outlier detection and consensus validation"""
    
    if not predictions:
        return {
            "x": map_width // 2,
            "y": map_height // 2,
            "confidence": 0.0,
            "reasoning": "No valid predictions generated",
            "method": "fallback"
        }
    
    # Remove outliers using statistical methods
    filtered_predictions = remove_outliers(predictions)
    
    if len(filtered_predictions) == 0:
        filtered_predictions = predictions  # Fallback to original if all are outliers
    
    # Calculate consensus metrics
    consensus_score = calculate_consensus(filtered_predictions)
    
    # Use weighted average with confidence and consensus
    total_weight = 0
    weighted_x = 0
    weighted_y = 0
    total_confidence = 0
    
    for pred in filtered_predictions:
        # Weight combines confidence and consensus
        weight = pred["confidence"] * (1 + consensus_score)
        weighted_x += pred["x"] * weight
        weighted_y += pred["y"] * weight
        total_weight += weight
        total_confidence += pred["confidence"]
    
    if total_weight > 0:
        final_x = int(round(weighted_x / total_weight))
        final_y = int(round(weighted_y / total_weight))
        avg_confidence = total_confidence / len(filtered_predictions)
    else:
        # Fallback to simple average
        final_x = int(round(sum(p["x"] for p in filtered_predictions) / len(filtered_predictions)))
        final_y = int(round(sum(p["y"] for p in filtered_predictions) / len(filtered_predictions)))
        avg_confidence = 0.5
    
    # Get best reasoning from highest confidence prediction
    best_pred = max(filtered_predictions, key=lambda p: p["confidence"])
    
    return {
        "x": final_x,
        "y": final_y,
        "confidence": float(avg_confidence),
        "reasoning": best_pred["reasoning"],
        "method": "advanced_aggregation",
        "num_predictions": len(filtered_predictions),
        "consensus_score": consensus_score,
        "predictions": filtered_predictions
    }

# -------------------------
# Multi-Scale Analysis Functions
# -------------------------

def multi_scale_localization(
    processor,
    model,
    camera_img: Image.Image,
    map_img: Image.Image,
    *,
    scales: List[float] = [0.75, 1.0, 1.25],
    num_samples_per_scale: int = 2,
    fast_mode: bool = False,
) -> Dict[str, Any]:
    """Analyze at multiple scales for improved accuracy"""
    
    map_width, map_height = map_img.size
    scale_results = []
    
    for scale in scales:
        if DEBUG:
            print(f"  Analyzing at scale {scale}x...")
        
        # Scale the map image
        if scale != 1.0:
            new_width = int(map_width * scale)
            new_height = int(map_height * scale)
            scaled_map = map_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else:
            scaled_map = map_img
        
        # Localize on scaled image
        result = localize_camera_position(
            processor, model, camera_img, scaled_map,
            num_samples=num_samples_per_scale,
            temperature=0.05 if not fast_mode else 0.0,
            fast_mode=fast_mode,
            high_accuracy=not fast_mode
        )
        
        # Scale coordinates back to original size
        if scale != 1.0:
            result['x'] = int(result['x'] / scale)
            result['y'] = int(result['y'] / scale)
        
        # Add scale information
        result['scale'] = scale
        result['scale_confidence'] = result['confidence']
        
        scale_results.append(result)
    
    # Combine multi-scale results
    return combine_multi_scale_results(scale_results, map_width, map_height)

def combine_multi_scale_results(scale_results: List[Dict[str, Any]], map_width: int, map_height: int) -> Dict[str, Any]:
    """Combine results from multiple scales"""
    
    if not scale_results:
        return {
            "x": map_width // 2,
            "y": map_height // 2,
            "confidence": 0.0,
            "reasoning": "No multi-scale results",
            "method": "fallback"
        }
    
    # Weight by confidence and scale (prefer native scale)
    weighted_x = 0
    weighted_y = 0
    total_weight = 0
    
    for result in scale_results:
        # Weight combines confidence and scale preference
        scale_weight = 1.0 if result['scale'] == 1.0 else 0.8
        weight = result['confidence'] * scale_weight
        
        weighted_x += result['x'] * weight
        weighted_y += result['y'] * weight
        total_weight += weight
    
    if total_weight > 0:
        final_x = int(round(weighted_x / total_weight))
        final_y = int(round(weighted_y / total_weight))
    else:
        # Fallback to simple average
        final_x = int(round(sum(r['x'] for r in scale_results) / len(scale_results)))
        final_y = int(round(sum(r['y'] for r in scale_results) / len(scale_results)))
    
    # Get best reasoning from highest confidence result
    best_result = max(scale_results, key=lambda r: r['confidence'])
    avg_confidence = sum(r['confidence'] for r in scale_results) / len(scale_results)
    
    return {
        "x": final_x,
        "y": final_y,
        "confidence": float(avg_confidence),
        "reasoning": best_result['reasoning'],
        "method": "multi_scale",
        "scale_results": scale_results,
        "num_scales": len(scale_results)
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
    mode: str = "balanced",  # "fast", "balanced", "high_accuracy"
    num_samples: int = 3,
    temperature: float = 0.1,
    max_image_size: int = 1024,
    json_only: bool = False,
    no_viz: bool = False,
):
    """
    High-performance localization pipeline with multiple accuracy modes
    """
    camera_name = os.path.splitext(os.path.basename(camera_path))[0]
    
    # Configure parameters based on mode
    if mode == "fast":
        fast_mode = True
        high_accuracy = False
        use_multi_scale = False
        confidence_threshold = 0.7
    elif mode == "balanced":
        fast_mode = False
        high_accuracy = False
        use_multi_scale = False
        confidence_threshold = 0.8
    elif mode == "high_accuracy":
        fast_mode = False
        high_accuracy = True
        use_multi_scale = True
        confidence_threshold = 0.85
        num_samples = max(2, num_samples)  # Minimum 2 for high accuracy
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'fast', 'balanced', or 'high_accuracy'")
    
    if not json_only:
        print("=" * 60)
        print(f"High-Performance Camera Localization - {mode.upper()} MODE")
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
    
    # Perform localization
    if not json_only:
        if use_multi_scale:
            print(f"\n[3/4] Multi-scale localization ({num_samples} samples per scale)...")
        else:
            mode_str = "FAST MODE" if fast_mode else f"STANDARD MODE ({num_samples} samples)"
            print(f"\n[3/4] Localizing camera position - {mode_str}...")
    
    inference_start = time.time()
    
    if use_multi_scale:
        result = multi_scale_localization(
            processor, model, camera_img, map_img,
            num_samples_per_scale=num_samples,
            fast_mode=fast_mode
        )
    else:
        result = localize_camera_position(
            processor,
            model,
            camera_img,
            map_img,
            num_samples=num_samples,
            temperature=temperature,
            fast_mode=fast_mode,
            high_accuracy=high_accuracy,
        )
    
    inference_time = time.time() - inference_start
    
    if not json_only:
        print(f"  ✓ Position: ({result['x']}, {result['y']})")
        print(f"  ✓ Confidence: {result['confidence']:.2%}")
        print(f"  ✓ Reasoning: {result['reasoning']}")
        if 'consensus_score' in result:
            print(f"  ✓ Consensus: {result['consensus_score']:.2%}")
        if 'num_scales' in result:
            print(f"  ✓ Scales analyzed: {result['num_scales']}")
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
            "mode": mode,
            "num_samples": num_samples,
            "temperature": temperature,
            "map_size": list(map_img.size),
            "fast_mode": fast_mode,
            "high_accuracy": high_accuracy,
            "multi_scale": use_multi_scale,
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
        "--mode",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "high_accuracy"],
        help="Localization mode: fast (1 sample, deterministic), balanced (3 samples, optimized), high_accuracy (multi-scale analysis)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of prediction samples (default: 3)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)"
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="Max image dimension (default: 1024)"
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
        mode=args.mode,
        num_samples=args.samples,
        temperature=args.temperature,
        max_image_size=args.max_size,
        json_only=args.json_only,
        no_viz=args.no_viz,
    )

