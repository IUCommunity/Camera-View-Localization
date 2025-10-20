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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList

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

def load_image(path: str, max_size: int = 0) -> Image.Image:
    """Load image without resizing (process full resolution)."""
    img = Image.open(path).convert("RGB")
    return img

def simple_localize(processor, model, camera_img: Image.Image, map_img: Image.Image) -> Dict[str, Any]:
    """
    Simple localization with basic prompting
    """
    map_width, map_height = map_img.size
    
    # Minimal, explicit prompt (no examples/ranges)
    prompt = (
        f"Map size: {map_width}x{map_height} pixels. "
        "Given the street-view (first image) and the aerial map (second image), "
        "return ONLY a JSON object with: x (integer pixel), y (integer pixel), confidence (0 to 1), reason (short)."
    )

    chat = [
        {
            "role": "system", 
            "content": "You are a geo-localization expert. Output only a JSON code block with keys x, y, confidence, reason. No other text."
        },
        {
            "role": "user", 
            "content": [
                {"type": "image"},  # camera image
                {"type": "image"},  # map image  
                {"type": "text", "text": prompt + " Respond in a single ```json code block with only the JSON object."},
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
        
        # Custom stopper: end early once we likely have a full JSON object or we hit a wall-time
        class JsonStopper(StoppingCriteria):
            def __init__(self, tokenizer, start_time: float, timeout_s: float = 20.0):
                self.tokenizer = tokenizer
                self.start_time = start_time
                self.timeout_s = timeout_s
                self.seen_brace = False
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                if time.time() - self.start_time > self.timeout_s:
                    return True
                # Check only the generated continuation (last 200 tokens)
                tail = input_ids[0][-200:]
                text = self.tokenizer.decode(tail, skip_special_tokens=True)
                if '{' in text:
                    self.seen_brace = True
                if self.seen_brace and ('"x"' in text or '"y"' in text) and '}' in text:
                    return True
                return False

        stop_criteria = StoppingCriteriaList([JsonStopper(processor.tokenizer, time.time(), timeout_s=25.0)])

        # Bounded, deterministic generation
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                repetition_penalty=1.02,
                stopping_criteria=stop_criteria,
            )
        
        # Decode only the newly generated tokens (exclude the prompt)
        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = output[0][prompt_len:]
        response = processor.decode(generated_ids, skip_special_tokens=True)
        print(f"Raw response: {response}")
        
        # Prefer extracting from the last fenced json block immediately
        fenced = re.findall(r"```json\s*([\s\S]*?)```", response, re.IGNORECASE)
        if fenced:
            candidate = fenced[-1].strip()
            try:
                data = json.loads(candidate)
                x = max(0, min(int(data.get("x", map_width // 2)), map_width - 1))
                y = max(0, min(int(data.get("y", map_height // 2)), map_height - 1))
                confidence = max(0.0, min(float(data.get("confidence", 0.5)), 1.0))
                reason = str(data.get("reason", data.get("reasoning", ""))) or ""
                return {"x": x, "y": y, "confidence": confidence, "reason": reason}
            except Exception as e:
                print(f"Immediate fenced JSON parse failed: {e}")
        
        # Extract JSON from response
        result = extract_coordinates(response, map_width, map_height)
        print(f"Extracted result: {result}")
        if "error" not in result:
            return result

        # Retry once with strict key=value format
        print("Retrying with key=value format prompt...")
        retry_prompt = (
            f"Map size: {map_width}x{map_height}. "
            "Return ONLY this (no extra words): x=<int>, y=<int>, confidence=<0..1>, reason=<short>"
        )
        chat2 = [
            {"role": "system", "content": "Output only: x=<int>, y=<int>, confidence=<0..1>, reason=<short>."},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": retry_prompt},
            ]},
        ]
        chat_str2 = processor.apply_chat_template(chat2, add_generation_prompt=True)
        inputs2 = processor(text=chat_str2, images=[camera_img, map_img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out2 = model.generate(
                **inputs2,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens for retry as well
        prompt_len2 = inputs2["input_ids"].shape[-1]
        gen_ids2 = out2[0][prompt_len2:]
        resp2 = processor.decode(gen_ids2, skip_special_tokens=True)
        print(f"Retry raw response: {resp2}")
        m = re.search(r"x\s*=\s*(\d+)\s*[,\s]+y\s*=\s*(\d+)\s*[,\s]+confidence\s*=\s*([0-1]?(?:\.\d+)?)\s*[,\s]+reason\s*=\s*(.+)$", resp2, re.IGNORECASE)
        if m:
            try:
                x = max(0, min(int(m.group(1)), map_width - 1))
                y = max(0, min(int(m.group(2)), map_height - 1))
                conf = float(m.group(3)); conf = max(0.0, min(conf, 1.0))
                reason = m.group(4).strip().strip('"')
                return {"x": x, "y": y, "confidence": conf, "reason": reason}
            except Exception as e:
                print(f"Retry parse failed: {e}")
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
    
    sanitized = text
    
    # Prefer JSON inside fenced code blocks first
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fenced:
        for block in reversed(fenced):
            try:
                candidate = block.strip()
                print(f"Trying fenced JSON: {candidate[:120]}...")
                data = json.loads(candidate)
                x = max(0, min(int(data.get("x", map_width // 2)), map_width - 1))
                y = max(0, min(int(data.get("y", map_height // 2)), map_height - 1))
                confidence = max(0.0, min(float(data.get("confidence", 0.5)), 1.0))
                reason = str(data.get("reason", data.get("reasoning", ""))) or ""
                return {"x": x, "y": y, "confidence": confidence, "reason": reason}
            except Exception as e:
                print(f"Fenced JSON parse failed: {e}")
    
    # Method 1a: Balanced-brace extraction of first JSON object
    def try_extract_balanced(s: str) -> Optional[Dict[str, Any]]:
        for start in [m.start() for m in re.finditer(r"\{", s)]:
            depth = 0
            in_str = False
            esc = False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = s[start:i+1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict) and ("x" in obj and "y" in obj):
                                    return obj
                            except Exception:
                                pass
                            break
            # continue with next '{'
        return None

    obj_balanced = try_extract_balanced(text)
    if obj_balanced is not None:
        try:
            x = max(0, min(int(obj_balanced.get("x", map_width // 2)), map_width - 1))
            y = max(0, min(int(obj_balanced.get("y", map_height // 2)), map_height - 1))
            confidence = max(0.0, min(float(obj_balanced.get("confidence", 0.5)), 1.0))
            reason = str(obj_balanced.get("reason", obj_balanced.get("reasoning", ""))) or ""
            print(f"Balanced extraction success: x={x}, y={y}, conf={confidence}")
            return {"x": x, "y": y, "confidence": confidence, "reason": reason}
        except Exception as e:
            print(f"Balanced extraction parse failed: {e}")

    # Method 1b: Try to find JSON with more flexible regex pattern
    json_patterns = [
        r'\{[^{}]*"x"[^{}]*"y"[^{}]*\}',  # Original pattern
        r'\{[^}]*"x"[^}]*"y"[^}]*\}',     # Simpler pattern
        r'\{"x":\s*\d+[^}]*"y":\s*\d+[^}]*\}',  # More specific
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, sanitized, re.IGNORECASE | re.DOTALL)
        if matches:
            for match in reversed(matches):
                try:
                    print(f"Trying to parse JSON: {match}")
                    obj = json.loads(match)
                    if not ("x" in obj and "y" in obj):
                        continue
                    x_val = int(obj["x"])  # let this raise if not int-like
                    y_val = int(obj["y"])  # let this raise if not int-like
                    x = max(0, min(x_val, map_width - 1))
                    y = max(0, min(y_val, map_height - 1))
                    conf = float(obj.get("confidence", 0.5))
                    conf = max(0.0, min(conf, 1.0))
                    reason = str(obj.get("reason", obj.get("reasoning", "")))
                    print(f"Successfully parsed JSON: x={x}, y={y}, conf={conf}")
                    return {"x": x, "y": y, "confidence": conf, "reason": reason}
                except Exception as e:
                    print(f"JSON parsing failed: {e}")
                    continue
    
    # Final fallback: mark as error so caller sets success=false
    print("Using fallback coordinates")  # Debug output
    return {
        "x": map_width // 2,
        "y": map_height // 2,
        "confidence": 0.0,
        "reason": "",
        "error": "Could not extract coordinates from response",
    }

def visualize_result(map_img: Image.Image, result: Dict[str, Any], camera_name: str) -> str:
    """Create simple visualization"""
    
    # Work on RGBA to support alpha overlays; clamp coordinates to image bounds
    img = map_img.copy().convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    
    x = max(0, min(int(result.get("x", 0)), img.width - 1))
    y = max(0, min(int(result.get("y", 0)), img.height - 1))
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
        outline=(255, 255, 0, 200),
        width=2
    )
    
    # Add label
    label = f"({x}, {y})\nConf: {confidence:.2f}"
    label_x = max(10, min(x + 20, img.width - 100))
    label_y = max(10, min(y - 30, img.height - 50))
    
    # Background for text
    draw.rectangle(
        [label_x - 5, label_y - 5, label_x + 120, label_y + 36],
        fill=(0, 0, 0, 180),
        outline=(255, 255, 255, 220)
    )
    draw.text((label_x, label_y), label, fill=(255, 255, 255, 255))
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, f"simple-localization-{camera_name}.png")
    # Save as RGB for compatibility
    img.convert("RGB").save(save_path)
    return save_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple geo-localization")
    parser.add_argument("--camera", required=True, help="Path to camera image")
    parser.add_argument("--map", required=True, help="Path to map image")
    # Keep arg for compatibility, but no resizing is applied
    parser.add_argument("--max-size", type=int, default=0, help="(ignored) no resizing is applied")
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
    camera_img = load_image(args.camera)
    map_img = load_image(args.map)
    print(f"Camera image: {camera_img.size}")
    print(f"Map image: {map_img.size}")
    
    # Localize
    print(f"\nPerforming localization...")
    start_time = time.time()
    result = simple_localize(processor, model, camera_img, map_img)
    inference_time = time.time() - start_time

    # Auto-retry with safe dtype on CUDA assert
    if (not args.safe) and ("error" in result) and (
        "device-side assert" in str(result.get("error", "")).lower() or
        "cuda error" in str(result.get("error", "")).lower()
    ):
        print("\nEncountered CUDA device-side assert. Retrying with safe float32...")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        # Reload model safely and retry once
        safe_start = time.time()
        processor, model = load_model(use_safe_dtype=True)
        model_time += time.time() - safe_start
        retry_start = time.time()
        result = simple_localize(processor, model, camera_img, map_img)
        inference_time = time.time() - retry_start
    
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
