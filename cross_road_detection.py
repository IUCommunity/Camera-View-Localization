# Project: Cross Road Detection with Qwen2.5-VL
# ---------------------------------------------------
# This pipeline detects cross roads in a single input image.
# Input: One image
# Output: "YES" if there's any cross road, "NO" if there isn't any cross roads.

import os
import json
import argparse
import time
from typing import Dict, Any
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# -------------------------
# Configuration
# -------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # Hugging Face repo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG = True  # set False for quieter logs

# -------------------------
# Model Loader
# -------------------------

def load_model():
    """Load the Qwen2.5-VL model following the same pattern as main.py"""
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

def load_image(path: str, max_size: int = None) -> Image.Image:
    """Load and optionally resize image following main.py pattern"""
    img = Image.open(path).convert("RGB")
    if max_size and (img.width > max_size or img.height > max_size):
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img

# -------------------------
# Cross Road Detection
# -------------------------

def detect_cross_road(
    processor,
    model,
    image: Image.Image,
    *,
    samples: int = 3,
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_new_tokens: int = 10,
    fast_mode: bool = False,
) -> str:
    """
    Detect if there are any cross roads in the input image.
    Returns "YES" if cross roads are detected, "NO" otherwise.
    """
    
    # Clear and specific system prompt for cross road and curved road detection
    system_prompt = (
        "You are a precise road detector. Your task is to analyze the given image and determine if there are any cross roads or curved roads present.\n\n"
        "A cross road is defined as:\n"
        "- An intersection where two or more roads meet and cross each other\n"
        "- Roads that intersect at angles (typically perpendicular or at significant angles)\n"
        "- Visible road markings, lane lines, or traffic patterns that indicate road intersections\n"
        "- Infrastructure elements like traffic lights, stop signs, or crosswalks at intersections\n\n"
        "A curved road is defined as:\n"
        "- A road that has a noticeable curve, bend, or arc in its path\n"
        "- Roads that change direction gradually or sharply (not straight lines)\n"
        "- Roads with visible curvature in their lane markings or boundaries\n"
        "- Roads that follow a curved trajectory rather than being perfectly straight\n\n"
        "Important guidelines:\n"
        "- Look for actual road intersections (cross roads) or curved road segments\n"
        "- Consider both major and minor road intersections and curves\n"
        "- Ignore weather conditions, lighting, or seasonal changes\n"
        "- Focus on permanent road infrastructure and markings\n"
        "- A single straight road without intersections or curves is NOT a cross road or curved road\n"
        "- Both cross roads AND curved roads should result in 'YES'\n\n"
        "Output ONLY one word: 'YES' if you detect any cross roads or curved roads, or 'NO' if you do not detect any cross roads or curved roads.\n"
        "Do not provide explanations, justifications, or additional text."
    )

    # Fast mode: reduce samples and use deterministic generation
    if fast_mode:
        samples = 1
        temperature = 0.0
        max_new_tokens = 5

    results = []
    
    for i in range(max(1, samples)):
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Analyze this image for cross roads. Output only YES or NO."},
            ]},
        ]
        
        chat_str = processor.apply_chat_template(chat, add_generation_prompt=True)
        inputs = processor(text=chat_str, images=[image], return_tensors="pt").to(model.device)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if samples > 1 or temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": max(0.01, float(temperature)), "top_p": float(top_p)})
        else:
            gen_kwargs.update({"do_sample": False})

        out = model.generate(**inputs, **gen_kwargs)
        resp = processor.decode(out[0], skip_special_tokens=True)
        
        # Extract the response after the assistant marker
        if "assistant" in resp:
            resp = resp.split("assistant", 1)[-1].strip()
        
        # Clean up the response to get just YES or NO
        resp = resp.strip().upper()
        if "YES" in resp:
            results.append("YES")
        elif "NO" in resp:
            results.append("NO")
        else:
            # Fallback: if unclear response, default to NO
            results.append("NO")
        
        if DEBUG:
            print(f"Sample {i+1}: Raw response: '{resp}' -> Processed: '{results[-1]}'")

    # Majority voting for final result
    yes_count = results.count("YES")
    no_count = results.count("NO")
    
    if yes_count > no_count:
        return "YES"
    elif no_count > yes_count:
        return "NO"
    else:
        # Tie case: default to NO (conservative approach)
        return "NO"

# -------------------------
# Main Pipeline
# -------------------------

def main(image_path: str, *, samples: int = 3, temperature: float = 0.1, top_p: float = 0.9, fast_mode: bool = False, max_image_size: int = None, json_output: bool = False):
    """
    Main function to detect cross roads in an image.
    
    Args:
        image_path: Path to the input image
        samples: Number of samples for majority voting
        temperature: Sampling temperature
        top_p: Top-p nucleus sampling
        fast_mode: Use fast mode (1 sample, deterministic)
        max_image_size: Maximum image size for resizing
        json_output: Output result in JSON format
    """
    start_time = time.time()
    
    if not json_output:
        print("Loading model...")
    model_load_start = time.time()
    processor, model = load_model()
    model_load_time = time.time() - model_load_start
    if not json_output:
        print(f"Model loaded in {model_load_time:.2f}s")

    # Load and process image
    if not json_output:
        print(f"Loading image: {image_path}")
    image = load_image(image_path, max_image_size)
    if not json_output:
        print(f"Image size: {image.size}")

    # Detect cross roads
    if not json_output:
        print("Detecting cross roads...")
    detection_start = time.time()
    
    result = detect_cross_road(
        processor,
        model,
        image,
        samples=samples,
        temperature=temperature,
        top_p=top_p,
        fast_mode=fast_mode,
    )
    
    detection_time = time.time() - detection_start
    total_time = time.time() - start_time

    if json_output:
        output = {
            "success": True,
            "result": result,
            "image_path": image_path,
            "timing": {
                "model_load_time": round(model_load_time, 2),
                "detection_time": round(detection_time, 2),
                "total_time": round(total_time, 2)
            },
            "meta": {
                "model_id": MODEL_ID,
                "samples": samples,
                "temperature": temperature,
                "top_p": top_p,
                "fast_mode": fast_mode,
                "max_image_size": max_image_size
            }
        }
        print(json.dumps(output))
    else:
        print(f"\n🎯 Cross Road Detection Result: {result}")
        print(f"\n[Pipeline finished] — Total time: {total_time:.2f}s")
        print(f"Model load: {model_load_time:.2f}s, Detection: {detection_time:.2f}s")

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross Road Detection with Qwen2.5-VL")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples for majority voting")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode (1 sample, deterministic)")
    parser.add_argument("--max-image-size", type=int, default=None, help="Resize images if larger than this (for speed)")
    parser.add_argument("--json", action="store_true", help="Output result in JSON format")
    parser.add_argument("--quiet", action="store_true", help="Disable debug output")
    args = parser.parse_args()

    # Set debug mode
    if args.quiet:
        DEBUG = False

    main(
        args.image,
        samples=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        fast_mode=args.fast,
        max_image_size=args.max_image_size,
        json_output=args.json,
    )
