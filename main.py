# Project: Camera-to-Map ROI Localization with Visualization (Optimized for Qwen2.5-VL, NL→ROI)
# ---------------------------------------------------
# This pipeline uses a dual-stage approach:
# 1) Camera → Natural-language scene description (ignore weather/season/transients).
# 2) Map tile → Natural-language semantic description (roads, buildings, landmarks; ignore weather/season).
# 3) Text-only semantic matching between camera-NL and map-NL to score plausibility.
# 4) ROI extraction: given camera-NL + map-NL + MAP_TILE image, ask LLM for a bounding box JSON.
# 5) Visualize tile-level ROIs and also project ROIs onto the full map with offsets.

import os
import io
import json
import re
import argparse
import time
from typing import List, Tuple, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from cross_road_detection import detect_cross_road

# -------------------------
# Configuration
# -------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # Hugging Face repo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "results"
DEBUG = True  # set False for quieter logs
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Model Loader
# -------------------------

def load_model():
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
    img = Image.open(path).convert("RGB")
    if max_size and (img.width > max_size or img.height > max_size):
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img


def split_into_tiles(img: Image.Image, tile_size: int = 1024, stride: int = 512) -> List[Tuple[Image.Image, Tuple[int, int]]]:
    tiles = []
    w, h = img.size
    if w < tile_size or h < tile_size:
        tiles.append((img, (0, 0)))
        return tiles
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            crop = img.crop((x, y, x + tile_size, y + tile_size))
            tiles.append((crop, (x, y)))
    return tiles

# -------------------------
# JSON Extraction Helper (robust)
# -------------------------

def safe_json_parse(resp: str) -> Dict[str, Any]:
    # keep only assistant section if present
    if "assistant" in resp:
        resp = resp.split("assistant", 1)[-1]
    # prefer fenced blocks
    matches = re.findall(r"```json(.*?)```", resp, re.DOTALL)
    if matches:
        candidate = matches[-1].strip()
    else:
        # fallback: last JSON-like object
        matches = re.findall(r"\{[\s\S]*\}", resp)
        if not matches:
            return {}
        candidate = matches[-1]
    try:
        return json.loads(candidate)
    except Exception:
        # quick auto-repair for common truncations
        fixed = candidate.strip()
        if fixed.endswith(","):
            fixed = fixed[:-1]
        if fixed.count("{") > fixed.count("}"):
            fixed += "}" * (fixed.count("{") - fixed.count("}"))
        if fixed.count("[") > fixed.count("]"):
            fixed += "]" * (fixed.count("[") - fixed.count("]"))
        try:
            return json.loads(fixed)
        except Exception as e:
            if DEBUG:
                print(":warning: JSON parse failed:", e)
                print("Raw candidate:", candidate)
            return {}

# -------------------------
# Direct Camera→Tile Localization (JSON)
# -------------------------

def localize_camera_on_tile(
    processor,
    model,
    cam_img: Image.Image,
    map_tile: Image.Image,
    *,
    samples: int = 3,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_new_tokens: int = 220,
    fast_mode: bool = False,
) -> Dict[str, Any]:
    prompt = (
        "You are a precise spatial localizer.\n"
        "Task: Given a CAMERA image (ground) and a MAP_TILE image (aerial), predict where the camera view lies inside the MAP_TILE.\n"
        "Rules:\n"
        "- Use persistent structures: roads, intersections, building footprints, plazas, fences/gates, sidewalks, crosswalks, permanent landmarks.\n"
        "- Ignore weather, lighting, shadows, seasons.\n"
        "Output ONLY valid JSON with fields: {\n"
        "  \"bounding_box\": [x_min, y_min, x_max, y_max],\n"
        "  \"confidence\": float\n"
        "}. Coordinates are MAP_TILE pixel coordinates. No extra text."
    )

    best_pack: Dict[str, Any] = {}
    best_conf = -1.0
    packs: List[Dict[str, Any]] = []
    
    # Fast mode: reduce samples and use deterministic generation
    if fast_mode:
        samples = 1
        temperature = 0.0
        max_new_tokens = 150

    for i in range(max(1, samples)):
        chat = [
            {"role": "system", "content": "You output JSON only. Do not include explanations."},
            {"role": "user", "content": [
                {"type": "image"},  # CAMERA
                {"type": "image"},  # MAP_TILE
                {"type": "text", "text": prompt},
            ]},
        ]
        chat_str = processor.apply_chat_template(chat, add_generation_prompt=True)
        inputs = processor(text=chat_str, images=[cam_img, map_tile], return_tensors="pt").to(model.device)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if samples > 1 or temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": max(0.01, float(temperature)), "top_p": float(top_p)})
        else:
            gen_kwargs.update({"do_sample": False})

        out = model.generate(**inputs, **gen_kwargs)
        resp = processor.decode(out[0], skip_special_tokens=True)
        pack = safe_json_parse(resp)
        if not isinstance(pack, dict):
            continue
        conf = float(pack.get("confidence", 0) or 0)
        bbox = pack.get("bounding_box")
        # quick bbox sanity
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        packs.append(pack)
        if conf > best_conf:
            best_conf = conf
            best_pack = pack

    # fallback if nothing parsed
    if not best_pack:
        return {"bounding_box": [0, 0, map_tile.size[0]-1, map_tile.size[1]-1], "confidence": 0.0}

    # Optional: combine boxes if multiple samples parsed → weighted average center + size
    if len(packs) > 1:
        import math
        w, h = map_tile.size
        total_w = 0.0
        total = 0.0
        sum_x1 = sum_y1 = sum_x2 = sum_y2 = 0.0
        for p in packs:
            c = max(0.0, float(p.get("confidence", 0) or 0))
            b = p.get("bounding_box", [0, 0, w-1, h-1])
            x1, y1, x2, y2 = [float(v) for v in b]
            sum_x1 += c * x1
            sum_y1 += c * y1
            sum_x2 += c * x2
            sum_y2 += c * y2
            total += c
        if total > 0:
            avg_bbox = [int(round(sum_x1/total)), int(round(sum_y1/total)), int(round(sum_x2/total)), int(round(sum_y2/total))]
            avg_conf = max_conf = max(p.get("confidence", 0) for p in packs)
            return {"bounding_box": avg_bbox, "confidence": float(avg_conf)}

    return best_pack

# -------------------------
# Visualization Helpers
# -------------------------

# Removed draw_bbox_on_tile function as it's no longer needed


# Removed draw_rois_on_full_map function as it's no longer needed


def draw_camera_position_on_map(full_map: Image.Image, x: int, y: int, confidence: float, camera_name: str, save_path: str = None):
    """Draw the camera position as a highlighted point on the map"""
    img = full_map.copy()
    draw = ImageDraw.Draw(img)
    
    # Draw a large circle for the camera position
    radius = 15
    # Outer circle (white border)
    draw.ellipse([x-radius-3, y-radius-3, x+radius+3, y+radius+3], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    # Inner circle (red)
    draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=(255, 0, 0), outline=(255, 255, 255), width=2)
    
    # Add a crosshair for precision
    draw.line([x-20, y, x+20, y], fill=(255, 255, 255), width=3)
    draw.line([x, y-20, x, y+20], fill=(255, 255, 255), width=3)
    
    # Add confidence label
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    
    label = f"Camera Position (conf: {confidence:.3f})"
    label_x = max(10, x - 100)
    label_y = max(10, y - 40)
    
    # Label background
    if font:
        try:
            # Try new PIL method first
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            # Fallback to old method
            tw, th = draw.textsize(label, font=font)
    else:
        try:
            bbox = draw.textbbox((0, 0), label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(label)
    draw.rectangle([label_x-5, label_y-5, label_x + tw + 10, label_y + th + 5], fill=(0, 0, 0), outline=(255, 255, 255))
    draw.text((label_x, label_y), label, fill=(255, 255, 255), font=font)
    
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, f"result-{camera_name}.png")
    img.save(save_path)
    return save_path

# -------------------------
# Helpers: Candidate selection and global point
# -------------------------

def compute_global_point_from_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    result = candidate.get("result", {})
    bbox = result.get("bounding_box") or []
    offset_x, offset_y = candidate.get("tile_offset", (0, 0))
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x_min, y_min, x_max, y_max = bbox
        cx = (float(x_min) + float(x_max)) / 2.0 + float(offset_x)
        cy = (float(y_min) + float(y_max)) / 2.0 + float(offset_y)
        x_px = int(round(cx))
        y_px = int(round(cy))
    else:
        x_px, y_px = None, None

    return {
        "x": x_px,
        "y": y_px,
        "bounding_box": bbox,
        "tile_index": candidate.get("tile_index"),
        "tile_offset": list(candidate.get("tile_offset", (0, 0))),
        "roi_confidence": float(result.get("confidence", 0) or 0),
        "semantic_score": float(candidate.get("semantic", {}).get("score", 0) or 0),
        "rank_score": float(candidate.get("rank_score", 0) or 0),
    }


# -------------------------
# Main Pipeline
# -------------------------

def main(camera_path: str, map_path: str, *, tile_size: int = 1024, stride: int = 512, json_only: bool = False, no_viz: bool = False, top_k: int = 3, samples: int = 3, temperature: float = 0.2, top_p: float = 0.9, early_termination_threshold: float = 0.85, max_tiles: int = None, fast_mode: bool = False, max_image_size: int = None):
    # Extract camera name from path for result naming
    camera_name = os.path.splitext(os.path.basename(camera_path))[0]
    start_time = time.time()
    
    if not json_only:
        print("Loading model...")
    model_load_start = time.time()
    processor, model = load_model()
    model_load_time = time.time() - model_load_start
    if not json_only:
        print(f"Model loaded in {model_load_time:.2f}s")

    cam_img = load_image(camera_path, max_image_size)
    map_img = load_image(map_path, max_image_size)

    if not json_only:
        print("[Step 1] Splitting map into tiles and localizing camera on each tile...")
    tiles = split_into_tiles(map_img, tile_size=tile_size, stride=stride)
    if not json_only:
        print(f"Generated {len(tiles)} map tiles.")

    candidates = []
    inference_start = time.time()
    total_tiles = len(tiles)
    
    # Limit tiles if max_tiles is specified
    if max_tiles and max_tiles < total_tiles:
        tiles = tiles[:max_tiles]
        total_tiles = len(tiles)
        if not json_only:
            print(f"Limited to first {total_tiles} tiles for faster processing")
    
    for idx, (tile_img, offset) in enumerate(tiles):
        if not json_only:
            print(f"Processing tile {idx+1}/{total_tiles}...")
        
        tile_start = time.time()
        roi_pack = localize_camera_on_tile(
            processor,
            model,
            cam_img,
            tile_img,
            samples=samples,
            temperature=temperature,
            top_p=top_p,
            fast_mode=fast_mode,
        )
        tile_time = time.time() - tile_start
        
        if DEBUG and not json_only:
            print(f"[TILE {idx}] ROI pack:", roi_pack)
            print(f"[TILE {idx}] Processed in {tile_time:.2f}s")
        
        conf = float(roi_pack.get("confidence", 0) or 0)
        # rank by confidence only now
        rank_score = conf
        candidates.append({
            "tile_index": idx,
            "tile_offset": offset,
            "result": roi_pack,
            "rank_score": rank_score,
            "tile_img": tile_img,
        })
        
        # Early termination if high confidence found
        if conf >= early_termination_threshold:
            if not json_only:
                print(f"🎯 Early termination: Found high confidence match ({conf:.3f}) at tile {idx+1}")
            break
    
    inference_time = time.time() - inference_start

    candidates = sorted(candidates, key=lambda x: x.get("rank_score", 0), reverse=True)

    if json_only:
        if not candidates:
            print(json.dumps({
                "success": False,
                "error": "no_candidates",
                "message": "No candidates found",
            }))
            return
        best = candidates[0]
        point = compute_global_point_from_candidate(best)
        # Generate camera position visualization
        if not no_viz:
            camera_pos_save = draw_camera_position_on_map(
                map_img, 
                point["x"], 
                point["y"], 
                point["roi_confidence"],
                camera_name
            )
            print(f"→ Camera position visualization saved: {camera_pos_save}")
        
        total_time = time.time() - start_time
        
        output = {
            "success": True,
            "position": {"x": point["x"], "y": point["y"]},
            "bounding_box": point["bounding_box"],
            "tile_index": point["tile_index"],
            "tile_offset": point["tile_offset"],
            "roi_confidence": point["roi_confidence"],
            "rank_score": point["rank_score"],
            "timing": {
                "model_load_time": round(model_load_time, 2),
                "inference_time": round(inference_time, 2),
                "total_time": round(total_time, 2),
                "tiles_processed": total_tiles,
                "avg_time_per_tile": round(inference_time / total_tiles, 2) if total_tiles > 0 else 0
            },
            "meta": {
                "model_id": MODEL_ID,
                "tile_size": tile_size,
                "stride": stride,
                "samples": samples,
                "temperature": temperature,
                "top_p": top_p,
            }
        }
        print(json.dumps(output))
        return

    print("\n[Step 3] Top candidate ROIs (blended rank: 0.6*ROI_conf + 0.4*NL_score):")
    if not candidates:
        print("⚠️ No candidates found.")
    else:
        for i, c in enumerate(candidates[:top_k]):
            printable = {k: v for k, v in c.items() if k not in ("tile_img",)}
            print(json.dumps(printable, indent=2))

        if not no_viz:
            # Camera position visualization
            best = candidates[0]
            point = compute_global_point_from_candidate(best)
            camera_pos_save = draw_camera_position_on_map(
                map_img, 
                point["x"], 
                point["y"], 
                point["roi_confidence"],
                camera_name
            )
            print(f"→ Camera position visualization saved: {camera_pos_save}")

    total_time = time.time() - start_time
    print(f"\n[Pipeline finished] — Total time: {total_time:.2f}s")
    print(f"Model load: {model_load_time:.2f}s, Inference: {inference_time:.2f}s")
    print("Next step is fine geometric alignment with CV matcher (e.g., LoFTR).")

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera-to-Map ROI Localization")
    parser.add_argument("--camera", type=str, default="camera.jpg", help="Path to camera image")
    parser.add_argument("--map", dest="map_path", type=str, default="map.jpg", help="Path to map image")
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size for map tiling")
    parser.add_argument("--stride", type=int, default=512, help="Stride for map tiling")
    parser.add_argument("--json", dest="json_only", action="store_true", help="Output JSON with global (x,y) only")
    parser.add_argument("--no-viz", action="store_true", help="Disable saving visualization images")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top candidates to show/visualize")
    parser.add_argument("--samples", type=int, default=3, help="Number of stochastic samples per tile")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--early-termination", type=float, default=0.85, help="Stop processing when confidence >= this threshold")
    parser.add_argument("--max-tiles", type=int, default=None, help="Maximum number of tiles to process (for speed)")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode (1 sample, deterministic, shorter output)")
    parser.add_argument("--max-image-size", type=int, default=None, help="Resize images if larger than this (for speed)")
    args = parser.parse_args()

    # If strict JSON mode, reduce logs
    if args.json_only:
        DEBUG = False

    main(
        args.camera,
        args.map_path,
        tile_size=args.tile_size,
        stride=args.stride,
        json_only=args.json_only,
        no_viz=args.no_viz,
        top_k=args.top_k,
        samples=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        early_termination_threshold=args.early_termination,
        max_tiles=args.max_tiles,
        fast_mode=args.fast,
        max_image_size=args.max_image_size,
    )
