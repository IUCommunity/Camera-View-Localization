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
from typing import List, Tuple, Dict, Any
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


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

def draw_bbox_on_tile(tile_img: Image.Image, bbox: List[int], label: str, save_path: str):
    img = tile_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        draw.rectangle(bbox, outline=(255, 0, 0), width=4)
        # label background
        x0, y0 = bbox[0], max(0, bbox[1] - 18)
        text = label
        if font:
            tw, th = draw.textsize(text, font=font)
        else:
            tw, th = draw.textsize(text)
        draw.rectangle([x0, y0, x0 + tw + 6, y0 + th + 4], fill=(255, 0, 0))
        draw.text((x0 + 3, y0 + 2), text, fill=(255, 255, 255), font=font)
    img.save(save_path)


def draw_rois_on_full_map(full_map: Image.Image, candidates: List[Dict[str, Any]], top_k: int = 3, save_path: str = None):
    img = full_map.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, c in enumerate(candidates[:top_k]):
        offset = c.get("tile_offset", (0, 0))
        bbox = c.get("result", {}).get("bounding_box", None)
        conf = c.get("result", {}).get("confidence", 0)
        if bbox and len(bbox) == 4:
            # project to full-map coords
            b = [bbox[0] + offset[0], bbox[1] + offset[1], bbox[2] + offset[0], bbox[3] + offset[1]]
            color = (255, 0, 0) if i == 0 else (255, 165, 0)
            draw.rectangle(b, outline=color, width=4)
            label = f"#{i+1} conf={conf:.2f}"
            x0, y0 = b[0], max(0, b[1] - 18)
            if font:
                tw, th = draw.textsize(label, font=font)
            else:
                tw, th = draw.textsize(label)
            draw.rectangle([x0, y0, x0 + tw + 6, y0 + th + 4], fill=color)
            draw.text((x0 + 3, y0 + 2), label, fill=(255, 255, 255), font=font)
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "full_map_with_rois.png")
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

def main(camera_path: str, map_path: str, *, tile_size: int = 1024, stride: int = 512, json_only: bool = False, no_viz: bool = False, top_k: int = 3, samples: int = 3, temperature: float = 0.2, top_p: float = 0.9):
    processor, model = load_model()

    cam_img = load_image(camera_path)
    map_img = load_image(map_path)

    if not json_only:
        print("[Step 1] Splitting map into tiles and localizing camera on each tile...")
    tiles = split_into_tiles(map_img, tile_size=tile_size, stride=stride)
    if not json_only:
        print(f"Generated {len(tiles)} map tiles.")

    candidates = []
    for idx, (tile_img, offset) in enumerate(tiles):
        roi_pack = localize_camera_on_tile(
            processor,
            model,
            cam_img,
            tile_img,
            samples=samples,
            temperature=temperature,
            top_p=top_p,
        )
        if DEBUG and not json_only:
            print(f"[TILE {idx}] ROI pack:", roi_pack)
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
        output = {
            "success": True,
            "position": {"x": point["x"], "y": point["y"]},
            "bounding_box": point["bounding_box"],
            "tile_index": point["tile_index"],
            "tile_offset": point["tile_offset"],
            "roi_confidence": point["roi_confidence"],
            "rank_score": point["rank_score"],
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
                # per-tile visualization
                bbox = c["result"].get("bounding_box")
                conf = c["result"].get("confidence", 0)
                label = f"#{i+1} conf={float(conf or 0):.2f}"
                tile_save = os.path.join(OUTPUT_DIR, f"candidate_tile_{i+1}.png")
                draw_bbox_on_tile(c["tile_img"], bbox, label, tile_save)
                print(f"→ Tile ROI saved: {tile_save}")

        if not no_viz:
            # full map visualization
            full_map_save = draw_rois_on_full_map(map_img, candidates, top_k=min(top_k, len(candidates)))
            print(f"→ Full map with ROIs saved: {full_map_save}")

    print("\n[Pipeline finished] — next step is fine geometric alignment with CV matcher (e.g., LoFTR).")

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
    )
