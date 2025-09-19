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
# NL Descriptions (ignore weather/season/transients)
# -------------------------

def describe_camera_nl(processor, model, cam_img: Image.Image) -> str:
    prompt = (
        "You are a meticulous scene describer. Describe the CAMERA image in natural language.\n"
        "Focus ONLY on structural, spatial, and semantic cues that persist on maps: roads, intersections, building facades, footprints, gates/fences, sidewalks, crosswalks, traffic lights/signs, permanent landmarks.\n"
        "IGNORE weather, lighting, time of day, shadows, reflections, clouds, fog/snow/rain, and seasonal vegetation changes.\n"
        "Use concise sentences; include relative positions (left/center/right/top/bottom) and relations (next to, across from, corner of)."
    )
    chat = [
        {"role": "system", "content": "You produce detailed yet concise scene descriptions. No JSON."},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]},
    ]
    chat_str = processor.apply_chat_template(chat, add_generation_prompt=True)
    inputs = processor(text=chat_str, images=[cam_img], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=350)
    resp = processor.decode(out[0], skip_special_tokens=True)
    # strip any role echoes
    if "assistant" in resp:
        resp = resp.split("assistant", 1)[-1].strip()
    return resp.strip()


def describe_map_tile_nl(processor, model, map_tile: Image.Image) -> str:
    prompt = (
        "You are a cartography analyst. Describe the MAP_TILE in natural language.\n"
        "Focus on map features: road layout (orientation, intersections, corners), building footprints/shapes, parking lots, plazas, fences/gates, notable landmarks (banks, stores, schools), open areas.\n"
        "IGNORE weather/season, lighting, shadows.\n"
        "Use relative positions within the tile (north/south/east/west/center) and relations (adjacent to, at corner, across)."
    )
    chat = [
        {"role": "system", "content": "You describe map tiles clearly. No JSON."},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]},
    ]
    chat_str = processor.apply_chat_template(chat, add_generation_prompt=True)
    inputs = processor(text=chat_str, images=[map_tile], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=350)
    resp = processor.decode(out[0], skip_special_tokens=True)
    if "assistant" in resp:
        resp = resp.split("assistant", 1)[-1].strip()
    return resp.strip()

# -------------------------
# Text-only Semantic Matching (NL↔NL) → score & rationale
# -------------------------

def match_descriptions_nl(processor, model, cam_nl: str, map_nl: str) -> Dict[str, Any]:
    prompt = f"""
    You are a spatial reasoning assistant. Compare a ground-level CAMERA description with an aerial MAP_TILE description.
    The camera view is a partial subset of the map. Ignore weather/seasonal/lighting details.

    CAMERA (natural language):\n{cam_nl}\n\nMAP_TILE (natural language):\n{map_nl}

    Task: Decide how well the map tile can contain the camera view as a subregion. Return ONLY valid JSON:
    {{
      "match": true/false,
      "score": float,   // 0..1 semantic fit
      "rationale": "short explanation"
    }}
    """
    chat = [
        {"role": "system", "content": "You compare NL descriptions and output JSON only."},
        {"role": "user", "content": prompt},
    ]
    chat_str = processor.apply_chat_template(chat, add_generation_prompt=True)
    inputs = processor(text=chat_str, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=220)
    resp = processor.decode(out[0], skip_special_tokens=True)
    return safe_json_parse(resp)

# -------------------------
# ROI Extraction (JSON) from NL + MAP_TILE image
# -------------------------

def roi_from_descriptions(processor, model, cam_nl: str, map_nl: str, map_tile: Image.Image) -> Dict[str, Any]:
    prompt = f"""
    You are a spatial localizer. Given a MAP_TILE image and the following NL descriptions:
    - CAMERA: {cam_nl}
    - MAP_TILE: {map_nl}

    The camera view is a partial subset of the map tile. Use persistent structures (roads, intersections, buildings, gates, sidewalks, traffic lights/signs). Ignore weather/season/shadows/reflections.

    Return ONLY valid JSON with a best-guess ROI in MAP_TILE pixel coordinates:
    {{
      "match": true/false,
      "rationale": "...",
      "bounding_box": [x_min, y_min, x_max, y_max],
      "confidence": float
    }}
    """
    chat = [
        {"role": "system", "content": "You output JSON only, with a bounding box ROI."},
        {"role": "user", "content": [
            {"type": "image"},  # map tile image for pixel coordinates
            {"type": "text", "text": prompt},
        ]},
    ]
    chat_str = processor.apply_chat_template(chat, add_generation_prompt=True)
    inputs = processor(text=chat_str, images=[map_tile], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=280)
    resp = processor.decode(out[0], skip_special_tokens=True)
    return safe_json_parse(resp)

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
# Main Pipeline
# -------------------------

def main(camera_path: str, map_path: str):
    processor, model = load_model()

    cam_img = load_image(camera_path)
    map_img = load_image(map_path)

    print("[Step 1] Describing camera (natural language, no weather)...")
    cam_nl = describe_camera_nl(processor, model, cam_img)
    if DEBUG:
        print("Camera NL:\n", cam_nl)

    print("[Step 2] Splitting map into tiles and describing (natural language, no weather)...")
    tiles = split_into_tiles(map_img, tile_size=1024, stride=512)
    print(f"Generated {len(tiles)} map tiles.")

    candidates = []
    for idx, (tile_img, offset) in enumerate(tiles):
        map_nl = describe_map_tile_nl(processor, model, tile_img)
        if DEBUG:
            print(f"\n[TILE {idx}] Map NL:\n", map_nl)
        # NL-only semantic matching
        score_pack = match_descriptions_nl(processor, model, cam_nl, map_nl)
        if DEBUG:
            print(f"[TILE {idx}] Semantic match:", score_pack)
        # Always attempt ROI from descriptions + tile image (best guess)
        roi_pack = roi_from_descriptions(processor, model, cam_nl, map_nl, tile_img)
        if DEBUG:
            print(f"[TILE {idx}] ROI pack:", roi_pack)
        # combine info
        conf = float(roi_pack.get("confidence", 0) or 0)
        sem_score = float(score_pack.get("score", 0) or 0)
        # simple blended rank (tweak as needed)
        rank_score = 0.6 * conf + 0.4 * sem_score
        candidates.append({
            "tile_index": idx,
            "tile_offset": offset,
            "camera_nl": cam_nl,
            "map_nl": map_nl,
            "semantic": score_pack,
            "result": roi_pack,
            "rank_score": rank_score,
            "tile_img": tile_img,
        })

    candidates = sorted(candidates, key=lambda x: x.get("rank_score", 0), reverse=True)

    print("\n[Step 3] Top candidate ROIs (blended rank: 0.6*ROI_conf + 0.4*NL_score):")
    if not candidates:
        print("⚠️ No candidates found.")
    else:
        for i, c in enumerate(candidates[:3]):
            printable = {k: v for k, v in c.items() if k not in ("tile_img", "camera_nl", "map_nl")}
            print(json.dumps(printable, indent=2))
            # per-tile visualization
            bbox = c["result"].get("bounding_box")
            conf = c["result"].get("confidence", 0)
            label = f"#{i+1} conf={float(conf or 0):.2f}"
            tile_save = os.path.join(OUTPUT_DIR, f"candidate_tile_{i+1}.png")
            draw_bbox_on_tile(c["tile_img"], bbox, label, tile_save)
            print(f"→ Tile ROI saved: {tile_save}")

        # full map visualization
        full_map_save = draw_rois_on_full_map(map_img, candidates, top_k=min(3, len(candidates)))
        print(f"→ Full map with ROIs saved: {full_map_save}")

    print("\n[Pipeline finished] — next step is fine geometric alignment with CV matcher (e.g., LoFTR).")

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    camera_path = "camera.jpg"
    map_path = "map.jpg"  # or .png
    main(camera_path, map_path)
