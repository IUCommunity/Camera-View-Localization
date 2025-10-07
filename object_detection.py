import os
import json
import re
import argparse
import time
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Configuration
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def extract_json(response: str) -> dict:
    """Extract JSON from model response"""
    # Remove assistant prefix if present
    if "assistant" in response.lower():
        parts = response.split("assistant", 1)
        if len(parts) > 1:
            response = parts[-1]
    
    # Try to find JSON in code blocks
    matches = re.findall(r"```(?:json)?\s*(.*?)```", response, re.DOTALL)
    if matches:
        candidate = matches[-1].strip()
    else:
        # Find JSON object
        matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
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
        except:
            return {}

def detect_objects(processor, model, image: Image.Image) -> dict:
    """Detect objects in image and return bounding boxes"""
    
    prompt = """Analyze this image and detect all objects. For each object, provide:
1. Object class/name
2. Bounding box coordinates [x_min, y_min, x_max, y_max]
3. Confidence score

Return ONLY valid JSON in this exact format:
{
    "objects": [
        {
            "class": "car",
            "bbox": [100, 150, 300, 250],
            "confidence": 0.95
        },
        {
            "class": "person", 
            "bbox": [50, 200, 120, 400],
            "confidence": 0.88
        }
    ]
}

Detect all visible objects including: vehicles, people, animals, buildings, signs, trees, etc.
Use pixel coordinates where (0,0) is top-left corner."""

    chat = [
        {
            "role": "system",
            "content": "You are an object detection system. Output only valid JSON."
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    chat_str = processor.apply_chat_template(chat, add_generation_prompt=True)
    inputs = processor(
        text=chat_str,
        images=[image],
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
    
    response = processor.decode(output[0], skip_special_tokens=True)
    result = extract_json(response)
    
    return result

def draw_bounding_boxes(image: Image.Image, detections: dict) -> Image.Image:
    """Draw bounding boxes on the image"""
    
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Try to load a font
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    objects = detections.get("objects", [])
    
    # Colors for different object types
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green  
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
    ]
    
    for i, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
            
        bbox = obj.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
            
        try:
            x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
        except (ValueError, TypeError):
            continue
        
        # Clamp coordinates to image bounds
        x_min = max(0, min(x_min, img.width - 1))
        y_min = max(0, min(y_min, img.height - 1))
        x_max = max(0, min(x_max, img.width - 1))
        y_max = max(0, min(y_max, img.height - 1))
        
        # Skip invalid boxes
        if x_min >= x_max or y_min >= y_max:
            continue
        
        # Choose color
        color = colors[i % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
        
        # Draw label
        class_name = obj.get("class", f"object_{i}")
        confidence = obj.get("confidence", 0.0)
        label = f"{class_name}: {confidence:.2f}"
        
        # Calculate text size
        if font:
            try:
                bbox_text = draw.textbbox((0, 0), label, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except AttributeError:
                text_width, text_height = draw.textsize(label, font=font)
        else:
            try:
                bbox_text = draw.textbbox((0, 0), label)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except AttributeError:
                text_width, text_height = 100, 20
        
        # Draw label background
        label_x = x_min
        label_y = max(0, y_min - text_height - 5)
        
        draw.rectangle(
            [label_x, label_y, label_x + text_width + 10, label_y + text_height + 5],
            fill=color,
            outline=color
        )
        
        # Draw label text
        draw.text((label_x + 5, label_y + 2), label, fill=(255, 255, 255), font=font)
    
    return img

def main(input_path: str, output_path: str = None):
    """Main function: input image -> output image with bounding boxes"""
    
    total_start = time.time()
    
    # Load model
    print("Loading model...")
    model_start = time.time()
    processor, model = load_model()
    model_time = time.time() - model_start
    print(f"Model loaded in {model_time:.2f} seconds.")
    
    # Load input image
    print(f"Loading image: {input_path}")
    image = Image.open(input_path).convert("RGB")
    print(f"Image size: {image.size}")
    
    # Detect objects
    print("Detecting objects...")
    inference_start = time.time()
    detections = detect_objects(processor, model, image)
    inference_time = time.time() - inference_start
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    # Handle different response formats
    if isinstance(detections, dict):
        objects = detections.get('objects', [])
    elif isinstance(detections, list):
        objects = detections
    else:
        objects = []
    
    print(f"Detected {len(objects)} objects")
    
    # Draw bounding boxes
    print("Drawing bounding boxes...")
    # Create a proper detections dict for the drawing function
    detections_dict = {"objects": objects}
    result_image = draw_bounding_boxes(image, detections_dict)
    
    # Save output
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_detected{ext}"
    
    result_image.save(output_path)
    
    total_time = time.time() - total_start
    print(f"Output saved: {output_path}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Breakdown: Model load: {model_time:.2f}s, Inference: {inference_time:.2f}s, Drawing: {total_time - model_time - inference_time:.2f}s")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object detection with bounding boxes")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", help="Output image path (optional)")
    
    args = parser.parse_args()
    
    main(args.input, args.output)