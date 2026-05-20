import gradio as gr
import torch
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import zipfile
import os

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "best.pt"
sys.path.insert(0, str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.augmentations import letterbox

device = select_device("cpu")
model = DetectMultiBackend(str(WEIGHTS), device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size((640, 640), s=stride)
model.warmup(imgsz=(1, 3, *imgsz))

def run_detection(pil_image, img_size=640, conf_thres=0.25):
    img_size = check_img_size(img_size, s=model.stride)
    img0 = np.array(pil_image.convert("RGB"))[:, :, ::-1].copy()
    img = letterbox(img0, img_size, stride=model.stride, auto=pt)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred[0] if isinstance(pred, list) else pred, conf_thres, 0.45, max_det=300)
    im0 = img0.copy()
    imc = img0.copy()
    crops = []
    stats = {"count": 0, "classes": {}}
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                label = names[int(cls)]
                conf_val = float(conf)
                if label != 'Specimen':
                    im0[y1:y2, x1:x2] = 255
                else:
                    mask = 255 * np.ones_like(im0)
                    mask[y1:y2, x1:x2] = im0[y1:y2, x1:x2]
                    im0 = mask
                crop = imc[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append((label, conf_val, Image.fromarray(crop[:, :, ::-1])))
                stats["count"] += 1
                stats["classes"][label] = stats["classes"].get(label, 0) + 1
    return Image.fromarray(im0[:, :, ::-1]), crops, stats

def handle(image, img_size, conf_thres, save_crops):
    if image is None:
        return None, "No image uploaded.", None
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    result_img, crops, stats = run_detection(image, int(img_size), float(conf_thres))
    class_summary = ", ".join([f"{k}: {v}" for k, v in stats["classes"].items()])
    stats_text = f"✅ Detection complete\n📦 Objects detected: {stats['count']}\n🔍 Classes: {class_summary if class_summary else 'none'}"
    zip_path = None
    if save_crops and crops:
        zip_tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        with zipfile.ZipFile(zip_tmp.name, "w") as zf:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            result_img.save(tmp.name)
            zf.write(tmp.name, "result.png")
            for i, (lbl, conf_val, crop_pil) in enumerate(crops):
                safe = lbl.replace(" ", "_")
                ct = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                crop_pil.save(ct.name)
                zf.write(ct.name, f"crops/crop_{i+1:02d}_{safe}.png")
        zip_path = zip_tmp.name
    return result_img, stats_text, zip_path

with gr.Blocks(title="HerbPAI") as demo:
    gr.HTML("<div style='text-align:center;padding:2rem 1rem 1rem'><h1 style='font-size:2rem;font-weight:300;color:#1a3a0a'><span style='font-weight:500;color:#3B6D11'>Herb</span>PAI</h1><p style='color:#5a7a4a'>Herbarium Image Preprocessing for AI Identification</p></div>")
    with gr.Tab("🌿 Detection"):
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload herbarium specimen image")
                img_size = gr.Dropdown(choices=[640, 1280], value=640, label="Image size")
                conf_thres = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence threshold")
                save_crops = gr.Checkbox(value=True, label="Save detected crops separately")
                run_btn = gr.Button("▶ Run detection", variant="primary")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Result (non-plant components removed)")
                stats_out = gr.Textbox(label="Detection stats", lines=3, interactive=False)
                zip_out = gr.File(label="⬇ Download results (ZIP)")
        run_btn.click(fn=handle,
                     inputs=[input_image, img_size, conf_thres, save_crops],
                     outputs=[output_image, stats_out, zip_out])
    with gr.Tab("📖 How to use"):
        gr.Markdown("""
## How to use HerbPAI
1. Upload a herbarium specimen image
2. Click **▶ Run detection**
3. Download the ZIP with results and crops

### What gets detected?
Non-plant components (Ruler, Label, Palette, DB_stamp etc.) are filled white.
Specimen area is kept.
        """)

demo.launch(share=True)
