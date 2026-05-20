import gradio as gr
import torch
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import zipfile

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
                    crop_pil = Image.fromarray(crop[:, :, ::-1])
                    crops.append((f"{label} ({conf_val:.0%})", crop_pil))
                stats["count"] += 1
                stats["classes"][label] = stats["classes"].get(label, 0) + 1
    result_pil = Image.fromarray(im0[:, :, ::-1])
    return result_pil, crops, stats

def handle(image, img_size, conf_thres, save_crops):
    if image is None:
        return None, "No image uploaded.", []
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    result_img, crops, stats = run_detection(image, int(img_size), float(conf_thres))
    zip_tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    with zipfile.ZipFile(zip_tmp.name, "w") as zf:
        tmp_result = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        result_img.save(tmp_result.name)
        zf.write(tmp_result.name, "result.png")
        crop_imgs = []
        if save_crops:
            for i, (lbl, crop_pil) in enumerate(crops):
                safe_label = lbl.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
                fname = f"crops/crop_{i+1:02d}_{safe_label}.png"
                crop_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                crop_pil.save(crop_tmp.name)
                zf.write(crop_tmp.name, fname)
                crop_imgs.append(crop_pil)
    class_summary = ", ".join([f"{k}: {v}" for k, v in stats["classes"].items()])
    stats_text = f"Detection complete\nObjects detected: {stats['count']}\nClasses: {class_summary if class_summary else 'none'}"
    return result_img, stats_text, crop_imgs

with gr.Blocks(title="HerbPAI") as demo:
    gr.HTML("<div style='text-align:center;padding:2rem 1rem 1rem'><h1 style='font-size:2rem;font-weight:300;color:#1a3a0a'><span style='font-weight:500;color:#3B6D11'>Herb</span>PAI</h1><p style='color:#5a7a4a'>Herbarium Image Preprocessing for AI Identification</p></div>")
    with gr.Tabs():
        with gr.Tab("Detection"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Upload herbarium specimen image")
                    with gr.Row():
                        img_size = gr.Dropdown(choices=[640, 1280], value=640, label="Image size")
                        conf_thres = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence")
                    save_crops = gr.Checkbox(value=True, label="Save detected crops separately")
                    run_btn = gr.Button("Run detection", variant="primary")
                with gr.Column():
                    output_image = gr.Image(label="Result (non-plant components removed)")
                    stats_text = gr.Textbox(label="Detection stats", lines=3, interactive=False)
            crop_gallery = gr.Textbox(label="Detected crop labels", interactive=False)
            run_btn.click(fn=handle, inputs=[input_image, img_size, conf_thres, save_crops],
                         outputs=[output_image, stats_text, crop_gallery])
        with gr.Tab("How to use"):
            gr.Markdown("""
## How to use HerbPAI
1. Upload a herbarium specimen image
2. Adjust settings if needed
3. Click **Run detection**

### What gets detected?
- Non-plant components (Ruler, Label, Palette, DB_stamp etc.) are filled white
- Specimen area is kept, everything else becomes white
            """)

if __name__ == "__main__":
    demo.launch(share=True)
