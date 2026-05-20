import gradio as gr
import torch
import sys
import numpy as np
from pathlib import Path
from PIL import Image

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
model.warmup(imgsz=(1, 3, 640, 640))

def handle(image, img_size, conf_thres):
    if image is None:
        return None, "No image uploaded."
    img_size = int(img_size)
    conf_thres = float(conf_thres)
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    img0 = np.array(image.convert("RGB"))[:, :, ::-1].copy()
    img = letterbox(img0, img_size, stride=model.stride, auto=pt)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred[0] if isinstance(pred, list) else pred, conf_thres, 0.45, max_det=300)
    im0 = img0.copy()
    stats = {"count": 0, "classes": {}}
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                label = names[int(cls)]
                if label != 'Specimen':
                    im0[y1:y2, x1:x2] = 255
                else:
                    mask = 255 * np.ones_like(im0)
                    mask[y1:y2, x1:x2] = im0[y1:y2, x1:x2]
                    im0 = mask
                stats["count"] += 1
                stats["classes"][label] = stats["classes"].get(label, 0) + 1
    result = Image.fromarray(im0[:, :, ::-1])
    summary = ", ".join([f"{k}: {v}" for k, v in stats["classes"].items()])
    stats_text = f"✅ Detection complete\n📦 Objects: {stats['count']}\n🔍 {summary if summary else 'none'}"
    return result, stats_text

with gr.Blocks(title="HerbPAI") as demo:
    gr.HTML("<div style='text-align:center;padding:2rem'><h1 style='color:#3B6D11'>HerbPAI</h1><p>Herbarium Image Preprocessing for AI</p></div>")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload image")
            img_size = gr.Radio(choices=["640", "1280"], value="640", label="Image size")
            conf_thres = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence")
            run_btn = gr.Button("▶ Run detection", variant="primary")
        with gr.Column():
            output_image = gr.Image(type="pil", label="Result")
            stats_out = gr.Textbox(label="Stats", lines=3, interactive=False)
    run_btn.click(fn=handle, inputs=[input_image, img_size, conf_thres], outputs=[output_image, stats_out])

demo.launch(share=True)
