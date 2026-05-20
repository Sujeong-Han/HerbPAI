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


def handle(files, img_size, conf_thres, save_crops):
    if not files:
        return [], None, "No images uploaded.", []

    images = [Image.open(f).convert("RGB") for f in files]
    all_results = []
    all_crops = []
    all_stats = {"count": 0, "classes": {}}
    zip_tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)

    with zipfile.ZipFile(zip_tmp.name, "w") as zf:
        for idx, pil_image in enumerate(images):
            result_img, crops, stats = run_detection(pil_image, int(img_size), float(conf_thres))

            tmp_result = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            result_img.save(tmp_result.name)
            zf.write(tmp_result.name, f"results/result_{idx+1:02d}.png")
            all_results.append((result_img, f"result_{idx+1:02d}"))

            if save_crops:
                for i, (lbl, crop_pil) in enumerate(crops):
                    safe_label = lbl.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
                    fname = f"crops/img{idx+1:02d}_crop{i+1:02d}_{safe_label}.png"
                    crop_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    crop_pil.save(crop_tmp.name)
                    zf.write(crop_tmp.name, fname)
                    all_crops.append((crop_pil, lbl))

            all_stats["count"] += stats["count"]
            for k, v in stats["classes"].items():
                all_stats["classes"][k] = all_stats["classes"].get(k, 0) + v

    class_summary = ", ".join([f"{k}: {v}" for k, v in all_stats["classes"].items()])
    stats_text = (
        f"✅ Detection complete — {len(images)} image(s)\n"
        f"📦 Total objects detected: {all_stats['count']}\n"
        f"🔍 Classes: {class_summary if class_summary else 'none'}"
    )

    return all_results, zip_tmp.name, stats_text, all_crops


with gr.Blocks(title="HerbPAI") as demo:
    gr.HTML("""
    <div style='text-align:center; padding:2rem 1rem 1rem'>
        <h1 style='font-size:2rem; font-weight:300; color:#1a3a0a'>
            <span style='font-weight:500; color:#3B6D11'>Herb</span>PAI
        </h1>
        <p style='color:#5a7a4a'>Herbarium Image Preprocessing for AI Identification</p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("🌿 Detection"):
            with gr.Row():
                with gr.Column():
                    input_images = gr.File(
                        label="Upload herbarium specimen images (single or batch)",
                        file_count="multiple",
                        file_types=["image"]
                    )
                    with gr.Row():
                        img_size = gr.Dropdown(choices=[640, 1280], value=640, label="Image size")
                        conf_thres = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Confidence")
                    save_crops = gr.Checkbox(value=True, label="Save detected crops separately")
                    run_btn = gr.Button("▶ Run detection", variant="primary")

                with gr.Column():
                    output_gallery = gr.Gallery(
                        label="Result images (non-plant components removed)",
                        columns=2,
                        height=300
                    )
                    stats_text = gr.Textbox(label="Detection stats", lines=3, interactive=False)
                    download_all = gr.File(label="⬇ Download all results + crops (ZIP)")

            crop_gallery = gr.Gallery(label="Detected crops", columns=6, height=180)

            run_btn.click(
                fn=handle,
                inputs=[input_images, img_size, conf_thres, save_crops],
                outputs=[output_gallery, download_all, stats_text, crop_gallery]
            )

        with gr.Tab("📖 How to use"):
            gr.Markdown("""
## How to use HerbPAI
1. **Upload** one or more herbarium specimen images (JPG, PNG, TIFF)
2. **Adjust** image size and confidence threshold if needed
3. Click **▶ Run detection**
4. **Download** the ZIP containing all result images and individual crops

### What gets detected?
- **Non-plant components** (Ruler, Label, Palette, DB_stamp, Annotation_label, Institution_stamp) → filled white
- **Specimen** → kept, everything else becomes white

### Tips
- Use **1280px** for high-resolution scans
- Lower confidence if some objects are missed
- All results and crops are saved in the ZIP file
            """)

if __name__ == "__main__":
    demo.launch(share=True)
