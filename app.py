import fix_gradio_client  # patch BEFORE importing gradio

import gradio as gr
import torch
import sys
import numpy as np
import os
import zipfile
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

ROOT = Path(__file__).parent
WEIGHTS = ROOT / "best.pt"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.augmentations import letterbox

device = select_device("cpu")
model = DetectMultiBackend(str(WEIGHTS), device=device)
stride, names, pt = model.stride, model.names, model.pt
model.warmup(imgsz=(1, 3, 640, 640))

SPECIMEN_CLASS = "Specimen"

def detect_single(pil_image, save_dir=None, filename=None):
    if pil_image is None:
        return None, None, {}
    if not isinstance(pil_image, Image.Image):
        pil_image = Image.fromarray(pil_image).convert("RGB")
    img0 = np.array(pil_image.convert("RGB"))[:, :, ::-1].copy()
    img = letterbox(img0, 640, stride=model.stride, auto=pt)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(
        pred[0] if isinstance(pred, list) else pred,
        0.25, 0.45, max_det=300
    )
    stats = {"count": 0, "classes": {}}
    non_specimen_boxes = []
    specimen_boxes = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                label = names[int(cls)]
                stats["count"] += 1
                stats["classes"][label] = stats["classes"].get(label, 0) + 1
                if label == SPECIMEN_CLASS:
                    specimen_boxes.append((x1, y1, x2, y2, float(conf)))
                else:
                    non_specimen_boxes.append((x1, y1, x2, y2, label, float(conf)))
    labeled_img = img0.copy()
    labeled_pil = Image.fromarray(labeled_img[:, :, ::-1])
    draw = ImageDraw.Draw(labeled_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()
    palette = [
        (220, 53, 69),
        (255, 140, 0),
        (32, 178, 170),
        (106, 90, 205),
        (34, 139, 34),
    ]
    for idx, (x1, y1, x2, y2, label, conf) in enumerate(non_specimen_boxes):
        color = palette[idx % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = f"{label} {conf:.2f}"
        draw.rectangle([x1, y1 - 22, x1 + len(label_text) * 10, y1], fill=color)
        draw.text((x1 + 2, y1 - 20), label_text, fill="white", font=font)
    for (x1, y1, x2, y2, conf) in specimen_boxes:
        color = (59, 109, 17)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label_text = f"Specimen {conf:.2f}"
        draw.rectangle([x1, y1 - 22, x1 + len(label_text) * 10, y1], fill=color)
        draw.text((x1 + 2, y1 - 20), label_text, fill="white", font=font)
    final_pil = None
    if specimen_boxes:
        largest = max(specimen_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        x1, y1, x2, y2, _ = largest
        result_img = img0.copy()
        for (bx1, by1, bx2, by2, blabel, _) in non_specimen_boxes:
            result_img[by1:by2, bx1:bx2] = 255
        white_canvas = 255 * np.ones_like(result_img)
        white_canvas[y1:y2, x1:x2] = result_img[y1:y2, x1:x2]
        final_pil = Image.fromarray(white_canvas[:, :, ::-1])
    else:
        result_img = img0.copy()
        for (bx1, by1, bx2, by2, blabel, _) in non_specimen_boxes:
            result_img[by1:by2, bx1:bx2] = 255
        final_pil = Image.fromarray(result_img[:, :, ::-1])
    if save_dir and filename:
        stem = Path(filename).stem
        labeled_path = Path(save_dir) / f"{stem}_labeled.png"
        result_path = Path(save_dir) / f"{stem}_result.png"
        labeled_pil.save(str(labeled_path), format="PNG")
        final_pil.save(str(result_path), format="PNG")
    return labeled_pil, final_pil, stats

def handle_batch(files):
    if not files:
        return [], [], "❌ 이미지를 업로드해주세요.", None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = RESULTS_DIR / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    labeled_gallery = []
    result_gallery = []
    all_stats = []
    total_count = 0
    for i, file_obj in enumerate(files):
        filename = f"image_{i+1}.jpg"
        try:
            if hasattr(file_obj, 'name'):
                filepath = file_obj.name
                filename = Path(filepath).name
                pil_img = Image.open(filepath).convert("RGB")
            elif isinstance(file_obj, str):
                filepath = file_obj
                filename = Path(filepath).name
                pil_img = Image.open(filepath).convert("RGB")
            else:
                continue
            labeled, result, stats = detect_single(pil_img, save_dir=save_dir, filename=filename)
            if labeled:
                labeled_gallery.append(labeled)
            if result:
                result_gallery.append(result)
            class_summary = ", ".join([f"{k}: {v}" for k, v in stats.get("classes", {}).items()])
            all_stats.append(
                f"🖼️ {filename}\n"
                f"   📦 감지 객체: {stats.get('count', 0)}개  |  {class_summary or '없음'}"
            )
            total_count += stats.get("count", 0)
        except Exception as e:
            all_stats.append(f"⚠️ {filename}: 처리 중 오류 - {str(e)}")
    zip_path = None
    saved_files = list(save_dir.glob("*.png"))
    if saved_files:
        zip_path = str(save_dir / "HerbPAI_results.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for f in saved_files:
                zf.write(f, f.name)
    stats_text = (
        f"✅ 처리 완료: {len(files)}장\n"
        f"📦 전체 감지 객체: {total_count}개\n"
        f"💾 저장 위치: results/{timestamp}/\n\n"
        + "\n".join(all_stats)
    )
    return labeled_gallery, result_gallery, stats_text, zip_path

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');
:root {
    --green-dark:  #2C5010;
    --green-mid:   #3B6D11;
    --green-light: #6B9E3A;
    --cream:       #F7F4EE;
    --warm-white:  #FDFCF9;
    --border:      #D8D0C0;
    --text-main:   #2A2A22;
    --text-muted:  #7A7060;
}
body, .gradio-container {
    background: var(--cream) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-main);
}
.hero {
    text-align: center;
    padding: 3rem 2rem 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: var(--green-dark);
    letter-spacing: -1px;
    margin: 0 0 0.4rem;
}
.hero p {
    font-size: 1rem;
    color: var(--text-muted);
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.section-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: var(--green-mid);
    margin-bottom: 0.5rem;
}
.guide-box {
    background: var(--warm-white);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-size: 0.88rem;
    color: var(--text-muted);
    line-height: 1.8;
    margin-bottom: 1rem;
}
.guide-box strong { color: var(--green-mid); }
footer { display: none !important; }
"""

GUIDE_HTML = """
<div class="guide-box">
    <strong>사용 방법</strong><br>
    ① 아래 업로드 영역에 이미지를 <strong>여러 장</strong> 한꺼번에 드래그하거나 클릭해서 선택하세요.<br>
    ② <strong>▶ 디텍션 시작</strong> 버튼을 누르면 자동으로 처리됩니다.<br>
    ③ <em>라벨링 결과</em> 탭에서 감지된 객체의 박스를, <em>최종 결과</em> 탭에서 Specimen만 남긴 이미지를 확인하세요.<br>
    ④ <strong>결과 ZIP 다운로드</strong> 버튼으로 모든 PNG 파일을 받을 수 있습니다.
</div>
"""

with gr.Blocks(title="HerbPAI", css=CSS) as demo:
    gr.HTML("""
    <div class="hero">
        <h1>HerbPAI</h1>
        <p>Herbarium Image Preprocessing for AI</p>
    </div>
    """)
    gr.HTML(GUIDE_HTML)
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="section-label">📂 이미지 업로드</div>')
            input_files = gr.File(
                label="이미지 파일 선택 (여러 장 가능)",
                file_count="multiple",
                file_types=["image"]
            )
            run_btn = gr.Button("▶ 디텍션 시작", variant="primary")
            stats_out = gr.Textbox(
                label="📊 처리 결과 요약",
                lines=8,
                interactive=False,
                placeholder="디텍션 후 결과가 여기 표시됩니다."
            )
            download_file = gr.File(
                label="📥 결과 ZIP 다운로드",
                interactive=False
            )
        with gr.Column(scale=2):
            gr.HTML('<div class="section-label">🔬 결과 확인</div>')
            with gr.Tabs():
                with gr.TabItem("🏷️ 라벨링 결과 (바운딩박스)"):
                    labeled_gallery = gr.Gallery(
                        label="감지된 객체 위치",
                        columns=3,
                        height=480,
                        object_fit="contain"
                    )
                with gr.TabItem("🌿 최종 결과 (Specimen)"):
                    result_gallery = gr.Gallery(
                        label="Specimen만 남긴 최종 이미지 (PNG)",
                        columns=3,
                        height=480,
                        object_fit="contain"
                    )
    run_btn.click(
        fn=handle_batch,
        inputs=[input_files],
        outputs=[labeled_gallery, result_gallery, stats_out, download_file]
    )

demo.launch(share=True)