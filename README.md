# HerbPAI

**Herbarium specimen image preprocessing for AI.**

HerbPAI automatically removes non-specimen elements — barcodes, rulers, colour
palettes, labels, stamps — from herbarium sheet photographs, leaving only the
plant specimen.

![HerbPAI before and after](docs/images/before_after_en.png)

*The two output modes shown side by side. See [Choosing an output mode](#choosing-an-output-mode).*

---

## Which one do I need?

HerbPAI can be used in two ways. **Most users only need the first.**

| | 🌐 **Web demo** | 💻 **This repository** |
|---|---|---|
| **Best for** | Trying it out, a few images | Hundreds or thousands of images |
| **Installation** | **None** | Python + setup (~20 min, once) |
| **Coding needed** | No | Copy-paste a few commands |
| **Start here** | 👉 **[Open the web demo](https://huggingface.co/spaces/SjSj1008/HerbPAI)** | 👉 **[Read the manual](docs/MANUAL_EN.md)** |

The web demo (`app.py`) and the local tool (`detect_dual.py`) use the same model
and the same processing logic.

---

## How it works

`detect_dual.py` runs a single pipeline. Steps 1–3 are always the same; only the
**final step** differs depending on the output mode you choose.

![HerbPAI pipeline](docs/images/pipeline_en.png)

1. **Detect** — YOLOv9-e finds bounding boxes for 11 classes.
2. **Sort the boxes** — into `Specimen` and everything else. (Nothing is changed
   yet; this ordering step is what makes the result reproducible.)
3. **Erase non-specimen** — every non-specimen box is painted white.
4. **Output** — either crop to the specimen box (default) or place it on a
   white canvas (`--keep-canvas`).

**Edge cases.** If several `Specimen` boxes are found, only the **largest** is
kept — this guards against over-detection. If **no** `Specimen` is found, no
crop or canvas replacement happens: you get the original-size image with the
non-specimen elements erased.

---

## Choosing an output mode

Both modes are useful — pick based on what you will do with the images next.

### Default — crop

```bash
python detect_dual.py --weights best.pt --source my_images --name result
```

The image is cut down to the specimen box. **Output size varies per photo**,
depending on how large the specimen appears. There is no white margin.

**Choose this when:**
- You are building an AI training set. Training pipelines resize images anyway,
  and with no wasted white padding the specimen keeps more of its resolution.
- You want the specimen to fill the frame — e.g. for figures or a gallery.

### `--keep-canvas`

```bash
python detect_dual.py --weights best.pt --source my_images --name result --keep-canvas
```

The specimen is placed on a white background of the **same dimensions as the
input**. Everything else is white.

**Choose this when:**
- Output dimensions must match the input exactly — e.g. the coordinates need to
  stay aligned with the original photo, or an existing workflow expects a fixed
  image size.
- You want to compare input and output side by side.

> 💡 You can run both on the same folder — use a different `--name` for each
> (e.g. `--name cropped` and `--name canvas`) and compare the results.

### Fine-tuning the crop edge

The detected specimen box is rarely pixel-perfect. If the box is slightly **too
large**, a strip of whatever was behind the sheet appears at the edge (a mat, a
table, another sheet — on a white background you may not even notice). If it is
slightly **too small**, the sheet edge or a leaf tip gets clipped.

`--crop-margin` shifts the crop edge in pixels — **negative trims inward,
positive expands outward**:

```bash
# Trim 10 px inward
python detect_dual.py --weights best.pt --source my_images --name result --crop-margin -10

# Expand 5 px outward
python detect_dual.py --weights best.pt --source my_images --name result --crop-margin 5
```

⚠️ The same value applies to **every image in the batch**. If your photos were
taken in very different setups, process them in separate batches.

---

## Quick start

**New to coding? Follow the [manual](docs/MANUAL_EN.md) instead.**

```bash
# 1. Git LFS is required — the AI model file needs it
git lfs install

# 2. Download
git clone https://github.com/Sujeong-Han/HerbPAI.git
cd HerbPAI

# 3. Check the model downloaded — must be ~133 MB, not a few hundred bytes
ls -lh best.pt

# 4. Environment
conda create -n herbpai python=3.10 -y
conda activate herbpai
pip install -r requirements.txt

# 5. Run
python detect_dual.py --weights best.pt --source my_images --name result
```

Results appear in `runs/detect/result/`.

---

## Options

| Option | Meaning | Default |
|---|---|---|
| `--source` | Folder or single image to process | `data/images` |
| `--weights` | Model file | `best.pt` |
| `--name` | Name of the results folder | `exp` |
| `--conf-thres` | Confidence threshold (0–1) | `0.25` |
| `--crop-margin` | Shift the crop edge (px). Negative trims inward, positive expands outward | `0` |
| `--keep-canvas` | Do not crop; keep the input dimensions with a white background | off |
| `--save-crop` | Also save each detected object separately, in `crops/` | off |

---

## Requirements

- **Python 3.10**
- **Git LFS** — required. `best.pt` (~133 MB) is stored with Git LFS. Without it
  you only download a small placeholder and HerbPAI will not run.
- macOS, Windows or Linux. No GPU needed (CPU works, just slower).

---

## Common problems

<details>
<summary><b><code>ImportError: numpy.core.multiarray failed to import</code></b></summary>

NumPy 2.x is installed, but the libraries need 1.x:

```bash
pip install "numpy<2"
```
</details>

<details>
<summary><b>The model does not load / no output is produced</b></summary>

Check the model file size:

```bash
ls -lh best.pt
```

If it is a few hundred **bytes** instead of ~133 **MB**, Git LFS did not fetch it:

```bash
git lfs install
git lfs pull
```
</details>

<details>
<summary><b><code>conda: command not found</code></b></summary>

Anaconda is not installed, or the terminal was not restarted. See the
[manual](docs/MANUAL_EN.md).
</details>

---

## Model

- Architecture: **YOLOv9-e**
- Weights: `best.pt` (~133 MB, via Git LFS). CPU inference; no GPU required.

HerbPAI detects **11 classes**. Only `Specimen` is kept:

| Kept | Removed |
|---|---|
| `Specimen` | `Annotation_label`, `Barcode`, `DB_stamp`, `Envolope`, `Image`, `Institution_stamp`, `Label`, `Map`, `Palette`, `Ruler` |

With `--save-crop`, every detected object is also saved to
`runs/detect/<name>/crops/<class_name>/`, cropped from the **original**
(pre-whitening) image — so this is unaffected by the output mode.

---

## Limitations

HerbPAI performs best on standard herbarium sheet layouts. Known failure cases:

- Grey or beige backgrounds may cause over-detection of the specimen region.
- Rulers placed very close to the specimen are occasionally missed.
- Institution headers printed on the sheet may be kept.
- Handwritten labels on brown paper and newspaper-textured labels are not always
  removed.

Please check the output visually before using it for downstream training.

---

## Citation

<!-- TODO: replace with the final SoftwareX citation + Zenodo DOI -->

```bibtex
@article{han_herbpai,
  title   = {HerbPAI: an open-source tool for herbarium specimen image preprocessing},
  author  = {Han, Sujeong},
  journal = {SoftwareX},
  year    = {2026}
}
```

## License

**GNU General Public License v3.0** — see [LICENSE.md](LICENSE.md).

## Contact

Questions and bug reports are welcome:
[open an issue](https://github.com/Sujeong-Han/HerbPAI/issues).
