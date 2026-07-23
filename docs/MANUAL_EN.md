# HerbPAI — Step-by-step Manual

**For users with no programming experience.**

This guide assumes you have never used a terminal before. Follow the steps in
order. You only need to do the installation **once**; after that, processing
images takes a single command.

> 🌐 **Do you really need this?**
> If you only have a handful of images, use the
> **[web demo](https://huggingface.co/spaces/SjSj1008/HerbPAI)** instead —
> no installation at all. This manual is for processing **large batches**
> (hundreds or thousands of images) on your own computer.

---

## Contents

1. [What you will install](#1-what-you-will-install)
2. [Opening the terminal](#2-opening-the-terminal)
3. [Install Anaconda](#3-install-anaconda)
4. [Install Git and Git LFS](#4-install-git-and-git-lfs)
5. [Download HerbPAI](#5-download-herbpai)
6. [Set up the environment](#6-set-up-the-environment)
7. [Process your images](#7-process-your-images)
8. [Finding your results](#8-finding-your-results)
9. [Using HerbPAI again later](#9-using-herbpai-again-later)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What you will install

| Software | Why |
|---|---|
| **Anaconda** | Runs Python and keeps HerbPAI's libraries separate from your computer's other software |
| **Git + Git LFS** | Downloads HerbPAI, including the large AI model file |

This takes about 20–30 minutes, once.

---

## 2. Opening the terminal

The **terminal** is a window where you type commands instead of clicking buttons.
It looks intimidating, but you will only ever copy and paste.

- **macOS:** Press `Cmd + Space`, type `Terminal`, press `Enter`.
- **Windows:** After installing Anaconda (step 3), open the Start menu and
  search for **Anaconda Prompt**. Use that, **not** the regular Command Prompt.

> 💡 **How to use commands in this manual:** copy the line, paste it into the
> terminal (`Cmd+V` on Mac, right-click on Windows), then press `Enter`.
> Run one line at a time and wait for it to finish.

---

## 3. Install Anaconda

1. Go to <https://www.anaconda.com/download>
2. Download the installer for your operating system.
3. Run the installer and accept all the default options.
4. **Close the terminal and open a new one** (important — it only picks up
   Anaconda after a restart).

**Check it worked.** Type this and press Enter:

```bash
conda --version
```

If you see something like `conda 24.5.0`, it worked. If you see
`command not found`, see [Troubleshooting](#10-troubleshooting).

---

## 4. Install Git and Git LFS

Git downloads HerbPAI. **Git LFS** is essential: the AI model file (`best.pt`,
about 133 MB) is stored using Git LFS, and without it you will download a broken
placeholder file instead of the real model.

**macOS** — install [Homebrew](https://brew.sh) first if you don't have it, then:

```bash
brew install git git-lfs
```

**Windows** — download and install:
- Git: <https://git-scm.com/download/win>
- Git LFS: <https://git-lfs.github.com>

**Then, on both systems, activate Git LFS:**

```bash
git lfs install
```

**Check it worked:**

```bash
git lfs version
```

You should see something like `git-lfs/3.7.1`.

---

## 5. Download HerbPAI

First, move to the folder where you want HerbPAI to live. Your Desktop is a
simple choice:

```bash
cd Desktop
```

Then download it:

```bash
git clone https://github.com/Sujeong-Han/HerbPAI.git
cd HerbPAI
```

### ⚠️ Check the model file — do not skip this

```bash
ls -lh best.pt
```

On Windows use `dir best.pt` instead.

- ✅ **~133 MB** → correct, continue.
- ❌ **A few hundred bytes** → Git LFS did not work. Run `git lfs pull` and
  check again.

---

## 6. Set up the environment

This creates an isolated space for HerbPAI's libraries so they cannot break
other software on your computer.

```bash
conda create -n herbpai python=3.10 -y
```

Activate it:

```bash
conda activate herbpai
```

Your prompt should now start with `(herbpai)`. **You must see `(herbpai)` before
running HerbPAI** — if you don't, run the activate command again.

Install the libraries (this takes a few minutes):

```bash
pip install -r requirements.txt
```

---

## 7. Process your images

**Put your images in a folder.** Inside the `HerbPAI` folder, create a folder
called `my_images` and copy your herbarium photos into it (just drag them there
in Finder or File Explorer — no commands needed).

Then run:

```bash
python detect_dual.py --weights best.pt --source my_images --name result
```

You will see progress messages as each image is processed. A folder of 500
images may take a while on a laptop — this is normal.

### Useful variations

Process a single image:

```bash
python detect_dual.py --weights best.pt --source my_images/photo1.jpg --name result
```

Also save each detected object (barcode, ruler...) as a separate image:

```bash
python detect_dual.py --weights best.pt --source my_images --name result --save-crop
```

This creates `runs/detect/result/crops/` with one folder per class
(`Barcode`, `Ruler`, `Label`, ...) containing the cropped objects.

### Which mode should I use?

HerbPAI can write the result in two ways. Choose based on **what you will do with
the images next.**

| | Default (crop) | `--keep-canvas` |
|---|---|---|
| Result | Cut down to the specimen box | Specimen on a white background |
| Output size | Varies per photo | Same as the input |
| White margin | None | Around the specimen |
| Recommended for | Building an AI training set | Keeping original dimensions |

**For an AI training set, use the default (crop).** Training pipelines resize
images anyway, and with no white padding the specimen keeps more of its
resolution.

**If the output must match the input size, use `--keep-canvas`** — for example
when coordinates need to stay aligned with the original photo, or an existing
workflow expects a fixed image size.

> 💡 **Not sure? Run both.** Give each a different `--name` and the results are
> saved to separate folders, so you can compare them.
>
> ```bash
> python detect_dual.py --weights best.pt --source my_images --name cropped
> python detect_dual.py --weights best.pt --source my_images --name canvas --keep-canvas
> ```

For the full processing pipeline, see *How it works* in the
[README](../README.md).

### Adjusting the border with `--crop-margin`

By default HerbPAI **crops** the output to the detected specimen box. That box is
rarely pixel-perfect, which causes one of two things:

- **The box is slightly too large** → a strip of whatever was behind the sheet is
  included at the edge. Depending on how the photo was taken this could be a
  cutting mat, a table, a cloth, or another sheet of paper. On a white background
  you may not even notice it.
- **The box is slightly too small** → the edge of the sheet, or a leaf tip, gets
  clipped off.

`--crop-margin` shifts the crop edge in pixels. **Negative** trims inward,
**positive** expands outward.

```bash
# Trim 10 px inward — removes a thin strip of background at the edge
python detect_dual.py --weights best.pt --source my_images --name result --crop-margin -10

# Expand 5 px outward — recovers a slightly clipped specimen
python detect_dual.py --weights best.pt --source my_images --name result --crop-margin 5
```

**How to choose a value.** Run once with the default (`0`), open a few results,
and look at the edges. Then adjust in small steps (`-5`, `-10`, `-20`...) until
the edges look right.

> ⚠️ The same value is applied to **every image in the batch**, so pick a value
> that works for your photos as a whole. If your images were taken in very
> different setups, process them in separate batches. Be careful with large
> negative values — they will cut into the specimen itself.

### Keeping the original image size

To paint everything except the specimen white **without cropping**:

```bash
python detect_dual.py --weights best.pt --source my_images --name result --keep-canvas
```

> 💡 **About output size:** in the default (crop) mode, output dimensions depend
> on how large the specimen appears in each photo, so **different images produce
> different output sizes**. This is usually preferable for building a training
> set, since no white padding is wasted. Use `--keep-canvas` if you need the
> output to match the input dimensions exactly.

---

## 8. Finding your results

Your cleaned images are in:

```
HerbPAI/runs/detect/result/
```

Open that folder normally in Finder or File Explorer.

> 💡 If you run the command again with the same `--name result`, HerbPAI creates
> `result2`, `result3`, and so on. Your old results are never overwritten.

**Always check a few results visually.** HerbPAI is accurate but not perfect —
see *Limitations* in the [README](../README.md).

---

## 9. Using HerbPAI again later

Installation is done once. Next time, you only need three lines:

```bash
cd Desktop/HerbPAI
conda activate herbpai
python detect_dual.py --weights best.pt --source my_images --name result
```

---

## 10. Troubleshooting

### `conda: command not found`

Anaconda is not installed, or the terminal was not restarted. Close the terminal,
open a new one, and try again. On Windows, make sure you are using **Anaconda
Prompt**, not Command Prompt.

### `ImportError: numpy.core.multiarray failed to import`

A version conflict. Fix it with:

```bash
pip install "numpy<2"
```

### The model does not load / no images are produced

Your `best.pt` is probably a broken placeholder. Check its size:

```bash
ls -lh best.pt
```

If it is not ~133 MB:

```bash
git lfs install
git lfs pull
```

### `No such file or directory: my_images`

The terminal is not in the `HerbPAI` folder, or your image folder has a
different name. Check where you are:

```bash
pwd
ls
```

`pwd` shows your current folder; `ls` lists what is inside it (`dir` on Windows).

### The specimen is cropped incorrectly

Two different problems:

**1) A dark edge remains, or leaf tips are clipped** → adjust `--crop-margin`.
See *Adjusting the border* in step 7.

**2) The specimen region itself is detected wrongly** → a known limitation,
especially on grey or beige backgrounds. See *Limitations* in the
[README](../README.md).

### Something else

Please [open an issue](https://github.com/Sujeong-Han/HerbPAI/issues) and
include the full error message you saw in the terminal.
