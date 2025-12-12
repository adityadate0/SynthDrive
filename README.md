# SynthDrive

**🧬 SynthDrive** is an offline synthetic data factory for automotive imagery: given a single input frame and a natural-language instruction (e.g., *“make it snowy”*, *“night time”*, *“heavy rain”*), it generates a photorealistic variant while preserving scene geometry via **ControlNet (Canny)**. It is designed to be practical on modest hardware (including older 4GB VRAM GPUs) and can also run fully on CPU.

It ships with:
- A **CLI pipeline** for batch-style generation
- A **Gradio “live build” GUI** that streams intermediate previews as the image forms
- A **showcase builder** that stitches **Input + Edge Lock + Output** into a portfolio triptych

---

## Why SynthDrive

When you need more training data for perception and road-scene tasks (domain shifts like weather, time-of-day, lighting), hand-labeling new collections is expensive. SynthDrive helps you expand coverage from what you already have, without sending data to online services after the initial model download.

---

## How it works (high level)

SynthDrive runs a 3-stage pipeline:

1. **Vision (Auto-captioning)**
   - A BLIP captioner describes the input frame in plain language.
   - This scene description is combined with your instruction into a richer prompt.

2. **Geometry (Edge Lock)**
   - Canny edges are extracted from the input image.
   - The edge map acts as a structural constraint for the generator.

3. **Generation (Stable Diffusion + ControlNet)**
   - Stable Diffusion v1.5 generates the new image.
   - ControlNet-Canny conditions the diffusion process on the edge map to preserve geometry.

The GUI version also decodes latent previews each step to stream a live “image forming” view.

---

## Repository layout

Recommended structure (matches imports used by the code):

```
SynthDrive/
├─ app.py
├─ main.py
├─ setup_factory.py
├─ create_showcase.py
├─ requirements.txt
├─ models/                 # created by setup_factory.py
├─ outputs/                # generated results
└─ src/
   ├─ __init__.py
   ├─ generator.py
   └─ processor.py
```

If your repo currently has `generator.py` / `processor.py` at the root, move them into `src/` (or adjust imports in `app.py` and `main.py` accordingly).

---

## Requirements

### Software
- Python **3.9–3.11** (recommended: 3.10)
- pip / venv (or conda)
- CUDA-capable GPU optional (CPU mode supported)

### Hardware (practical expectations)
- **CPU mode:** works broadly but slower
- **GPU mode:** optimized for low VRAM with sequential offload and VAE slicing; 4GB VRAM may work depending on driver and environment

---

## Quickstart

### 1) Clone
```bash
git clone <your-repo-url>
cd SynthDrive
```

### 2) Create a virtual environment
**Linux / macOS**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Download models (one-time)
This step requires internet and downloads several GB into `./models/`.

```bash
python setup_factory.py
```

After this completes, you can run SynthDrive offline.

---

## Usage

## Option A: CLI (single image)

```bash
python main.py --input path/to/image.png --instruction "make it snowy"
```

Save to a specific location:
```bash
python main.py --input path/to/image.png --instruction "make it night" --output outputs/night.png
```

Force CPU mode (safe fallback if CUDA fails or VRAM is tight):
```bash
python main.py --input path/to/image.png --instruction "heavy rain" --cpu
```

**Outputs**
- `outputs/result.png` (or your specified `--output`)
- `outputs/debug_edges.png` (edge map used for ControlNet)

---

## Option B: GUI (live streaming)

```bash
python app.py
```

This launches a local Gradio app and opens your browser. The GUI shows:
- **Live Construction**: step-by-step preview as diffusion runs
- **Final Showcase**: a stitched triptych suitable for documentation or reports

**Outputs**
- `outputs/result_<timestamp>.png`
- `outputs/showcase_<timestamp>.png`
- Temporary files: `outputs/temp_input.png`, `outputs/temp_edges.png`

---

## Showcase builder (triptych)

You can create a stitched “Input / ControlNet / Output” image:

```bash
python create_showcase.py --input outputs/temp_input.png --output outputs/result.png
```

In the default script flow, the edge map is assumed at:
- `outputs/debug_edges.png`

You can also call the helper from Python:

```python
from create_showcase import create_triptych
create_triptych("input.png", "edges.png", "output.png", "showcase.png")
```

---

## Configuration knobs (where to tweak)

### Canny thresholds
In `src/processor.py`:
- `ImageProcessor.get_canny_edges(image_path, low_threshold=100, high_threshold=200)`

Lower thresholds = more edges (stronger conditioning), but may over-constrain.

### Diffusion steps and guidance
In `src/generator.py`:
- `inference_steps=20`
- `guidance=7.5`

If you have GPU headroom and want more fidelity, try 25–35 steps.
If you’re seeing prompt overreach, reduce guidance.

---

## Troubleshooting

### “Models missing. Run setup_factory.py”
Run:
```bash
python setup_factory.py
```
This creates `models/` and downloads all required weights.

### CUDA out of memory / crashes on GPU
Run CPU mode:
```bash
python main.py --input <img> --instruction "<text>" --cpu
```

If you want to stay on GPU:
- Close other GPU-heavy apps
- Keep inference steps at 20
- Ensure you’re using FP16 + sequential offload (enabled by default in GPU mode)

### Gradio UI theme looks wrong / keeps resetting
The GUI forces dark mode on load via a query parameter and hides the footer.

---

## What gets downloaded

`setup_factory.py` downloads and stores locally:
- Stable Diffusion v1.5
- ControlNet (Canny)
- BLIP image captioning base

These are pulled from public model hubs and saved into `./models/` so runtime can be offline.

---

## Security & privacy

- After you download models once, SynthDrive can run **fully offline**.
- Your images stay local to your machine.
- Do not expose the Gradio server publicly unless you understand the implications (default is local use).

---

## Roadmap ideas

If you want to evolve SynthDrive into a more complete “dataset factory,” typical next steps are:
- Batch folder processing with metadata logging (CSV/JSONL)
- Deterministic seeding for reproducibility
- Multi-control conditioning (depth, segmentation, pose)
- Augmentation presets (snow/rain/night/fog packs)
- Automatic dataset splits and export formats

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Add changes with clear commits
4. Open a PR with before/after examples (images preferred)

---

## License

Choose a license before publishing. Common choices:
- MIT (permissive)
- Apache-2.0 (patent grant)
- GPL-3.0 (copyleft)

Add a `LICENSE` file matching your preference.

---

## Acknowledgements

SynthDrive stands on the shoulders of:
- **Diffusers** (Stable Diffusion pipeline)
- **ControlNet** conditioning
- **BLIP** image captioning
- Gradio for the UI
