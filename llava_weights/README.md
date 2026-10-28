<div align="center">

# ReText: Fine-tuned VLM Weights

**LLaVA fine-tuned on UFine3C for person image captioning**

[Project overview](../README.md) · [Paper (arXiv)](https://arxiv.org/abs/2602.05785) · [Textual annotations](../texts_for_multi_camera)

</div>

## Overview

This directory contains the fine-tuned VLM weights used to generate textual descriptions for ReText and a script for captioning your own images using HAM.

The weights and captioning files are also available on [Hugging Face](https://huggingface.co/timmzak/ReText-LLaVA).

## Release contents

| File | Description |
| :--- | :--- |
| [ham-release/ham/non_lora_trainables.safetensors](ham-release/ham/non_lora_trainables.safetensors) | Fine-tuned weights loaded on top of the base model; stored with Git LFS. |
| [ham-release/ham/config.json](ham-release/ham/config.json) | Model configuration. |
| [ham-release/caption_own_images.py](ham-release/caption_own_images.py) | Batch caption generation script. |
| [ham-release/requirements.txt](ham-release/requirements.txt) | Python dependencies. |

The base models are not bundled. The script uses `liuhaotian/llava-v1.6-vicuna-7b` and `openai/clip-vit-large-patch14-336` from Hugging Face unless local copies are present under `MODELS_DIR` in directories named `llava-v1.6-vicuna-7b` and `clip-vit-large-patch14-336`.

## Caption your own images

Requirements: Linux, Python 3.10, and an NVIDIA GPU with approximately 16 GB of memory.

### 1. Download the weights

From the root of your ReText checkout, retrieve the Git LFS files:

```bash
git lfs install
git lfs pull
```

### 2. Get HAM and the captioning script

Run from the root of your ReText checkout:

```bash
git clone https://github.com/MPI-Lab/HAM.git
cp llava_weights/ham-release/caption_own_images.py llava_weights/ham-release/requirements.txt HAM/
cd HAM
```

### 3. Create the environment

```bash
python3.10 -m venv ham-env
source ham-env/bin/activate
pip install -r requirements.txt
```

Do not run `pip install -e .` for HAM itself.

### 4. Configure paths

Edit the settings at the top of `caption_own_images.py`:

```python
MODELS_DIR = "/absolute/path/to/ReText/llava_weights/ham-release"
IMAGE_DIR = "/absolute/path/to/images"
OUTPUT_PATH = "captions.jsonl"
```

`MODELS_DIR` must contain the `ham` directory with the weights and configuration. The script also exposes `GPU`, `BATCH_SIZE`, and `NUM_WORKERS` settings.

### 5. Generate captions

```bash
python caption_own_images.py
```

## Output

The script writes a JSONL file with one record per image. Each record has an `image` field containing the path relative to `IMAGE_DIR` and a `caption` field containing the generated description.

Restarting the script with the same output file skips images already recorded there. The captioning prompt can be changed through `PROMPT`; each caption uses a randomly chosen annotator style, as in HAM.

## Citation

Please use the [ReText citation](../README.md#citation).
