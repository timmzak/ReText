import os

MODELS_DIR = "/path/to/ham_release"
IMAGE_DIR = "/path/to/images"
OUTPUT_PATH = "captions.jsonl"
GPU = 0
BATCH_SIZE = 16
NUM_WORKERS = 8
MAX_RETRIES = 3                       # regenerate captions that are too short/long or malformed
PROMPT = (
    "Describe the person focusing only on stable visual features across cameras, "
    "including clothing, shoes, hairstyle, accessories, and carried objects, without "
    "mentioning body shape, height, face, emotions, pose, actions, background, or context."
)

WEIGHTS_DIR = os.path.join(MODELS_DIR, "ham")
MODEL_BASE = os.path.join(MODELS_DIR, "llava-v1.6-vicuna-7b")
CLIP_PATH = os.path.join(MODELS_DIR, "clip-vit-large-patch14-336")
if not os.path.isdir(MODEL_BASE):
    MODEL_BASE = "liuhaotian/llava-v1.6-vicuna-7b"
if not os.path.isdir(CLIP_PATH):
    CLIP_PATH = "openai/clip-vit-large-patch14-336"
if os.path.isdir(MODEL_BASE) and os.path.isdir(CLIP_PATH):
    os.environ["HF_HUB_OFFLINE"] = "1"

import json
import sys

HAM_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HAM_ROOT, "transformers-4.31.0", "src"))
sys.path.insert(0, HAM_ROOT)

import torch
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer

import llava.model.llava_arch as llava_arch
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, expand2square, tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
from llava.model.multimodal_encoder.clip_encoder import CLIPVisionConfig, CLIPVisionTower
from llava.utils import disable_torch_init


def _clip_init(self, vision_tower, args, delay_load=False):
    torch.nn.Module.__init__(self)
    self.is_loaded = False
    self.vision_tower_name = CLIP_PATH
    self.select_layer = args.mm_vision_select_layer
    self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
    self.cfg_only = CLIPVisionConfig.from_pretrained(CLIP_PATH)


CLIPVisionTower.__init__ = _clip_init
llava_arch.build_vision_tower = lambda cfg, **kw: CLIPVisionTower(CLIP_PATH, args=cfg)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_model():
    disable_torch_init()
    torch.cuda.set_device(GPU)  # HAM moves tensors to "cuda", so everything must be on one GPU
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        MODEL_BASE, config=AutoConfig.from_pretrained(WEIGHTS_DIR),
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={"": GPU},
    )
    # Trained HAM weights: style prompts + projector
    weights = load_file(os.path.join(WEIGHTS_DIR, "non_lora_trainables.safetensors"))
    weights = {k.replace("base_model.model.", "", 1): v for k, v in weights.items()}
    unexpected = model.load_state_dict(weights, strict=False).unexpected_keys
    if unexpected:
        raise RuntimeError(f"HAM weights do not match the model: {unexpected[:5]}")
    if getattr(model.config, "mm_use_im_patch_token", True):  # same as llava/model/builder.py
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    vision_tower.load_model()
    vision_tower.to(device=f"cuda:{GPU}", dtype=torch.float16)
    model.eval()
    return tokenizer, model, vision_tower.image_processor


class ImageFolder(Dataset):
    def __init__(self, root, image_processor, skip=()):
        self.root = root
        self.image_processor = image_processor
        self.files = sorted(
            rel for d, _, fs in os.walk(root, followlinks=True) for f in fs
            if os.path.splitext(f)[1].lower() in IMG_EXT
            and (rel := os.path.relpath(os.path.join(d, f), root)) not in skip
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        image = Image.open(os.path.join(self.root, self.files[i])).convert("RGB")
        image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
        return self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0], self.files[i]


def collate(batch):
    return torch.stack([x[0] for x in batch]), [x[1] for x in batch]


def is_valid(caption):
    """Same filter as caption.py in HAM."""
    return 8 <= len(caption.split(" ")) <= 130 and not any(s in caption for s in ("<", ">", "\n", "/n"))


def main():
    tokenizer, model, image_processor = load_model()

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + PROMPT)
    conv.append_message(conv.roles[1], None)
    prompt_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

    def generate(images):
        # All prompts are identical, so no padding is needed.
        # HAM picks a random annotator style for every image.
        input_ids = prompt_ids.repeat(len(images), 1).cuda()
        with torch.inference_mode():
            out = model.generate(
                input_ids, images=images.cuda().half(), do_sample=False, max_new_tokens=512, use_cache=True,
                stopping_criteria=[KeywordsStoppingCriteria([conv.sep2], tokenizer, input_ids)],
            )
        captions = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
        return [c.strip().removesuffix(conv.sep2).strip() for c in captions]

    done = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            done = {json.loads(line)["image"] for line in f if line.strip()}
    dataset = ImageFolder(IMAGE_DIR, image_processor, skip=done)
    print(f"Already captioned: {len(done)}, remaining: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate)

    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        for step, (images, files) in enumerate(loader):
            captions = generate(images)
            for _ in range(MAX_RETRIES):  # a new random style is chosen on every retry
                bad = [i for i, c in enumerate(captions) if not is_valid(c)]
                if not bad:
                    break
                for i, c in zip(bad, generate(images[bad])):
                    captions[i] = c
            for f, c in zip(files, captions):
                out.write(json.dumps({"image": f, "caption": c}, ensure_ascii=False) + "\n")
            out.flush()
            print(f"batch {step + 1}/{len(loader)}")
    print(f"Done: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
