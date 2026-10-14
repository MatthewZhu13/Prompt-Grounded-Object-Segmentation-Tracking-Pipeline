"""SigLIP: score masked object crops against target phrases and negative phrases.

SigLIP (unlike CLIP) is trained to score each image–text pair independently, so its
sigmoid output is a per-pair match score that can be thresholded on its own.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .boxes import mask_to_box
from .device import get_device
from .parser import ParsedTask

DEFAULT_SIGLIP = "google/siglip2-so400m-patch14-384"
PROMPT_TEMPLATE = "a photo of a {}."

# Things commonly visible in tabletop robot footage that are never what a target phrase means.
GENERIC_NEGATIVES = ["robot gripper", "robot arm", "table", "background"]
# Common tabletop objects; included unless the task itself targets that noun.
DISTRACTOR_NOUNS = ["bowl", "cup", "plate", "mug", "box", "container", "bottle"]
# Negatives only describe a *different kind of thing*. Off-attribute variants ("blue bowl"
# when the task says "gray bowl") are deliberately excluded: a bluish-looking gray bowl, or a
# white fork the instruction calls pink, would be rejected outright. Choosing between
# attributes is left to the relative comparison in assignment (Hungarian + sibling margins).


class AttributeScorer:
    def __init__(self, model_id: str = DEFAULT_SIGLIP, device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        self.device = device or get_device()
        # fp16 picks the same best phrase as fp32 on every crop we checked, at 3-10x the speed.
        self.dtype = dtype or (torch.float16 if self.device.type in ("cuda", "mps") else torch.float32)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, dtype=self.dtype).to(self.device).eval()
        self._text_cache: dict[tuple[str, ...], torch.Tensor] = {}

    @torch.no_grad()
    def __call__(self, crops: list[Image.Image], phrases: list[str]) -> np.ndarray:
        """(len(crops), len(phrases)) sigmoid match scores in [0, 1].

        Same computation as the model's forward pass (normalized embeddings, cosine × scale +
        bias, sigmoid), split so phrase embeddings are computed once per phrase list: the
        phrases are fixed for a run while crops change every frame.
        """
        if not crops or not phrases:
            return np.zeros((len(crops), len(phrases)))
        text = self._text_embeddings(tuple(phrases))
        pixels = self.processor(images=crops, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        image = self.model.vision_model(pixel_values=pixels).pooler_output
        image = image / image.norm(dim=-1, keepdim=True)
        logits = image @ text.T * self.model.logit_scale.exp() + self.model.logit_bias
        return torch.sigmoid(logits).float().cpu().numpy()

    def _text_embeddings(self, phrases: tuple[str, ...]) -> torch.Tensor:
        if phrases not in self._text_cache:
            tokens = self.processor(text=[PROMPT_TEMPLATE.format(p) for p in phrases],
                                    padding="max_length", max_length=64, return_tensors="pt").to(self.device)
            text = self.model.text_model(input_ids=tokens["input_ids"]).pooler_output
            self._text_cache[phrases] = text / text.norm(dim=-1, keepdim=True)
        return self._text_cache[phrases]


def masked_crop(image: np.ndarray, mask: np.ndarray, pad: float = 0.15, background: int = 128) -> Image.Image:
    """Square crop around `mask`, with pixels outside the mask set to `background`.

    Blanking the background stops neighbouring objects and the table from influencing the
    color judgement; squaring avoids distortion when SigLIP resizes to its input size.
    """
    box = mask_to_box(mask)
    h, w = mask.shape
    if box is None:
        return Image.new("RGB", (32, 32), (background,) * 3)
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx0, sy0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = int(round(side))

    canvas = np.full((s, s, 3), background, dtype=np.uint8)
    src = np.where(mask[..., None], image, background).astype(np.uint8)
    ix0, iy0 = max(sx0, 0), max(sy0, 0)
    ix1, iy1 = min(sx0 + s, w), min(sy0 + s, h)
    canvas[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0] = src[iy0:iy1, ix0:ix1]
    return Image.fromarray(canvas)


def build_negatives(parsed: ParsedTask) -> list[str]:
    """Negative phrases for rejecting non-target crops. Never includes a target phrase."""
    target_phrases = {t.phrase for t in parsed.targets}
    target_nouns = {t.noun for t in parsed.targets}
    negatives = list(GENERIC_NEGATIVES)
    negatives += parsed.destinations
    distractors = [n for n in DISTRACTOR_NOUNS if n not in target_nouns]
    negatives += distractors
    # Distractors also get the task's attributes, so the comparison differs only in the noun:
    # "white cup" vs "white bowl", not "white cup" vs "bowl" (where the color match alone
    # would let "white cup" win on a white bowl).
    attributes = list(dict.fromkeys(" ".join(t.attributes) for t in parsed.targets if t.attributes))
    negatives += [f"{a} {d}" for d in distractors for a in attributes]
    return [n for n in dict.fromkeys(negatives) if n not in target_phrases and n not in target_nouns]
