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
        return self.score(self.embed(crops), phrases)

    @torch.no_grad()
    def embed(self, crops: list[Image.Image]) -> np.ndarray:
        """(len(crops), D) unit-norm image embeddings (also used as appearance memory)."""
        if not crops:
            return np.zeros((0, self.model.config.vision_config.hidden_size), np.float32)
        pixels = self.processor(images=crops, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        image = self.model.vision_model(pixel_values=pixels).pooler_output
        image = image / image.norm(dim=-1, keepdim=True)
        return image.float().cpu().numpy()

    @torch.no_grad()
    def score(self, embeddings: np.ndarray, phrases: list[str]) -> np.ndarray:
        """Sigmoid match scores for precomputed image embeddings (see `embed`)."""
        if not len(embeddings) or not phrases:
            return np.zeros((len(embeddings), len(phrases)))
        text = self._text_embeddings(tuple(phrases)).float()
        image = torch.from_numpy(embeddings).to(self.device)
        logits = image @ text.T * self.model.logit_scale.exp().float() + self.model.logit_bias.float()
        return torch.sigmoid(logits).cpu().numpy()

    def _text_embeddings(self, phrases: tuple[str, ...]) -> torch.Tensor:
        if phrases not in self._text_cache:
            tokens = self.processor(text=[PROMPT_TEMPLATE.format(p) for p in phrases],
                                    padding="max_length", max_length=64, return_tensors="pt").to(self.device)
            text = self.model.text_model(input_ids=tokens["input_ids"]).pooler_output
            self._text_cache[phrases] = text / text.norm(dim=-1, keepdim=True)
        return self._text_cache[phrases]


def masked_crop(image: np.ndarray, mask: np.ndarray, pad: float = 0.15, background: int = 128,
                blank_outside: bool = True) -> Image.Image:
    """Square crop around `mask`, with pixels outside the mask set to `background`.

    Blanking the background stops neighbouring objects and the table from influencing the
    color judgement; squaring avoids distortion when SigLIP resizes to its input size.
    `blank_outside=False` keeps the surroundings (see `context_crop`).
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
    src = np.where(mask[..., None], image, background).astype(np.uint8) if blank_outside else image
    ix0, iy0 = max(sx0, 0), max(sy0, 0)
    ix1, iy1 = min(sx0 + s, w), min(sy0 + s, h)
    canvas[iy0 - sy0:iy1 - sy0, ix0 - sx0:ix1 - sx0] = src[iy0:iy1, ix0:ix1]
    return Image.fromarray(canvas)


def context_crop(image: np.ndarray, mask: np.ndarray, pad: float = 0.15) -> Image.Image:
    """Square crop around `mask` without blanking: for telling a cup from a bowl up close.

    On a wrist-camera close-up the mask's outline alone is ambiguous (a bowl's rim and a cup's
    rim, both cut off by the frame edge); the unblanked crop shows the depth of the inside and
    how the object sits, and SigLIP reads cup vs bowl from it far better (on this episode's
    close-ups: 74% vs 48% of bowl crops read as bowl, 13% vs 29% of cup crops wrongly so).
    """
    return masked_crop(image, mask, pad=pad, blank_outside=False)


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
    # Same noun, a color the task uses only for other nouns ("gray bowl" when the task names
    # white and yellow bowls and a gray cup). Only for nouns the task tells apart by color
    # (two or more attributed instances): there color is decisive, so a third color is a
    # different object. A noun with one instance keeps color descriptive, not decisive (a
    # pink fork under glare may read white).
    by_noun: dict[str, set[str]] = {}
    for t in parsed.targets:
        if t.attributes:
            by_noun.setdefault(t.noun, set()).add(" ".join(t.attributes))
    for noun, used in by_noun.items():
        if len(used) >= 2:
            negatives += [f"{a} {noun}" for a in attributes if a not in used]
    return [n for n in dict.fromkeys(negatives) if n not in target_phrases and n not in target_nouns]
