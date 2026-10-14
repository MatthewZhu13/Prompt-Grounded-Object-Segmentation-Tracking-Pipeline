"""SAM 2 (image mode): turn candidate boxes into masks."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

from .device import get_device

DEFAULT_SAM2 = "facebook/sam2.1-hiera-small"


class BoxSegmenter:
    def __init__(self, model_id: str = DEFAULT_SAM2, device: torch.device | None = None):
        self.device = device or get_device()
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, image: np.ndarray | Image.Image, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (masks (N, H, W) bool, predicted mask quality (N,))."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        w, h = image.size
        if len(boxes) == 0:
            return np.zeros((0, h, w), dtype=bool), np.zeros(0)
        inputs = self.processor(images=image, input_boxes=[boxes.tolist()], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
        )[0][:, 0].numpy()                                                  # (N, H, W) bool
        quality = outputs.iou_scores[0, :, 0].float().cpu().numpy()
        return masks, quality
