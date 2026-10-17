"""SAM 2 (image mode): turn candidate boxes (or points) into masks."""

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
        # The image encoder is most of SAM 2's cost; box and point prompts on the same frame
        # reuse its output.
        self._cached: tuple[np.ndarray, list[torch.Tensor], dict] | None = None

    @torch.no_grad()
    def __call__(self, image: np.ndarray | Image.Image, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (masks (N, H, W) bool, predicted mask quality (N,))."""
        image = np.asarray(image)
        h, w = image.shape[:2]
        if len(boxes) == 0:
            return np.zeros((0, h, w), dtype=bool), np.zeros(0)
        embeddings, inputs = self._embed(image)
        prompt = self.processor(images=Image.fromarray(image), input_boxes=[boxes.tolist()], return_tensors="pt")
        outputs = self.model(input_boxes=prompt["input_boxes"].to(self.device), image_embeddings=embeddings,
                             multimask_output=False)
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0][:, 0].numpy()
        quality = outputs.iou_scores[0, :, 0].float().cpu().numpy()
        return masks, quality

    @torch.no_grad()
    def points(self, image: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray:
        """Object mask for each single positive (x, y) point → (N, H, W) bool.

        A single point is ambiguous (part / object / object plus its neighbours), so SAM 2
        returns three masks per point; the one it predicts to be most accurate is used. On two
        touching cups, that is the cup under the point; the largest is usually both cups.
        """
        image = np.asarray(image)
        h, w = image.shape[:2]
        if not points:
            return np.zeros((0, h, w), dtype=bool)
        embeddings, inputs = self._embed(image)
        prompt = self.processor(images=Image.fromarray(image), input_points=[[[[float(p[0]), float(p[1])]] for p in points]],
                                input_labels=[[[1] for _ in points]], return_tensors="pt")
        outputs = self.model(input_points=prompt["input_points"].to(self.device),
                             input_labels=prompt["input_labels"].to(self.device),
                             image_embeddings=embeddings, multimask_output=True)
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0].numpy()
        best = outputs.iou_scores[0].argmax(-1).cpu().numpy()             # (N, 3, H, W) → per point
        return masks[np.arange(len(points)), best]

    def _embed(self, image: np.ndarray):
        if self._cached is not None and self._cached[0] is image:
            return self._cached[1], self._cached[2]
        inputs = self.processor(images=Image.fromarray(image), return_tensors="pt")
        embeddings = self.model.get_image_embeddings(inputs["pixel_values"].to(self.device))
        meta = {"original_sizes": inputs["original_sizes"].cpu()}
        self._cached = (image, embeddings, meta)
        return embeddings, meta
