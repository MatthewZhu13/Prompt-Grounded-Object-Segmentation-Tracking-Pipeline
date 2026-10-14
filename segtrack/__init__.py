"""Prompt-grounded object segmentation and tracking for LeRobot episodes."""

import os

# Let ops without an Apple-GPU kernel fall back to CPU instead of erroring.
# Must be set before torch initializes MPS, i.e. before any model module imports torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
