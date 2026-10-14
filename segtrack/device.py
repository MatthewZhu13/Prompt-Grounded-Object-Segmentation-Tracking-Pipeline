"""Device selection: CUDA if available, then Apple MPS, then CPU."""

from __future__ import annotations

import torch


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Wait for queued GPU work, so wall-clock timings measure the actual compute."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        import platform
        return f"Apple MPS ({platform.machine()}, {platform.mac_ver()[0]})"
    return "CPU"
