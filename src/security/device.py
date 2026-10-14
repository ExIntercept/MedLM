"""Standalone device selection for security modules.

Do NOT import src.retrieval here — that loads heavy models at import time.
"""
import torch


def get_device() -> str:
    """Returns the best available compute device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
