"""
Model utility functions.
"""

import yaml


def count_parameters(model) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(path: str) -> dict:
    """Load a YAML config file and return as dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
