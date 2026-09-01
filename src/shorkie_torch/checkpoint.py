"""Safe public checkpoint loading for Shorkie-LM release artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from .model import ShorkieConfig, ShorkieLM


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("format_version") != 1 or config.get("model_type") != "shorkie_lm":
        raise ValueError("unsupported Shorkie-LM release format")
    architecture = config.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError("config.json lacks an architecture object")
    known = {item.name for item in fields(ShorkieConfig)}
    unknown = sorted(set(architecture) - known)
    if unknown:
        raise ValueError(f"unknown architecture fields: {unknown}")
    return config


def load_pretrained(
    model_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> ShorkieLM:
    """Load a release directory without invoking pickle deserialization.

    The directory must contain ``config.json`` and the safetensors file named by
    ``state_file``. The state hash is checked before strict model loading.
    """

    root = Path(model_dir)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    release = _read_config(config_path)
    state_path = root / release.get("state_file", "model.safetensors")
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    expected = release.get("state_sha256")
    observed = _sha256(state_path)
    if expected and observed != expected:
        raise ValueError(f"state SHA-256 mismatch: expected {expected}, got {observed}")

    cfg = ShorkieConfig(**release["architecture"])
    model = ShorkieLM(cfg)
    state = load_file(str(state_path), device=str(device))
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load failed: {incompatible}")
    model.to(device).eval()
    return model
