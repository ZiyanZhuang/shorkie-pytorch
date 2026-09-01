from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from shorkie_torch import ShorkieConfig, ShorkieLM, load_pretrained


def _bundle(root: Path) -> None:
    # The architecture has coupled channel-rounding constraints. Use the public
    # default rather than inventing a reduced configuration that is not valid.
    cfg = ShorkieConfig()
    model = ShorkieLM(cfg)
    state_path = root / "model.safetensors"
    save_file({name: tensor.contiguous() for name, tensor in model.state_dict().items()}, str(state_path))
    sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
    config = {"format_version": 1, "model_type": "shorkie_lm", "architecture": asdict(cfg), "state_file": state_path.name, "state_sha256": sha}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_safetensors_strict_load(tmp_path: Path) -> None:
    _bundle(tmp_path)
    model = load_pretrained(tmp_path)
    assert isinstance(model, ShorkieLM)
    assert not model.training


def test_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    _bundle(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["state_sha256"] = "0" * 64
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_pretrained(tmp_path)
