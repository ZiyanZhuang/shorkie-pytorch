"""Convert a trusted local training checkpoint to a public safetensors bundle.

PyTorch checkpoints use pickle. Run this command only on a checkpoint you
created or otherwise trust. Public consumers should use ``load_pretrained``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from shorkie_torch import ShorkieConfig, ShorkieLM


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-epoch", type=int)
    parser.add_argument("--expected-global-step", type=int)
    args = parser.parse_args()

    observed_checkpoint_sha = _sha256(args.checkpoint)
    if args.expected_checkpoint_sha256 and observed_checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("training checkpoint SHA-256 mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError("checkpoint does not contain a model state")
    epoch = int(checkpoint.get("epoch_completed", -1))
    global_step = int(checkpoint.get("global_step", -1))
    if args.expected_epoch is not None and epoch != args.expected_epoch:
        raise ValueError(f"epoch mismatch: {epoch}")
    if args.expected_global_step is not None and global_step != args.expected_global_step:
        raise ValueError(f"global-step mismatch: {global_step}")

    cfg = ShorkieConfig()
    model = ShorkieLM(cfg)
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict load failed: {incompatible}")
    state = {name: tensor.detach().contiguous().cpu() for name, tensor in model.state_dict().items()}
    if len(state) != 356:
        raise ValueError(f"unexpected tensor count: {len(state)}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    state_path = args.output_dir / "model.safetensors"
    save_file(state, str(state_path), metadata={"format": "pt", "model_type": "shorkie_lm"})
    state_sha = _sha256(state_path)
    release = {
        "format_version": 1,
        "model_type": "shorkie_lm",
        "architecture": asdict(cfg),
        "state_file": state_path.name,
        "state_sha256": state_sha,
    }
    (args.output_dir / "config.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    training = {
        "release_candidate": "v0.1.0-rc1",
        "training_checkpoint_sha256": observed_checkpoint_sha,
        "epoch_completed": epoch,
        "global_step": global_step,
        "tensor_count": len(state),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_element_count": sum(tensor.numel() for tensor in state.values()),
        "best_valid": float(checkpoint["best_valid"]),
        "excluded_from_release": ["optimizer", "rng_state", "sampler_state", "metrics", "paths", "tracking identifiers"],
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(training, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "state_sha256": state_sha, **training}, indent=2))


if __name__ == "__main__":
    main()
