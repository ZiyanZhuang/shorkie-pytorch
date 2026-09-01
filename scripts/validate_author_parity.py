"""Validate the public PyTorch port against a frozen TensorFlow reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from shorkie_torch import ShorkieLM
from shorkie_torch.h5_weights import load_released_lm_weights


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(reference: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    left = np.asarray(reference, dtype=np.float64).ravel()
    right = np.asarray(observed, dtype=np.float64).ravel()
    delta = right - left
    return {
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "cosine": float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))),
        "max_abs_error": float(np.max(np.abs(delta))),
        "mean_abs_error": float(np.mean(np.abs(delta))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    with np.load(args.reference, allow_pickle=False) as frozen:
        inputs = frozen["x"].astype(np.float32, copy=False)
        tf_probability = frozen["y"].astype(np.float32, copy=False)
        tf_attention = frozen["multihead_attention"].astype(np.float32, copy=False)

    model = ShorkieLM().cpu().eval()
    load_released_lm_weights(model, args.weights)
    captured: list[torch.Tensor] = []
    handle = model.transformers[0].attention.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach().cpu())
    )
    with torch.inference_mode():
        logits = model(torch.from_numpy(inputs)).numpy()
    handle.remove()
    probability = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    canonical_tf = np.log(np.clip(tf_probability, 1e-30, 1.0))
    canonical_tf -= canonical_tf.mean(axis=-1, keepdims=True)
    canonical_pt = logits - logits.mean(axis=-1, keepdims=True)
    comparisons = {
        "probability": compare(tf_probability, probability),
        "canonical_logit": compare(canonical_tf, canonical_pt),
        "first_attention": compare(tf_attention, captured[0].numpy()),
    }
    passed = all(
        metric["pearson"] >= 0.99999
        and metric["cosine"] >= 0.99999
        and metric["max_abs_error"] <= 1e-3
        for metric in comparisons.values()
    )
    target = passed and all(metric["max_abs_error"] <= 1e-4 for metric in comparisons.values())
    report = {
        "status": "PASS" if passed else "FAIL",
        "target_status": "PASS" if target else "ABOVE_TARGET",
        "reference_sha256": sha256(args.reference),
        "weights_npz_sha256": sha256(args.weights),
        "input_shape": list(inputs.shape),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
