"""Verify the public safetensors model bundle without loading pickle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from shorkie_torch import load_pretrained


EXPECTED_FILES = {
    "README.md",
    "model.safetensors",
    "config.json",
    "evaluation.json",
    "training_summary.json",
    "release_verification.json",
    "SHA256SUMS",
    "LICENSE",
}
PLATFORM_FILES = {".gitattributes"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.model_dir.resolve()
    observed_files = {
        path.name for path in root.iterdir()
        if path.is_file() and path.name not in PLATFORM_FILES
    }
    if observed_files != EXPECTED_FILES:
        raise ValueError(
            f"bundle file contract mismatch: missing={sorted(EXPECTED_FILES - observed_files)}, "
            f"unexpected={sorted(observed_files - EXPECTED_FILES)}"
        )

    sums: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        sums[name.strip()] = digest
    expected_sum_names = EXPECTED_FILES - {"SHA256SUMS"}
    if set(sums) != expected_sum_names:
        raise ValueError("SHA256SUMS file set mismatch")
    mismatches = {
        name: {"expected": expected, "observed": sha256(root / name)}
        for name, expected in sums.items()
        if sha256(root / name) != expected
    }
    if mismatches:
        raise ValueError(f"bundle checksum mismatch: {mismatches}")

    parsed = {
        name: json.loads((root / name).read_text(encoding="utf-8"))
        for name in ("config.json", "evaluation.json", "training_summary.json", "release_verification.json")
    }
    card = (root / "README.md").read_text(encoding="utf-8")
    if not card.startswith("---\n") or "license: apache-2.0" not in card or "library_name: shorkie_torch" not in card:
        raise ValueError("README.md lacks required Hugging Face metadata")

    model = load_pretrained(root)
    report = {
        "status": "PASS",
        "files": sorted(observed_files),
        "safetensors_sha256": sha256(root / "model.safetensors"),
        "strict_load": True,
        "all_finite": all(bool(torch.isfinite(tensor).all()) for tensor in model.state_dict().values()),
        "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_tensor_count": len(model.state_dict()),
        "state_element_count": sum(tensor.numel() for tensor in model.state_dict().values()),
        "release": parsed["config.json"].get("release"),
    }
    if not report["all_finite"]:
        raise ValueError("non-finite model state")
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
