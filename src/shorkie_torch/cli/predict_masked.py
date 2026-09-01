"""Predict masked DNA bases with a released Shorkie-LM safetensors bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from shorkie_torch import load_pretrained
from shorkie_torch.datasets import apply_mlm_mask, encode_sequence


DNA = np.asarray(list("ACGT"))
DNA_CODE = {base: index for index, base in enumerate("ACGT")}
DNA_CODE["N"] = 4


def _read_fasta(path: Path) -> str:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    sequence = "".join(line for line in lines if line and not line.startswith(">"))
    if not sequence:
        raise ValueError(f"no sequence found in {path}")
    return sequence.upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--positions", type=int, nargs="+", required=True, help="Zero-based positions to mask")
    parser.add_argument("--species-index", type=int, default=109)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sequence = _read_fasta(args.fasta)
    if len(sequence) != 16_384:
        raise ValueError(f"Shorkie-LM expects exactly 16,384 bp, received {len(sequence)}")
    try:
        codes = np.fromiter((DNA_CODE[base] for base in sequence), dtype=np.uint8, count=len(sequence))
    except KeyError as error:
        raise ValueError(f"unsupported FASTA base: {error.args[0]!r}") from error
    positions = np.asarray(args.positions, dtype=np.int64)
    if np.any(positions < 0) or np.any(positions >= len(sequence)):
        raise ValueError("mask positions are outside the sequence")

    encoded = encode_sequence(codes, args.species_index)
    masked = apply_mlm_mask(encoded, positions)
    model = load_pretrained(args.model_dir, device=args.device)
    with torch.inference_mode():
        logits = model(torch.from_numpy(masked[None]).to(args.device))[0, positions]
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    rows = []
    for position, probability in zip(positions.tolist(), probabilities):
        rows.append({
            "position": position,
            "reference": sequence[position],
            "prediction": str(DNA[int(probability.argmax())]),
            "probabilities": {base: float(value) for base, value in zip(DNA.tolist(), probability.tolist())},
        })
    result = {"status": "PASS", "species_index": args.species_index, "predictions": rows}
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
