"""Run source-style full-position MLM perplexity on a method-rebuild corpus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from shorkie_torch import load_pretrained
from shorkie_torch.evaluation import evaluate, write_results


@dataclass(frozen=True)
class _Metadata:
    model_id: str
    checkpoint: str

    def to_dict(self) -> dict[str, str]:
        return {"model_id": self.model_id, "checkpoint": self.checkpoint}


class _Adapter:
    def __init__(self, model_dir: Path, device: str, model_id: str) -> None:
        self.model = load_pretrained(model_dir, device=device)
        self.device = torch.device(device)
        self.metadata = _Metadata(model_id=model_id, checkpoint=str(model_dir))

    def logits(self, inputs: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            tensor = torch.from_numpy(inputs).to(self.device)
            return self.model(tensor).float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--windows-tsv", type=Path, required=True)
    parser.add_argument("--gtf-gz", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="method_rebuild")
    parser.add_argument("--seed", type=int, default=165)
    parser.add_argument("--batch-windows", type=int, default=1)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rc-mode", choices=("seeded", "none"), default="seeded")
    args = parser.parse_args()

    adapter = _Adapter(args.model_dir, args.device, args.model_id)
    rows, summary = evaluate(
        adapter,
        corpus_root=args.corpus_root,
        windows_tsv=args.windows_tsv,
        gtf_gz=args.gtf_gz,
        split=args.split,
        seed=args.seed,
        batch_windows=args.batch_windows,
        max_windows=args.max_windows,
        rc_mode=args.rc_mode,
    )
    write_results(args.run_dir, args.model_id, rows, summary)


if __name__ == "__main__":
    main()
