"""BF16 forward/backward smoke for a CUDA-capable installation."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from shorkie_torch import ShorkieLM, make_mlm_batch, weighted_mlm_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--model-dir", type=str)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.length <= 0 or args.length % 128:
        raise SystemExit("--length must be a positive multiple of 128")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    device = torch.device("cuda")
    if args.model_dir:
        from shorkie_torch import load_pretrained

        model = load_pretrained(args.model_dir).to(device).train()
        model_source = "released_safetensors"
    else:
        model = ShorkieLM().to(device).train()
        model_source = "random_init"
    length = args.length
    codes = torch.arange(length, device=device).remainder(4)[None]
    bases = F.one_hot(codes, 4).float()
    species = F.one_hot(torch.tensor([109], device=device), 165).float()
    annotation = torch.zeros(1, length, device=device)
    inputs, targets, indices, weights = make_mlm_batch(bases, species, annotation, annotation)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(inputs)
        loss, nll = weighted_mlm_loss(logits, targets, indices, weights)
    loss.backward()
    torch.cuda.synchronize()
    finite_gradients = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    report = {
        "status": "PASS" if finite_gradients and torch.isfinite(loss) else "FAIL",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "model_source": model_source,
        "input_shape": [1, length, 170],
        "output_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "unweighted_nll": float(nll.detach()),
        "finite_gradients": bool(finite_gradients),
    }
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
