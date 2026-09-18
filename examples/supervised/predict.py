"""Predict normalized public track values for one 16,384-bp DNA sequence."""
import argparse
from pathlib import Path

import numpy as np
import torch

from shorkie_torch.checkpoint import load_pretrained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--fasta', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    lines = Path(args.fasta).read_text().splitlines()
    if sum(line.startswith('>') for line in lines) != 1:
        raise ValueError('Provide exactly one FASTA record')
    seq = ''.join(line.strip() for line in lines if not line.startswith('>')).upper()
    if len(seq) != 16384 or set(seq) - set('ACGT'):
        raise ValueError('This example requires exactly 16384 A/C/G/T bases')
    model = load_pretrained(args.model_dir, device=args.device)
    if model.cfg.output_channels != 78 or model.cfg.input_channels not in (4, 170):
        raise ValueError('Expected a released 78-track supervised model')
    x = torch.zeros(1, len(seq), model.cfg.input_channels, device=args.device)
    indices = torch.tensor(['ACGT'.index(base) for base in seq], device=args.device)
    x[0, torch.arange(len(seq), device=args.device), indices] = 1
    if model.cfg.input_channels == 170:
        x[:, :, 114] = 1  # S. cerevisiae channel used by this training campaign.
    with torch.inference_mode():
        pred = model(x)
    if tuple(pred.shape) != (1, 896, 78) or not torch.isfinite(pred).all():
        raise RuntimeError('Invalid prediction')
    np.save(args.output, pred[0].cpu().numpy(), allow_pickle=False)


if __name__ == '__main__':
    main()
