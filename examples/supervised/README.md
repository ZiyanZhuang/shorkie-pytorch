# Portable supervised training and OOF evaluation

Install the repository with `python -m pip install -e '.[supervised]'`.
Supply your own admitted cache and trusted local LM training checkpoint.
Do not load arbitrary downloaded pickle checkpoints. Public inference bundles
use safetensors; these research training scripts load your trusted local state.

```bash
python examples/supervised/train_tracks.py train \
  --cache ./datasets/cache --out ./runs/fold0-pretrained \
  --fold 0 --arm pretrained --pretrained ./private-checkpoints/lm-best.pt \
  --batch 8 --threads 2 --warmup-steps 5000 \
  --minimum-epochs 50 --patience 150 --epochs 5000 --max-hours 120

python examples/supervised/train_tracks.py evaluate \
  --cache ./datasets/cache --out ./runs/fold0-pretrained/oof \
  --fold 0 --arm pretrained --checkpoint ./runs/fold0-pretrained/best.pt --rc

python examples/supervised/aggregate_oof.py \
  --cache ./datasets/cache --run-root ./runs --out ./runs/pooled-oof
```

Repeat for folds 0–7 and `--arm scratch`. The aggregate requires all 16 models
to have reached source patience. Resume a partial model by adding
`--resume ./runs/fold0-pretrained/latest.pt` with the same scientific arguments.
Run independent processes with separate output directories; do not run two
writers on the same directory. Scheduling is platform-specific and omitted.

`build_cache.py` is a research building block, not a one-command reconstruction
of the complete published campaign. It does not contain every raw acquisition
and RNA-extension stage. The admitted cache contract requires `sequence.npy`,
`targets.npy`, `folds.npy`, and a `READY.json` containing content hashes and the
RC partner map. Do not manufacture a manifest to bypass biological admission.
See [methods](../../docs/public-supervised.md) for adaptations and limitations.

## Released-model inference

Extract one supervised bundle, then run:

```bash
python examples/supervised/predict.py --model-dir ./fold0-pretrained \
  --fasta ./sequence.fa --output ./prediction.npy --device cuda
```

The example accepts one 16,384-base A/C/G/T record. It returns 896 positions by
78 channels, with 16-base bins and 1,024 bases cropped from each end. Channel
identities and RC partners are in
[`channels.json`](../../benchmarks/public-chip74-rna4-v4/channels.json).
Predictions are in the campaign's transformed target space, not raw sequencing
counts. This example is a single forward prediction; the reported OOF metrics
use strand-aware forward/reverse-complement averaging. For held-out evaluation,
use only the model whose test fold contains the window: averaging all eight
models on these windows would include models trained on the evaluated region.
