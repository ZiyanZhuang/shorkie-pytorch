---
license: apache-2.0
language:
  - en
tags:
  - genomics
  - yeast
  - shorkie
  - pytorch
  - research
---

# Shorkie public ChIP74 / RNA4 adaptation, normalized-lograte-v4

Unofficial public-data adaptation of [Shorkie](https://elifesciences.org/reviewed-preprints/112217).
The original architecture and scientific method belong to the original authors;
this release documents a PyTorch reproduction and explicit public-data changes.

## What is released

Eight genomic folds, each with a scratch and a pretrained-initialized supervised
model: 16 independent bundles named `fold0-scratch` through `fold7-pretrained`.
Each contains `config.json`, pure `model.safetensors`, and a sanitized
`training_summary.json`. Optimizer state and private training paths are omitted.
These bundles are for inference; exact optimizer resume is not supported.

The input is 16,384 bp; outputs are 896 bins by 78 channels at 16-bp spacing,
with 1,024 bp cropped from each end. Scratch uses four base channels (12,090,734
parameters). The pretrained arm uses 170 channels (12,266,030 parameters),
including the S. cerevisiae indicator at absolute column 114. Base order is ACGT.
Outputs are nonnegative predictions in the transformed target space; they are
not unprocessed read counts or experimentally validated expression estimates.

Install [the code](https://github.com/ZiyanZhuang/shorkie-pytorch) at the matching
release, then use `shorkie_torch.load_pretrained(bundle_directory)`. The repository
provides `examples/supervised/predict.py` for one FASTA record. Download/extract
the chosen fold directory before loading; the campaign root is not one model.
`channels.json` maps all outputs to public source accessions and checksums.

## Data, training and evaluation

Public coverage targets comprise 74 ChIP tracks and four published nascent RNA
tag tracks. RNA represents two wild-type conditions with two strands each,
not four independent replicates. It is not the paper's full induction RNA panel.
An unresolved read1 filename versus read2 metadata discrepancy remains an
explicit source-semantics limitation. Private strain sequencing is not included.

The admitted cache contains 1,855 nuclear genomic windows in eight folds.
Training used batch 8, seed 44, FP32 with TF32 disabled, Keras-style Adam,
5,000-step warmup, gradient clipping 0.1, and validation patience 150 after a
50-epoch minimum. Learning rates were 0.0005 for scratch and 0.00002 for the
pretrained arm. The normalized-lograte-v4 objective and full-permutation shuffle
are documented adaptations, not claims of exact upstream training parity.
All 16 runs stopped under the configured patience rule; that does not prove
global convergence or optimality.

OOF predictions use the held-out fold model with strand-aware RC averaging.
All 1,855 windows occur once. Do not average all eight fold models to reevaluate
these same windows: that includes models trained on the evaluation region.

| Pooled OOF track mean | Scratch | Pretrained |
|---|---:|---:|
| ChIP Pearson | 0.203469 | 0.223263 |
| ChIP Spearman | 0.178265 | 0.195065 |
| ChIP R2 | 0.058090 | 0.072387 |
| RNA Pearson | 0.589980 | 0.633575 |
| RNA Spearman | 0.766001 | 0.793557 |
| RNA R2 | 0.348359 | 0.401319 |

These are descriptive public-data results. Different input channels and
learning rates confound a single-factor causal claim about pretraining benefit.
Different datasets, assays, preprocessing and folds prevent a matched ranking
against the original paper. The older released LM's perplexity result must not
be attributed to this campaign's distinct LM initialization.

## Intended use and limits

Research into yeast sequence-to-signal prediction and reproducibility. Not
validated for promoter/terminator generation, intron optimization, or wet-lab
expression improvement. No raw sequencing data, BigWigs, training cache, or
private data are redistributed. Consult accession-specific source terms before
redistributing data. Code and derived weight licensing does not override those
terms. See the repository's methods, per-track metrics, training records and
Chinese technical report for the full scope of completed and uncompleted work.
