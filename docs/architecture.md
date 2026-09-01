# Architecture and mathematical contract

## Tensors

- Sequence length: 16,384 bp.
- Input: `[B,L,170]` channel-last.
- Channels 0–3: A/C/G/T one-hot; N is all-zero.
- Channel 4: MLM mask token.
- Channels 5–169: one of 165 species; *S. cerevisiae* index 109 maps to
  absolute channel 114.
- Output: `[B,L,4]` A/C/G/T logits.

The trunk uses seven residual downsampling blocks, eight relative-position
self-attention blocks at 128 positions and 384 channels, and seven U-Net decoder
blocks. The LM head returns linear logits.

## MLM objective

Each sequence samples `floor(0.15L)` positions without replacement. Selected
positions use BERT-style 10% unchanged, 10% random nucleotide, and 80% mask
token replacement. Reverse complementation, when enabled for training, flips
the input, target, exon mask, and repeat mask together.

For selected positions `i`, training uses:

`mean_i(CE_i * w_i)`, where
`w_i=(0.1 if exon else 1)*(0.1 if repeat else 1)`.

The source-equivalent training denominator is the number of selected positions,
not the sum of weights. Evaluation additionally reports weight-normalized and
unweighted NLL so a low scaled loss is not misinterpreted.

## Optimization

- Adam betas `(0.7,0.9)`, epsilon `1e-7`.
- Global gradient clip norm `0.1`.
- 20,000-step linear warmup to `1e-4` in the source schedule.
- Trunk Conv/Linear kernel L2 `1e-6`.
- Transformer dense/MHA kernel L2 override `1e-8`.
- Biases, normalization parameters, and learned scale parameters are excluded
  from L2.

