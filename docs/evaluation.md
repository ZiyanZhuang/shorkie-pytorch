# Evaluation evidence

## TensorFlow to PyTorch parity

On a fixed real R64 window and random DNA, CPU FP32 comparison of the official
Shorkie_LM H5 and this PyTorch architecture produced:

| Tensor | Pearson | Max absolute error |
|---|---:|---:|
| probability | >0.999999999999 | 1.10e-6 |
| centered logit | >0.999999999999 | 5.48e-6 |
| first attention output | >0.999999999999 | 4.05e-6 |

This validates the checked LM forward path and weight mapping. It does not
automatically validate every supervised head or downstream pipeline.

## Frozen R64 validation comparison

The exact-plan comparison used 536 windows, seed 165, identical reverse-
complement decisions and seven masking passes covering every canonical base.

| Model | PPL | Gene PPL | Intergenic PPL |
|---|---:|---:|---:|
| official Shorkie_LM | 3.604430 | 3.469331 | 3.653980 |
| method-rebuild D best | 3.621104 | 3.473528 | 3.676541 |

D best is 0.4626% higher overall and therefore slightly worse. The paired
64-kb block-bootstrap PPL-ratio 95% interval is 1.004244-1.005013. The
residual difference is larger in repeat and intergenic strata. These are
method-rebuild results, not a reproduction of the authors' original-corpus
headline value. Machine-readable source values and the exact contract are in
`benchmarks/v0.1.0-rc1/ppl_summary.json`.

## Interpretation boundary

The overall PPL result is the only headline model-performance claim in this
release. Gene, intergenic, exon, and repeat values are descriptive strata on
the same reconstructed evaluation set. Numerical TensorFlow-to-PyTorch parity
validates a checked forward path; it does not establish equality of trained
weights, training data, or biological downstream utility.

The public safetensors rerun produced overall PPL 3.621134, an absolute
difference of 0.000030 from the frozen 3.621104 value and within the declared
`rtol=1e-4, atol=1e-5` gate. Its orientation-synchronized descriptive strata
were gene 3.473561, intergenic 3.676571, exon 3.513289, repeat 3.102040, and
exon+repeat 3.027286.

### RC-region correction

During release auditing, the older frozen evaluator was found to flip exon and
repeat weights for RC windows but reuse the unflipped exon/repeat region masks
when computing only those regional summaries. Overall PPL was calculated from
the correctly aligned weight vector; gene/intergenic masks were also returned
in the flipped orientation. Therefore the headline, gene, and intergenic
comparisons remain valid, while the legacy exon/repeat-only summaries are not
published as evidence. The public evaluator synchronizes sequence, annotation
weights, gene masks, and region masks before scoring.
