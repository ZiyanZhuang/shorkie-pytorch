# Data contract

The streaming loader reads ZLIB-compressed TFRecord files without importing
TensorFlow in the training process. Each record contains:

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `sequence` | uint8 | `[16384]` | A/C/G/T/N encoded as 0/1/2/3/4 |
| `mask` | bool | `[16384]` | exon annotation mask |
| `repeat_mask` | bool | `[16384]` | soft-masked repeat/low-complexity mask |
| `species` | int32 | `[1]` | species index in `[0,164]` |

Shards are discovered from the corpus manifest and partitioned exclusively
between DataLoader workers. A valid corpus also supplies `corpus_qc.json` and
window metadata linking every record to accession, contig, coordinates, split,
masked fraction, shard, and record index.

The method-rebuild corpus used 16,384-bp windows, 4,096-bp stride and excluded
windows with repeat/low-complexity masking greater than 7%. R64 chromosomes
XI/XIII/XV were held out as validation and XII/XIV/XVI as test. This repository
does not redistribute that corpus.

