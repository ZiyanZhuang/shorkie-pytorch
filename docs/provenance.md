# Provenance and source mapping

| Public component | Local audited source | Upstream relationship |
|---|---|---|
| `model.py` | `shorkie_pt/model.py` | independent PyTorch translation of Shorkie/Baskerville architecture |
| `data.py` | `shorkie_pt/shard_stream.py` | TensorFlow-free reader for the author-compatible record schema |
| `h5_weights.py` | `shorkie_pt/h5_weights.py` | explicit Keras/NPZ to PyTorch tensor mapping |
| `evaluation.py` | `shorkie_eval/mlm.py` | source-style full-position MLM evaluation on method-rebuild data |
| `cli/train_mlm.py` | source-schedule trainer | PyTorch implementation of the frozen MLM training contract |

The original implementation targeted upstream commit
`c6003ce98db2156891c424f08352e900b97db55d`. Later release auditing froze the
official repository at `71bf5164c4f4448827cfc0d3b4715b736a3e6ba8` and used
the official public Shorkie_LM H5 for numerical parity and oracle evaluation.

No source file, checkpoint, corpus, or report from the original working tree is
modified by preparing this repository. Only allowlisted code is copied and then
refactored under a new package name.

## Public release identity

This release is maintained by Ziyan Zhuang (Tianjin University and Shenzhen
Loop Area Institute, `ziyan@tju.edu.cn`). The original Shorkie paper remains
the preferred scientific citation; this repository provides an optional
software citation for users of the PyTorch implementation or released weight.

The official project now lists the requester-pays `165_Saccharomycetales`
corpus. Users with appropriate GCP access should prefer that author-distributed
corpus. The independently rebuilt Ensembl Fungi release 59 corpus used here is
retained only as a method-rebuild audit and fallback and is not distributed by
this repository.
