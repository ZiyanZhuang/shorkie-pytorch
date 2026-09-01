# Third-party notices

## Shorkie and Baskerville-yeast

This project reproduces architecture and training behavior described by the
[official Shorkie repository](https://github.com/calico/shorkie-paper) and its
`baskerville-yeast` submodule. Both source trees are distributed under the
Apache License 2.0. Their copyright and attribution notices remain controlling
for upstream material.

The initial implementation was compared with upstream commit
`c6003ce98db2156891c424f08352e900b97db55d`. Release validation later used the
official repository state `71bf5164c4f4448827cfc0d3b4715b736a3e6ba8` and
public Shorkie_LM weights. These identifiers describe provenance; this project
is not an upstream release.

## Training data

The method-rebuild model was trained from Ensembl Fungi release 59 assemblies
and annotations after an independently implemented repeat-masking and windowing
pipeline. The reconstructed corpus is not bundled here and must not be called
the authors' original corpus. Ensembl states that project-produced data are
freely available, while downstream users remain responsible for any third-party
constraints attached to individual source records.

## Runtime dependencies

PyTorch, NumPy, safetensors, TensorFlow, h5py, pytest, and optional experiment
tracking packages retain their own licenses. See `pyproject.toml` for the
declared dependency boundary.

