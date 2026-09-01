"""Deterministic input construction for source-locked Shorkie evaluation."""

from __future__ import annotations

import numpy as np


DNA_CHANNELS = 4
MASK_CHANNEL = 4
SPECIES_INDEX_SCEREVISIAE = 109
SPECIES_ABSOLUTE_CHANNEL_SCEREVISIAE = 114
INPUT_CHANNELS = 170


def encode_sequence(sequence: np.ndarray, species_index: int = SPECIES_INDEX_SCEREVISIAE) -> np.ndarray:
    """Encode A/C/G/T/N=0/1/2/3/4 as DNA+mask+165-species channels."""
    sequence = np.asarray(sequence, dtype=np.uint8)
    if sequence.ndim != 1:
        raise ValueError("sequence must be one-dimensional")
    if np.any(sequence > 4):
        raise ValueError("sequence codes must be in [0,4]")
    if not 0 <= species_index < 165:
        raise ValueError("species_index must be in [0,164]")
    output = np.zeros((sequence.size, INPUT_CHANNELS), dtype=np.float32)
    valid = sequence < 4
    output[np.nonzero(valid)[0], sequence[valid]] = 1.0
    output[:, 5 + species_index] = 1.0
    return output


def apply_mlm_mask(encoded: np.ndarray, positions: np.ndarray) -> np.ndarray:
    output = np.array(encoded, copy=True)
    positions = np.asarray(positions, dtype=np.int64)
    output[positions, :DNA_CHANNELS] = 0.0
    output[positions, MASK_CHANNEL] = 1.0
    return output


