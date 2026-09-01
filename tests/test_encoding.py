from __future__ import annotations

import numpy as np

from shorkie_torch.datasets import apply_mlm_mask, encode_sequence


def test_sequence_species_and_mask_channels() -> None:
    sequence = np.asarray([0, 1, 2, 3, 4], dtype=np.uint8)
    encoded = encode_sequence(sequence, species_index=109)
    assert encoded.shape == (5, 170)
    assert np.array_equal(encoded[:4, :4], np.eye(4, dtype=np.float32))
    assert encoded[4, :4].sum() == 0
    assert np.all(encoded[:, 114] == 1)
    masked = apply_mlm_mask(encoded, np.asarray([1, 4]))
    assert np.all(masked[[1, 4], :4] == 0)
    assert np.all(masked[[1, 4], 4] == 1)

