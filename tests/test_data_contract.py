from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

from shorkie_torch.data import decode_example, make_stream_loader, records


def _varint(value: int) -> bytes:
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(output)


def _field(number: int, payload: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _example(fields: dict[str, bytes]) -> bytes:
    entries = []
    for name, raw in fields.items():
        bytes_list = _field(1, raw)
        feature = _field(1, bytes_list)
        entries.append(_field(1, _field(1, name.encode()) + _field(2, feature)))
    return _field(1, b"".join(entries))


def _write_shard(path: Path, species: int) -> dict[str, np.ndarray]:
    item = {
        "sequence": np.arange(16_384, dtype=np.uint16).astype(np.uint8) % 5,
        "mask": np.arange(16_384) % 7 == 0,
        "repeat_mask": np.arange(16_384) % 11 == 0,
        "species": np.asarray([species], dtype=np.int32),
    }
    payload = _example({name: value.tobytes() for name, value in item.items()})
    frame = struct.pack("<Q", len(payload)) + bytes(4) + payload + bytes(4)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zlib.compress(frame))
    return item


def test_tfrecord_decode_readback_and_loader(tmp_path: Path) -> None:
    path = tmp_path / "per_species" / "a" / "tfrecords" / "train-000.tfr"
    expected = _write_shard(path, 109)
    raw = zlib.decompress(path.read_bytes())
    size = struct.unpack_from("<Q", raw, 0)[0]
    decoded = decode_example(raw[12:12 + size])
    for name in expected:
        assert np.array_equal(decoded[name], expected[name])
    assert len(list(records(path))) == 1
    _, loader = make_stream_loader(tmp_path, "train", batch_size=1, num_workers=0, pin_memory=False)
    batch = next(iter(loader))
    assert tuple(batch["sequence"].shape) == (1, 16_384)
    assert int(batch["species"][0, 0]) == 109
    assert "_shard_id" in batch


def test_two_workers_own_disjoint_shards(tmp_path: Path) -> None:
    for index in range(4):
        path = tmp_path / "per_species" / f"s{index}" / "tfrecords" / f"train-{index:03}.tfr"
        _write_shard(path, index)
    _, loader = make_stream_loader(tmp_path, "train", batch_size=1, num_workers=2, pin_memory=False, seed=7)
    species = sorted(int(batch["species"][0, 0]) for batch in loader)
    assert species == [0, 1, 2, 3]

