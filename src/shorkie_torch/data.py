"""TensorFlow-free streaming reader for the fixed Shorkie ZLIB TFRecord schema.

The corpus deliberately stays in the author's TFRecord format.  This module
implements only the small protobuf subset used by its four bytes-list fields,
so PyTorch training does not import TensorFlow or materialize the 165-genome
corpus in RAM.  Each DataLoader worker owns a disjoint shard subset.
"""
from __future__ import annotations

import random
import struct
import zlib
from multiprocessing import Value
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

LENGTH = 16_384
FIELDS = {"sequence", "mask", "repeat_mask", "species"}


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("truncated protobuf varint")
        byte = data[pos]; pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > 63:
            raise ValueError("oversized protobuf varint")


def _fields(message: bytes) -> Iterator[tuple[int, int, bytes | int]]:
    pos = 0
    while pos < len(message):
        tag, pos = _varint(message, pos)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value, pos = _varint(message, pos)
        elif wire == 2:
            size, pos = _varint(message, pos)
            value, pos = message[pos:pos + size], pos + size
            if len(value) != size:
                raise ValueError("truncated protobuf bytes field")
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield number, wire, value


def _first_bytes_list(feature: bytes) -> bytes:
    # Feature { bytes_list: BytesList { value: bytes } }.
    outer = list(_fields(feature))
    if len(outer) != 1 or outer[0][0] != 1 or outer[0][1] != 2:
        raise ValueError("expected a Feature.bytes_list")
    values = [value for number, wire, value in _fields(outer[0][2]) if number == 1 and wire == 2]
    if len(values) != 1 or not isinstance(values[0], bytes):
        raise ValueError("expected exactly one BytesList value")
    return values[0]


def decode_example(serialized: bytes) -> dict[str, np.ndarray]:
    """Decode exactly the writer's tf.train.Example schema, rejecting drift."""
    root = list(_fields(serialized))
    if len(root) != 1 or root[0][0] != 1 or root[0][1] != 2:
        raise ValueError("expected Example.features")
    parsed: dict[str, np.ndarray] = {}
    for number, wire, entry in _fields(root[0][2]):  # Features.feature map entries
        if number != 1 or wire != 2:
            continue
        parts = list(_fields(entry))
        key = next((x[2] for x in parts if x[0] == 1 and x[1] == 2), None)
        feature = next((x[2] for x in parts if x[0] == 2 and x[1] == 2), None)
        if not isinstance(key, bytes) or not isinstance(feature, bytes):
            raise ValueError("malformed Features map entry")
        name = key.decode("utf-8")
        raw = _first_bytes_list(feature)
        dtype = np.int32 if name == "species" else (np.uint8 if name == "sequence" else np.bool_)
        parsed[name] = np.frombuffer(raw, dtype=dtype).copy()
    if set(parsed) != FIELDS:
        raise ValueError(f"schema mismatch: {sorted(parsed)}")
    if any(parsed[k].shape != (LENGTH,) for k in ("sequence", "mask", "repeat_mask")) or parsed["species"].shape != (1,):
        raise ValueError("TFRecord feature shape mismatch")
    return parsed


def records(path: Path) -> Iterator[dict[str, np.ndarray]]:
    """Read a complete ZLIB-compressed TFRecord shard; framing CRC is trusted
    after corpus construction's TensorFlow full-readback and SHA-256 manifest."""
    raw = zlib.decompress(path.read_bytes())
    pos = 0
    while pos < len(raw):
        if pos + 12 > len(raw):
            raise ValueError(f"truncated TFRecord header: {path}")
        size = struct.unpack_from("<Q", raw, pos)[0]; pos += 12  # length + masked CRC32C
        end = pos + size
        if end + 4 > len(raw):
            raise ValueError(f"truncated TFRecord payload: {path}")
        yield decode_example(raw[pos:end])
        pos = end + 4  # skip payload masked CRC32C


class ShorkieShardDataset(IterableDataset[dict[str, torch.Tensor]]):
    def __init__(self, corpus_root: str | Path, split: str = "train", *, shuffle_shards: bool = True, seed: int = 0) -> None:
        super().__init__()
        self.root, self.split, self.shuffle_shards, self.seed = Path(corpus_root), split, shuffle_shards, seed
        # Shared so persistent DataLoader workers observe epoch reshuffles
        # without being respawned.  This mirrors upstream's make_dataset()
        # call at each source-training epoch.
        self._epoch = Value("q", 0)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = epoch

    @property
    def epoch(self) -> int:
        with self._epoch.get_lock():
            return self._epoch.value

    def shard_paths(self) -> list[Path]:
        paths = sorted((self.root / "per_species").glob(f"*/tfrecords/{self.split}-*.tfr"))
        if not paths:
            raise FileNotFoundError(f"no {self.split} shards beneath {self.root}")
        return paths

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        paths = self.shard_paths()
        info = get_worker_info()
        worker_id, workers = (info.id, info.num_workers) if info else (0, 1)
        with self._epoch.get_lock():
            epoch = self._epoch.value
        rng = random.Random(self.seed + 1_000_003 * epoch)
        if self.shuffle_shards:
            rng.shuffle(paths)
        # shard_id 是训练遥测元数据，不参与模型输入或损失计算。
        for shard_id, path in enumerate(paths):
            if shard_id % workers != worker_id:
                continue
            for example in records(path):
                item = {name: torch.from_numpy(value) for name, value in example.items()}
                item["_shard_id"] = torch.tensor(shard_id, dtype=torch.int32)
                yield item


def make_shorkie_stream_loader(corpus_root: str | Path, split: str, *, batch_size: int, num_workers: int = 0, seed: int = 0, pin_memory: bool = True, prefetch_factor: int = 4) -> tuple[ShorkieShardDataset, DataLoader]:
    """Create a bounded-prefetch loader without changing record order or schema.

    ``prefetch_factor`` is intentionally an infrastructure-only control: every
    batch still comes from the immutable TFRecord corpus and is transformed by
    the source-locked trainer unchanged.  It is ignored when loading in the
    main process because PyTorch rejects it for ``num_workers == 0``.
    """
    dataset = ShorkieShardDataset(corpus_root, split, shuffle_shards=(split == "train"), seed=seed)
    kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": (num_workers > 0),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return dataset, DataLoader(dataset, **kwargs)



# Public, concise alias.
make_stream_loader = make_shorkie_stream_loader

