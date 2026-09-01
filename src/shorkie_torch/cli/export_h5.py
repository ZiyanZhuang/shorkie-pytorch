"""Export a trusted Keras H5 weight file to a pickle-free NPZ archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite an existing export")
    arrays: dict[str, np.ndarray] = {}
    with h5py.File(args.h5, "r") as archive:
        def visit(name: str, node: object) -> None:
            if isinstance(node, h5py.Dataset):
                arrays[name.replace("/", "__").replace(":", "_")] = node[()]
        archive.visititems(visit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report = {
        "source_file_name": args.h5.name,
        "source_sha256": sha256(args.h5),
        "output_file_name": args.output.name,
        "output_sha256": sha256(args.output),
        "array_count": len(arrays),
        "numpy_version": np.__version__,
        "h5py_version": h5py.__version__,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
