"""Build deterministic model and GitHub release assets with SHA-256 receipts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


MODEL_FILES = {
    "README.md",
    "model.safetensors",
    "config.json",
    "evaluation.json",
    "training_summary.json",
    "release_verification.json",
    "LICENSE",
    "SHA256SUMS",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_model_sums(model_dir: Path) -> None:
    names = sorted(MODEL_FILES - {"SHA256SUMS"})
    unknown = {path.name for path in model_dir.iterdir() if path.is_file()} - MODEL_FILES
    missing = (MODEL_FILES - {"SHA256SUMS"}) - {path.name for path in model_dir.iterdir() if path.is_file()}
    if unknown or missing:
        raise ValueError(f"model file contract mismatch: unknown={sorted(unknown)}, missing={sorted(missing)}")
    lines = [f"{sha256(model_dir / name)}  {name}" for name in names]
    (model_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def deterministic_tar_gz(model_dir: Path, destination: Path, root_name: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=9) as zipped:
            with tarfile.open(fileobj=zipped, mode="w") as archive:
                for source in sorted(model_dir.iterdir(), key=lambda item: item.name):
                    if not source.is_file():
                        continue
                    info = archive.gettarinfo(str(source), arcname=f"{root_name}/{source.name}")
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--version", default="v0.1.0-rc1")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    dist_dir = args.dist_dir.resolve()
    release_dir = args.release_dir.resolve()
    release_dir.mkdir(parents=True, exist_ok=True)
    write_model_sums(model_dir)

    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError(f"expected one wheel and one sdist, got {wheels} and {sdists}")
    copied = []
    for source in (wheels[0], sdists[0]):
        target = release_dir / source.name
        shutil.copyfile(source, target)
        copied.append(target)

    bundle_name = f"ShorkieLM-165-method-rebuild-v1.1-D-{args.version}.tar.gz"
    bundle = release_dir / bundle_name
    deterministic_tar_gz(model_dir, bundle, bundle_name.removesuffix(".tar.gz"))
    copied.append(bundle)

    manifest = {
        "schema_version": 1,
        "release": args.version,
        "github_repository": "https://github.com/ZiyanZhuang/shorkie-pytorch",
        "model_repository": "https://huggingface.co/ZiyanZhuang/shorkie-lm-165-method-rebuild-v1.1",
        "assets": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(copied, key=lambda item: item.name)
        ],
        "model_safetensors_sha256": sha256(model_dir / "model.safetensors"),
    }
    manifest_path = release_dir / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksum_targets = sorted(copied + [manifest_path], key=lambda item: item.name)
    (release_dir / "SHA256SUMS").write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in checksum_targets) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
