"""Fail-closed privacy, path, binary, provenance, and packaging audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


TEXT_SUFFIXES = {".cff", ".cfg", ".in", ".ini", ".json", ".md", ".py", ".ps1", ".sh", ".svg", ".toml", ".tsv", ".txt", ".yaml", ".yml"}
FORBIDDEN_SUFFIXES = {".ckpt", ".h5", ".key", ".npz", ".pem", ".pt", ".pth", ".safetensors", ".swanlab", ".tfr"}
EXCLUDED_PARTS = {".git", ".pytest_cache", "__pycache__", "audit", "build", "dist", "logs", "outputs", "reports"}
ALLOWED_BINARY_PATHS = {"benchmarks/v0.1.0-rc1/overall_ppl.png"}
SPECIAL_TEXT_NAMES = {"LICENSE", "NOTICE", "PKG-INFO", ".gitignore", ".gitattributes"}
_PRIVATE_MARKERS = (
    "86" + "135",
    "2500" + "10150",
    "pdc_" + "rfdiffusion",
    "zhuangzhou" + "@local",
)

PATTERNS = {
    "windows_absolute_path": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"),
    "private_unix_path": re.compile(r"/(?:data|home|mnt/afs|root)/"),
    "private_endpoint": re.compile(r"\b118\.145\.\d{1,3}\.\d{1,3}\b"),
    "private_identifiers": re.compile(r"\b(?:" + "|".join(map(re.escape, _PRIVATE_MARKERS)) + r")\b", re.I),
    "credential_assignment": re.compile(r"(?i)(?:api[_-]?key|password|cookie|secret|token)\s*[:=]\s*['\"][^'\"]{8,}"),
    "private_key_material": re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----"),
}
REQUIRED = {
    "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "CITATION.cff", "CONTRIBUTING.md",
    "SECURITY.md", "README.md", "README.zh-CN.md", "pyproject.toml",
    "src/shorkie_torch/model.py", "src/shorkie_torch/checkpoint.py",
    "docs/provenance.md", "docs/limitations.md",
    "benchmarks/v0.1.0-rc1/ppl_summary.json",
    "benchmarks/v0.1.0-rc1/parity_summary.json",
    "benchmarks/v0.1.0-rc1/release_manifest.json",
    "benchmarks/v0.1.0-rc1/claim_evidence.tsv",
    "benchmarks/v0.1.0-rc1/overall_ppl.svg",
    "benchmarks/v0.1.0-rc1/overall_ppl.png",
    "benchmarks/v0.1.0-rc1/generate_ppl_figure.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    findings: list[dict[str, object]] = []
    manifest = []
    files = [
        path for path in root.rglob("*")
        if path.is_file()
        and not any(
            part in EXCLUDED_PARTS or part.endswith(".egg-info")
            for part in path.relative_to(root).parts
        )
    ]
    present = {path.relative_to(root).as_posix() for path in files}
    for missing in sorted(REQUIRED - present):
        findings.append({"kind": "missing_required", "path": missing})
    for path in sorted(files):
        relative = path.relative_to(root).as_posix()
        manifest.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append({"kind": "forbidden_artifact", "path": relative})
        if relative in ALLOWED_BINARY_PATHS:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in SPECIAL_TEXT_NAMES:
            findings.append({"kind": "unexpected_binary_or_type", "path": relative})
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append({"kind": "not_utf8", "path": relative})
            continue
        for kind, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append({"kind": kind, "path": relative, "line": line})
    report = {
        "status": "PASS" if not findings else "BLOCKED",
        "root_name": root.name,
        "files_checked": len(files),
        "findings": findings,
        "manifest": manifest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "files_checked", "findings")}, indent=2))
    if findings:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
