"""Write a deterministic 16,384-bp synthetic FASTA for API smoke testing."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sequence = "ACGT" * 4096
    lines = [">synthetic_api_smoke"] + [sequence[index:index + 80] for index in range(0, len(sequence), 80)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

