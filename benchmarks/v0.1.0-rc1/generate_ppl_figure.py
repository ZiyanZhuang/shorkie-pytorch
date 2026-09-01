"""Generate the single-panel public PPL figure from the frozen JSON receipt.

Figure contract
---------------
Core conclusion: v1.1 D-best is close to, but slightly worse than, the official
Shorkie_LM on the identical reconstructed R64 validation contract.
Archetype: quantitative comparison.
Backend: Python/matplotlib only.
Statistics: paired 64-kb genomic-block bootstrap PPL-ratio 95% interval.
Reviewer risk: do not truncate the y-axis or present the paired ratio interval
as independent model error bars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["svg.hashsalt"] = "shorkie-pytorch-v0.1.0-rc1"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["font.size"] = 7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render(source: Path, output_stem: Path) -> list[Path]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload["metric"] != "window_mean_source_weight_normalized_ppl":
        raise ValueError("Unexpected metric contract")
    models = payload["models"]
    values = [float(item["overall_weighted_ppl"]) for item in models]
    labels = ["Official\nShorkie_LM", "v1.1 D-best\nmethod-rebuild"]
    colors = ["#767676", "#3775BA"]

    fig, ax = plt.subplots(figsize=(89 / 25.4, 70 / 25.4))
    bars = ax.bar(
        labels,
        values,
        width=0.58,
        color=colors,
        edgecolor="#272727",
        linewidth=0.7,
    )
    ax.set_ylim(0, 4.6)
    ax.set_ylabel("Overall weighted perplexity (PPL)\n(lower is better)")
    ax.set_title("Frozen R64 validation contract", fontsize=8, fontweight="bold", pad=7)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.07, f"{value:.6f}", ha="center", va="bottom")

    comparison = payload["paired_comparison"]
    low, high = comparison["ppl_ratio_ci95"]
    annotation = (
        f"D / official = {values[1] / values[0]:.6f}\n"
        f"paired 64-kb block bootstrap 95% CI: {low:.6f}-{high:.6f}"
    )
    ax.text(
        0.5,
        4.28,
        annotation,
        ha="center",
        va="center",
        fontsize=6.2,
        color="#272727",
    )
    ax.text(
        0.5,
        -0.22,
        "536 windows; seed 165; identical sample, mask, and RC plan",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.8,
        color="#4D4D4D",
    )
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.27)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".svg"), output_stem.with_suffix(".png")]
    metadata = {"Creator": "shorkie-pytorch v0.1.0-rc1", "Date": None}
    fig.savefig(outputs[0], bbox_inches="tight", metadata=metadata)
    fig.savefig(
        outputs[1],
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "shorkie-pytorch v0.1.0-rc1"},
    )
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--source", type=Path, default=here / "ppl_summary.json")
    parser.add_argument("--output-stem", type=Path, default=here / "overall_ppl")
    parser.add_argument("--check", action="store_true", help="Render and verify both outputs are non-empty.")
    args = parser.parse_args()
    outputs = render(args.source, args.output_stem)
    report = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in outputs}
    if args.check and any(item["bytes"] <= 0 for item in report.values()):
        raise SystemExit(2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
