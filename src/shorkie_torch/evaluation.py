"""Source-locked full-position MLM evaluation on method-rebuild R64 windows."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .data import records
from .datasets import apply_mlm_mask, encode_sequence


SEQ_LENGTH = 16_384
MASK_SIZE = int(0.15 * SEQ_LENGTH)  # 2457, identical to author code.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(windows_tsv: Path, split: str, max_windows: int | None) -> list[dict[str, str]]:
    with windows_tsv.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle, delimiter="\t")
            if row["split"] == split and row["retained_reason"] == "kept"
        ]
    if max_windows is not None:
        rows = rows[:max_windows]
    if not rows:
        raise ValueError(f"no retained {split} windows in {windows_tsv}")
    return rows


def _load_examples(corpus_root: Path, rows: list[dict[str, str]]) -> list[dict[str, np.ndarray]]:
    output: list[dict[str, np.ndarray]] = []
    cached_name: str | None = None
    cached_records: list[dict[str, np.ndarray]] = []
    for row in rows:
        shard_name = row["shard"]
        if shard_name != cached_name:
            species_dir = row["accession"].replace(".", "_")
            shard_path = corpus_root / "per_species" / species_dir / "tfrecords" / shard_name
            cached_records = list(records(shard_path))
            cached_name = shard_name
        output.append(cached_records[int(row["shard_record_index"])])
    return output


def read_gene_intervals(gtf_gz: Path) -> dict[str, list[tuple[int, int]]]:
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with gzip.open(gtf_gz, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 5 and fields[2] == "gene":
                intervals[fields[0]].append((int(fields[3]) - 1, int(fields[4])))
    for contig, values in intervals.items():
        values.sort()
        merged: list[tuple[int, int]] = []
        for start, end in values:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        intervals[contig] = merged
    if not intervals:
        raise ValueError(f"no gene records found in {gtf_gz}")
    return dict(intervals)


def gene_mask(row: dict[str, str], intervals: dict[str, list[tuple[int, int]]]) -> np.ndarray:
    window_start, window_end = int(row["start"]), int(row["end"])
    output = np.zeros(SEQ_LENGTH, dtype=np.bool_)
    for start, end in intervals.get(row["gtf_contig"], []):
        if end <= window_start:
            continue
        if start >= window_end:
            break
        left, right = max(start, window_start) - window_start, min(end, window_end) - window_start
        output[left:right] = True
    return output


def make_plan(num_examples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Persistable seeded analogue of the source RC and masking draws."""
    rng = np.random.default_rng(seed)
    chunks = math.ceil(SEQ_LENGTH / MASK_SIZE)
    width = chunks * MASK_SIZE
    plans = np.empty((num_examples, width), dtype=np.int32)
    rc = rng.integers(0, 2, size=num_examples, dtype=np.uint8)
    for index in range(num_examples):
        order = rng.permutation(SEQ_LENGTH).astype(np.int32, copy=False)
        if width > SEQ_LENGTH:
            padding = rng.permutation(SEQ_LENGTH)[:width - SEQ_LENGTH].astype(np.int32)
            order = np.concatenate((order, padding))
        plans[index] = order
    if plans.shape[1] // MASK_SIZE != 7:
        raise RuntimeError("author masking partition contract drift")
    return rc, plans


def _reverse_complement_codes(sequence: np.ndarray) -> np.ndarray:
    output = np.asarray(sequence, dtype=np.uint8)[::-1].copy()
    canonical = output < 4
    output[canonical] = 3 - output[canonical]
    return output


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    maximum = logits.max(axis=-1, keepdims=True)
    shifted = logits - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _region(nll: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    valid = mask & np.isfinite(nll)
    if not np.any(valid):
        return math.nan, math.nan, 0
    weighted = float(np.sum(nll[valid] * weights[valid]) / np.sum(weights[valid]))
    unweighted = float(np.mean(nll[valid]))
    return weighted, unweighted, int(np.count_nonzero(valid))


def bootstrap_ppl(values: np.ndarray, seed: int, replicates: int = 2000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        means[index] = rng.choice(values, size=values.size, replace=True).mean()
    low, high = np.quantile(np.exp(means), [0.025, 0.975])
    return float(low), float(high)


def evaluate(
    adapter: Any,
    *,
    corpus_root: Path,
    windows_tsv: Path,
    gtf_gz: Path,
    split: str,
    seed: int,
    batch_windows: int,
    max_windows: int | None = None,
    species_mode: str = "normal",
    sequence_mode: str = "genomic",
    rc_mode: str = "seeded",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    if species_mode not in {"normal", "zero", "random"}:
        raise ValueError(f"unsupported species mode: {species_mode}")
    if sequence_mode not in {"genomic", "random"}:
        raise ValueError(f"unsupported sequence mode: {sequence_mode}")
    if rc_mode not in {"seeded", "none"}:
        raise ValueError(f"unsupported RC mode: {rc_mode}")
    rows = _read_rows(windows_tsv, split, max_windows)
    examples = _load_examples(corpus_root, rows)
    genes = read_gene_intervals(gtf_gz)
    if sequence_mode == "random":
        sequence_rng = np.random.default_rng(seed + 1)
        examples = [
            {
                **example,
                "sequence": sequence_rng.integers(0, 4, size=SEQ_LENGTH, dtype=np.uint8),
                "mask": np.zeros(SEQ_LENGTH, dtype=np.bool_),
                "repeat_mask": np.zeros(SEQ_LENGTH, dtype=np.bool_),
            }
            for example in examples
        ]
    plan_rc, plans = make_plan(len(rows), seed)
    if rc_mode == "none":
        plan_rc.fill(0)
    random_species = np.random.default_rng(seed + 2).integers(0, 165, size=len(rows), dtype=np.int32)
    genes_by_window = [gene_mask(row, genes) for row in rows]
    oriented_examples: list[dict[str, np.ndarray]] = []
    for example, gene, reverse in zip(examples, genes_by_window, plan_rc):
        oriented = {name: np.asarray(value).copy() for name, value in example.items()}
        if reverse:
            oriented["sequence"] = _reverse_complement_codes(oriented["sequence"])
            oriented["mask"] = oriented["mask"][::-1].copy()
            oriented["repeat_mask"] = oriented["repeat_mask"][::-1].copy()
            gene[:] = gene[::-1]
        oriented_examples.append(oriented)
    nll_by_window = [np.zeros(SEQ_LENGTH, dtype=np.float32) for _ in rows]
    seen_by_window = [np.zeros(SEQ_LENGTH, dtype=np.bool_) for _ in rows]

    for pass_index in range(7):
        for batch_start in range(0, len(rows), batch_windows):
            batch_end = min(batch_start + batch_windows, len(rows))
            inputs = []
            for index in range(batch_start, batch_end):
                species = int(oriented_examples[index]["species"][0])
                encoded = encode_sequence(oriented_examples[index]["sequence"], species)
                if species_mode == "zero":
                    encoded[:, 5:] = 0.0
                elif species_mode == "random":
                    encoded[:, 5:] = 0.0
                    encoded[:, 5 + int(random_species[index])] = 1.0
                positions = plans[index, pass_index * MASK_SIZE:(pass_index + 1) * MASK_SIZE]
                inputs.append(apply_mlm_mask(encoded, positions))
            logits = adapter.logits(np.stack(inputs))
            log_probabilities = _log_softmax(logits)
            for local, index in enumerate(range(batch_start, batch_end)):
                positions = plans[index, pass_index * MASK_SIZE:(pass_index + 1) * MASK_SIZE]
                first = ~seen_by_window[index][positions]
                fresh = positions[first]
                targets = oriented_examples[index]["sequence"][fresh]
                canonical = targets < 4
                canonical_fresh, canonical_targets = fresh[canonical], targets[canonical]
                nll_by_window[index][canonical_fresh] = -log_probabilities[
                    local, canonical_fresh, canonical_targets
                ]
                seen_by_window[index][positions] = True

    results: list[dict[str, Any]] = []
    global_weighted_numerator = global_weight_sum = global_unweighted_sum = global_valid = 0.0
    for row, example, gene, nll, seen in zip(
        rows, oriented_examples, genes_by_window, nll_by_window, seen_by_window
    ):
        valid = seen
        if not bool(np.all(valid)):
            raise RuntimeError(f"position coverage failure: {row['contig']}:{row['start']}")
        exon, repeat = example["mask"], example["repeat_mask"]
        weights = np.where(exon, 0.1, 1.0) * np.where(repeat, 0.1, 1.0)
        masks = {
            "all": valid,
            "gene": gene,
            "intergenic": ~gene,
            "exon": exon,
            "repeat": repeat,
            "exon_repeat": exon & repeat,
        }
        result: dict[str, Any] = {
            "sample_id": f"{row['accession']}:{row['contig']}:{row['start']}-{row['end']}",
            "accession": row["accession"], "contig": row["contig"],
            "start": int(row["start"]), "end": int(row["end"]), "split": split,
        }
        for name, mask in masks.items():
            weighted, unweighted, count = _region(nll, weights, mask)
            result[f"{name}_weighted_nll"] = weighted
            result[f"{name}_weighted_ppl"] = math.exp(weighted) if math.isfinite(weighted) else math.nan
            result[f"{name}_unweighted_nll"] = unweighted
            result[f"{name}_unweighted_ppl"] = math.exp(unweighted) if math.isfinite(unweighted) else math.nan
            result[f"{name}_positions"] = count
        result["source_scaled_loss"] = float(np.sum(nll[valid] * weights[valid]) / SEQ_LENGTH)
        results.append(result)
        global_weighted_numerator += float(np.sum(nll[valid] * weights[valid]))
        global_weight_sum += float(np.sum(weights[valid]))
        global_unweighted_sum += float(np.sum(nll[valid]))
        global_valid += int(np.count_nonzero(valid))

    all_window_nll = np.asarray([row["all_weighted_nll"] for row in results], dtype=np.float64)
    ci_low, ci_high = bootstrap_ppl(all_window_nll, seed)
    sample_bytes = "\n".join(row["sample_id"] for row in results).encode("utf-8")
    summary = {
        "status": "PASS",
        "equivalence_class": "method-rebuild R64; source full-position masking; FP32 metric path",
        "precision": "model FP32, log-softmax FP32, no source-script float16 prediction cache",
        "orientation": "seeded per-window RC" if rc_mode == "seeded" else "forward only",
        "model": adapter.metadata.to_dict(),
        "split": split, "seed": seed, "species_mode": species_mode,
        "sequence_mode": sequence_mode, "rc_mode": rc_mode, "rc_count": int(plan_rc.sum()),
        "windows": len(results), "mask_size": MASK_SIZE, "passes_per_window": 7,
        "plan_shape": list(plans.shape),
        "mask_rc_plan_content_sha256": hashlib.sha256(plan_rc.tobytes() + plans.tobytes()).hexdigest(),
        "sample_list_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "windows_tsv_sha256": sha256(windows_tsv), "gtf_sha256": sha256(gtf_gz),
        "window_mean_weighted_nll": float(all_window_nll.mean()),
        "window_mean_weighted_ppl": float(math.exp(all_window_nll.mean())),
        "window_mean_weighted_ppl_ci95": [ci_low, ci_high],
        "global_weighted_nll": global_weighted_numerator / global_weight_sum,
        "global_weighted_ppl": math.exp(global_weighted_numerator / global_weight_sum),
        "global_unweighted_nll": global_unweighted_sum / global_valid,
        "global_unweighted_ppl": math.exp(global_unweighted_sum / global_valid),
        "wall_seconds": time.perf_counter() - started,
    }
    for region in ("gene", "intergenic", "exon", "repeat", "exon_repeat"):
        values = np.asarray([row[f"{region}_weighted_nll"] for row in results], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[f"window_mean_{region}_weighted_nll"] = float(finite.mean()) if finite.size else math.nan
        summary[f"window_mean_{region}_weighted_ppl"] = float(np.exp(finite.mean())) if finite.size else math.nan
    return results, summary


def write_results(run_dir: Path, model_id: str, rows: Iterable[dict[str, Any]], summary: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=False)
    rows = list(rows)
    with (run_dir / f"{model_id}_window_metrics.tsv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    (run_dir / f"{model_id}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

