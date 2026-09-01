from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shorkie_torch.evaluation import make_plan


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "v0.1.0-rc1"


def test_public_ppl_claim_matches_frozen_contract() -> None:
    payload = json.loads((BENCHMARK / "ppl_summary.json").read_text(encoding="utf-8"))
    assert payload["contract"]["windows"] == 536
    assert payload["contract"]["seed"] == 165
    assert payload["contract"]["masked_positions_per_pass"] == 2457
    values = {item["id"]: item["overall_weighted_ppl"] for item in payload["models"]}
    assert values["official_shorkie_lm"] == 3.60443
    assert abs(values["method_rebuild_v11_d_best"] - 3.621103976380612) < 1e-12
    assert values["method_rebuild_v11_d_best"] > values["official_shorkie_lm"]
    assert payload["paired_comparison"]["ppl_ratio_ci95"] == [1.004244, 1.005013]
    rerun = payload["public_rerun"]
    assert np.isclose(
        rerun["overall_weighted_ppl"],
        rerun["frozen_value"],
        rtol=rerun["tolerance"]["rtol"],
        atol=rerun["tolerance"]["atol"],
    )
    assert rerun["exon_weighted_ppl"] == 3.513289045555656
    assert rerun["repeat_weighted_ppl"] == 3.102039502170281


def test_public_parity_claim_is_pipeline_only() -> None:
    payload = json.loads((BENCHMARK / "parity_summary.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASS_PIPELINE_ONLY"
    assert max(payload["results"].values()) <= payload["thresholds"]["max_abs_target"]


def test_public_figure_and_claim_matrix_exist() -> None:
    for name in ("overall_ppl.svg", "overall_ppl.png", "claim_evidence.tsv"):
        path = BENCHMARK / name
        assert path.is_file() and path.stat().st_size > 0


def test_frozen_rc_and_mask_plan_contract() -> None:
    rc, plans = make_plan(536, 165)
    assert rc.shape == (536,)
    assert int(rc.sum()) == 271
    assert plans.shape == (536, 17199)
    for row in plans[:4]:
        assert np.unique(row[:16384]).size == 16384
