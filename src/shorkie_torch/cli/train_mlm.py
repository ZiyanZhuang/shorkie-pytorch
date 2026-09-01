"""Source-schedule Shorkie-LM pretraining over the 165-genome rebuild corpus.

This is deliberately separate from the continuous-step smoke trainer.  It
implements the released Shorkie-LM run control: 150 train updates per epoch,
full validation with five independently masked repeats, lowest-validation-loss
selection, and elastic-safe checkpoint resume at completed epoch boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from shorkie_torch import (ShorkieLM, frozen_weighted_mlm_loss, make_mlm_batch,
    make_shorkie_adam, make_stream_loader, set_linear_warmup,
    set_shorkie_learning_rate, shorkie_l2_penalty)

LENGTH, BATCH_SIZE = 16_384, 8


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: object) -> None:
    """每 epoch 只追加一次完整分布，避免 checkpoint 重复序列化整个遥测历史。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush(); os.fsync(handle.fileno())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_torch_save(value: object, path: Path) -> None:
    """Never replace the last durable elastic checkpoint with a partial write."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def capture_rng_state() -> dict[str, object]:
    """保存所有会影响下一 epoch 随机性的状态，供精确 continuation 使用。"""
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, object]) -> None:
    """必须在模型/优化器恢复后调用，覆盖初始化过程消耗的随机数。"""
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError(f"RNG state schema mismatch: {sorted(state)}")
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"]); torch.cuda.set_rng_state_all(state["torch_cuda"])


def l2_norm(tensors: list[torch.Tensor]) -> float:
    """在 GPU 上汇总，再仅传回一个标量；每 epoch 调用一次。"""
    if not tensors:
        return 0.0
    total = torch.zeros((), device=tensors[0].device, dtype=torch.float64)
    for tensor in tensors:
        total += tensor.detach().double().square().sum()
    return float(total.sqrt().cpu())


def model_state_summary(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, float | int]:
    """记录可定位 late-stage regime shift 的低开销 epoch 级摘要。"""
    values: dict[str, float | int] = {"parameter_l2_norm": l2_norm(list(model.parameters()))}
    moment_1, moment_2 = [], []
    for state in optimizer.state.values():
        if torch.is_tensor(state.get("exp_avg")):
            moment_1.append(state["exp_avg"])
        if torch.is_tensor(state.get("exp_avg_sq")):
            moment_2.append(state["exp_avg_sq"])
    values["adam_exp_avg_l2_norm"] = l2_norm(moment_1)
    values["adam_exp_avg_sq_l2_norm"] = l2_norm(moment_2)
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    means = torch.cat([m.running_mean.detach().float() for m in bn_layers])
    variances = torch.cat([m.running_var.detach().float() for m in bn_layers])
    values.update({
        "bn_layer_count": len(bn_layers), "bn_running_mean_mean": float(means.mean()),
        "bn_running_mean_abs_mean": float(means.abs().mean()), "bn_running_mean_std": float(means.std(unbiased=False)),
        "bn_running_var_mean": float(variances.mean()), "bn_running_var_min": float(variances.min()),
        "bn_running_var_max": float(variances.max()),
    })
    return values


def distribution_summary(counts: torch.Tensor, prefix: str) -> dict[str, float | int]:
    """将物种或 shard 计数压缩为 SwanLab 可绘制摘要。"""
    total = int(counts.sum())
    nonzero = counts[counts > 0].float()
    if total == 0:
        return {f"{prefix}_unique": 0, f"{prefix}_max_fraction": 0.0, f"{prefix}_entropy": 0.0}
    probabilities = nonzero / total
    return {
        f"{prefix}_unique": int(nonzero.numel()), f"{prefix}_max_fraction": float(probabilities.max()),
        f"{prefix}_entropy": float(-(probabilities * probabilities.log()).sum()),
    }


def one_batch(raw: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    codes = raw["sequence"].to(device, non_blocking=True)
    exon = raw["mask"].to(device, non_blocking=True).float()
    repeat = raw["repeat_mask"].to(device, non_blocking=True).float()
    species = F.one_hot(raw["species"].to(device, non_blocking=True).long().squeeze(1), 165).float()
    bases = F.one_hot(codes.long(), 5)[..., :4].float()
    return bases, species, exon, repeat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=5_001)
    ap.add_argument("--steps-per-epoch", type=int, default=150)
    ap.add_argument("--repeat-eval", type=int, default=5)
    ap.add_argument("--train-epochs-min", type=int, default=100)
    ap.add_argument("--patience", type=int, default=1_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--warmup-steps", type=int, default=20_000)
    ap.add_argument("--base-lr", type=float, default=1e-4)
    ap.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    ap.add_argument("--final-lr", type=float, default=1e-4)
    ap.add_argument("--decay-steps", type=int, default=0)
    ap.add_argument("--branch-start-step", type=int, default=-1,
                    help="Historical global step at branch origin; skips replaying warmup")
    ap.add_argument("--branch-id", default="source")
    ap.add_argument("--reset-optimizer-state", action="store_true")
    ap.add_argument("--checkpoint-every-epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=165)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--resume-legacy-v11", action="store_true", help="仅允许已审计的旧 v1.1 checkpoint 迁移；不能保证逐 batch 重放")
    ap.add_argument("--swanlab", action="store_true")
    ap.add_argument("--swanlab-project", default="shorkie165-method-rebuild")
    ap.add_argument("--swanlab-name", default="shorkie165-source-schedule")
    a = ap.parse_args()
    if a.epochs < 1 or a.steps_per_epoch < 1 or a.repeat_eval < 1 or a.workers < 0:
        raise ValueError("epochs, steps-per-epoch and repeat-eval must be positive; workers must be non-negative")
    if a.base_lr <= 0 or a.final_lr < 0 or a.final_lr > a.base_lr:
        raise ValueError("require base_lr>0 and 0<=final_lr<=base_lr")
    if a.lr_schedule == "cosine" and a.decay_steps <= 0:
        raise ValueError("cosine schedule requires --decay-steps > 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    qc = a.corpus_root / "corpus_qc.json"
    if not qc.is_file():
        raise FileNotFoundError(qc)
    random.seed(a.seed); np.random.seed(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.set_num_threads(4)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # 固定 16,384bp shape，允许 cuDNN 选择最快确定形状算法。
    torch.backends.cudnn.allow_tf32 = True
    train_ds, train_loader = make_stream_loader(a.corpus_root, "train", batch_size=BATCH_SIZE, num_workers=a.workers, seed=a.seed, pin_memory=True, prefetch_factor=a.prefetch_factor)
    valid_ds, valid_loader = make_stream_loader(a.corpus_root, "valid", batch_size=BATCH_SIZE, num_workers=a.workers, seed=a.seed + 17, pin_memory=True, prefetch_factor=a.prefetch_factor)
    model = ShorkieLM().to(device); optimizer = make_shorkie_adam(model)
    package_dir = Path(__file__).resolve().parents[1]
    code_hashes = {
        "trainer": sha256(Path(__file__).resolve()),
        "model": sha256(package_dir / "model.py"),
        "stream": sha256(package_dir / "data.py"),
    }
    config = {
        "scope": "method-rebuild 165 corpus; source-schedule PyTorch reproduction",
        "corpus_root": str(a.corpus_root.resolve()), "corpus_qc_sha256": sha256(qc),
        "epochs": a.epochs, "steps_per_epoch": a.steps_per_epoch,
        "repeat_eval": a.repeat_eval, "train_epochs_min": a.train_epochs_min,
        "patience": a.patience, "batch_size": BATCH_SIZE, "workers": a.workers,
        "prefetch_factor": a.prefetch_factor, "warmup_steps": a.warmup_steps,
        "base_lr": a.base_lr, "lr_schedule": a.lr_schedule,
        "final_lr": a.final_lr, "decay_steps": a.decay_steps,
        "branch_start_step": a.branch_start_step, "branch_id": a.branch_id,
        "reset_optimizer_state": a.reset_optimizer_state,
        "seed": a.seed, "model_config": asdict(model.cfg), "code_sha256": code_hashes,
        "cuda_execution": {"bf16_autocast": True, "matmul_tf32": True, "cudnn_tf32": True, "cudnn_benchmark": True},
        "source_contract": "upstream params.json/run loop: batch8, 150 updates/epoch, valid repeat_eval=5, warmup20k, Adam(.7,.9,eps1e-7), clipnorm .1",
    }
    write_json(a.out.with_name("run_config.json"), config)
    start_epoch, global_step, best_valid, unimproved, metrics = 0, 0, float("inf"), 0, []
    resume_contract = {key: config[key] for key in (
        "corpus_qc_sha256", "batch_size", "steps_per_epoch", "repeat_eval", "warmup_steps", "seed",
        "workers", "prefetch_factor", "model_config", "code_sha256", "source_contract", "cuda_execution",
        "base_lr", "lr_schedule", "final_lr", "decay_steps", "branch_start_step",
        "branch_id", "reset_optimizer_state",
    )}
    resume_mode, resume_source, swanlab_run_id = "fresh", None, None
    if a.resume:
        # checkpoint 内含 CPU RNG ByteTensor；整体映射到 CUDA 会使 RNG 恢复失败。
        # load_state_dict 会将模型/优化器 state 迁移到已在 CUDA 上的参数。
        saved = torch.load(a.resume, map_location="cpu", weights_only=False)
        saved_contract = saved.get("resume_contract")
        if saved_contract == resume_contract:
            if "rng_state" not in saved or "sampler_state" not in saved:
                raise ValueError("new-format resume checkpoint lacks RNG or sampler state")
            expected_epoch = int(saved["epoch_completed"]) + 1
            if saved["sampler_state"].get("next_train_epoch") != expected_epoch:
                raise ValueError("sampler_state next_train_epoch disagrees with checkpoint epoch")
            resume_mode = "exact_epoch_boundary"
        elif a.resume_legacy_v11:
            # 旧 v1.1 未保存 RNG/sampler，且本次新增 telemetry 改变了源码 hash；
            # 仅允许合同其余部分相同的显式迁移，绝不伪称为逐 batch 精确续跑。
            legacy_keys = ("corpus_qc_sha256", "batch_size", "steps_per_epoch", "repeat_eval", "warmup_steps", "seed", "model_config", "source_contract")
            if not isinstance(saved_contract, dict) or any(saved_contract.get(k) != resume_contract[k] for k in legacy_keys):
                raise ValueError("legacy resume contract mismatch outside permitted telemetry/code migration")
            if a.workers != 4 or a.prefetch_factor != 4:
                raise ValueError("legacy v1.1 migration requires the historical workers=4 and prefetch_factor=4")
            resume_mode = "legacy_audited_nonexact"
        else:
            raise ValueError("resume contract mismatch; use --resume-legacy-v11 only for an audited v1.1 checkpoint")
        model.load_state_dict(saved["model"], strict=True)
        if saved_contract == resume_contract or not a.reset_optimizer_state:
            optimizer.load_state_dict(saved["optimizer"])
        elif resume_mode == "legacy_audited_nonexact":
            resume_mode = "legacy_branch_reset_optimizer"
        start_epoch, global_step = int(saved["epoch_completed"]) + 1, int(saved["global_step"])
        best_valid, unimproved, metrics = float(saved["best_valid"]), int(saved["unimproved"]), list(saved.get("metrics", []))
        resume_source = str(a.resume.resolve())
        swanlab_run_id = saved.get("swanlab_run_id")
        if resume_mode == "exact_epoch_boundary":
            restore_rng_state(saved["rng_state"])
    config.update({"resume_mode": resume_mode, "resume_source": resume_source, "telemetry_schema_version": 1})
    write_json(a.out.with_name("run_config.json"), config)
    if start_epoch >= a.epochs:
        raise ValueError(f"resume starts at epoch {start_epoch}, but --epochs={a.epochs}; target must be greater")
    run = None
    if a.swanlab:
        import swanlab
        if os.getenv("SWANLAB_API_KEY"):
            swanlab.login(api_key=os.environ["SWANLAB_API_KEY"])
        init_args = {"project": a.swanlab_project, "name": a.swanlab_name, "config": config, "log_dir": os.getenv("SWANLAB_DIR")}
        if swanlab_run_id:
            init_args.update({"id": swanlab_run_id, "resume": "must"})
        run = swanlab.init(**init_args); swanlab_run_id = run.id
    checkpoint_dir = a.out.parent / "checkpoints"; checkpoint_dir.mkdir(parents=True, exist_ok=True)
    overall_start = time.perf_counter()
    for epoch in range(start_epoch, a.epochs):
        if epoch >= a.train_epochs_min and unimproved > a.patience:
            break
        model.train(); train_ds.set_epoch(epoch); train_iter = iter(train_loader)
        train_sum = l2_sum = grad_pre_sum = grad_post_sum = 0.0
        clipped_batches = logit_nonfinite = 0; logit_abs_max = logit_abs_p99_last = 0.0
        input_positions = exon_positions = repeat_positions = 0
        species_counts = torch.zeros(165, dtype=torch.int64); shard_counts: dict[int, int] = {}
        epoch_start = time.perf_counter()
        for batch_index in range(a.steps_per_epoch):
            try: raw = next(train_iter)
            except StopIteration: train_ds.set_epoch(epoch + 1); train_iter = iter(train_loader); raw = next(train_iter)
            batch_species = raw["species"].reshape(-1).long().cpu()
            species_counts += torch.bincount(batch_species, minlength=165)
            for shard_id in raw["_shard_id"].reshape(-1).tolist():
                shard_counts[int(shard_id)] = shard_counts.get(int(shard_id), 0) + 1
            input_positions += int(raw["mask"].numel())
            exon_positions += int(raw["mask"].sum())
            repeat_positions += int(raw["repeat_mask"].sum())
            bases, species, exon, repeat = one_batch(raw, device)
            x, target, indices, weight = make_mlm_batch(bases, species, exon, repeat, training=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x); weighted, _ = frozen_weighted_mlm_loss(logits, target, indices, weight)
                l2_loss = shorkie_l2_penalty(model); total = weighted + l2_loss
            finite_logits = torch.isfinite(logits.detach())
            logit_nonfinite += int((~finite_logits).sum())
            if finite_logits.any():
                logit_abs_max = max(logit_abs_max, float(logits.detach()[finite_logits].abs().max()))
                if batch_index == a.steps_per_epoch - 1:
                    logit_abs_p99_last = float(torch.quantile(logits.detach().abs().float(), 0.99))
            total.backward(); grad_pre = float(torch.nn.utils.clip_grad_norm_(model.parameters(), .1))
            grad_post = min(grad_pre, .1); clipped_batches += int(grad_pre > .1)
            schedule_step = global_step
            if a.branch_start_step >= 0:
                schedule_step = a.warmup_steps + max(global_step - a.branch_start_step, 0)
            lr = set_shorkie_learning_rate(
                optimizer, schedule_step, base_lr=a.base_lr,
                warmup_steps=a.warmup_steps, schedule=a.lr_schedule,
                final_lr=a.final_lr, decay_steps=a.decay_steps)
            optimizer.step()
            train_sum += float(weighted.detach()); l2_sum += float(l2_loss.detach())
            grad_pre_sum += grad_pre; grad_post_sum += grad_post; global_step += 1
        model.eval(); valid_ds.set_epoch(epoch); valid_sum = valid_nll_sum = 0.0; valid_batches = 0
        with torch.no_grad():
            for raw in valid_loader:
                bases, species, exon, repeat = one_batch(raw, device)
                repeated_loss = repeated_nll = 0.0
                for _ in range(a.repeat_eval):
                    x, target, indices, weight = make_mlm_batch(bases, species, exon, repeat, training=False)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        weighted, nll = frozen_weighted_mlm_loss(model(x), target, indices, weight)
                    repeated_loss += float(weighted); repeated_nll += float(nll)
                valid_sum += repeated_loss / a.repeat_eval; valid_nll_sum += repeated_nll / a.repeat_eval; valid_batches += 1
        valid_loss, valid_nll = valid_sum / valid_batches, valid_nll_sum / valid_batches
        improved = valid_loss < best_valid
        if improved: best_valid, unimproved = valid_loss, 0
        else: unimproved += 1
        checkpoint_saved = (epoch + 1) % a.checkpoint_every_epochs == 0 or improved
        compact_telemetry = {
            "telemetry_schema_version": 1, "resume_exact_epoch_boundary": int(resume_mode == "exact_epoch_boundary"),
            "resume_legacy_nonexact": int(resume_mode == "legacy_audited_nonexact"),
            "train_l2_loss": l2_sum / a.steps_per_epoch, "grad_norm_pre_clip_mean": grad_pre_sum / a.steps_per_epoch,
            "grad_norm_post_clip_mean": grad_post_sum / a.steps_per_epoch, "grad_clip_fraction": clipped_batches / a.steps_per_epoch,
            "logit_nonfinite_count": logit_nonfinite, "logit_abs_max": logit_abs_max,
            "logit_abs_p99_last_batch": logit_abs_p99_last, "train_exon_fraction": exon_positions / input_positions,
            "train_repeat_fraction": repeat_positions / input_positions, "checkpoint_saved": int(checkpoint_saved),
            "checkpoint_best_saved": int(improved), **distribution_summary(species_counts, "train_species"),
            **distribution_summary(torch.tensor(list(shard_counts.values()), dtype=torch.int64), "train_shard"),
            **model_state_summary(model, optimizer),
        }
        item = {"epoch": epoch, "global_step": global_step, "train_loss_weighted": train_sum / a.steps_per_epoch, "valid_loss_weighted": valid_loss, "valid_nll_unweighted": valid_nll, "lr": lr, "improved": improved, "unimproved": unimproved, "epoch_seconds": time.perf_counter() - epoch_start, "elapsed_seconds": time.perf_counter() - overall_start, "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20, **compact_telemetry}
        metrics.append(item); write_json(a.out.with_name("metrics_live.json"), metrics)
        append_jsonl(a.out.with_name("telemetry.jsonl"), {"epoch": epoch, "global_step": global_step, "resume_mode": resume_mode, "train_species_counts": species_counts.tolist(), "train_shard_counts": {str(k): v for k, v in sorted(shard_counts.items())}})
        state = {"checkpoint_format": "shorkie165-epoch-resume-v2", "epoch_completed": epoch, "global_step": global_step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_valid": best_valid, "unimproved": unimproved, "metrics": metrics, "config": config, "resume_contract": resume_contract, "rng_state": capture_rng_state(), "sampler_state": {"next_train_epoch": epoch + 1, "train_seed": a.seed, "valid_seed": a.seed + 17, "workers": a.workers, "prefetch_factor": a.prefetch_factor, "checkpoint_boundary": "completed_epoch"}, "swanlab_run_id": swanlab_run_id}
        if checkpoint_saved:
            atomic_torch_save(state, checkpoint_dir / "checkpoint_last.pt")
            if improved: atomic_torch_save(state, checkpoint_dir / "checkpoint_best.pt")
        if run:
            swan_item = {key: value for key, value in item.items() if isinstance(value, (int, float, bool))}
            swan_item.update({f"train_species_count/{i}": int(v) for i, v in enumerate(species_counts.tolist()) if v})
            swan_item.update({f"train_shard_count/{i}": int(v) for i, v in shard_counts.items()})
            run.log(swan_item, step=global_step)
    if run: run.finish()
    result = {"status": "PASS", "config": config, "final": metrics[-1], "best_valid": best_valid, "checkpoint_last": str((checkpoint_dir / "checkpoint_last.pt").resolve()), "checkpoint_best": str((checkpoint_dir / "checkpoint_best.pt").resolve()), "swanlab_run_id": swanlab_run_id}
    write_json(a.out, result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
