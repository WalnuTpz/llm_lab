from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from llm_lab.config import load_config
from llm_lab.data import ByteLevelBPETokenizer, get_batch, get_batch_with_future_targets
from llm_lab.models import build_model
from llm_lab.training import cosine_lr, cross_entropy
from llm_lab.utils import dtype_from_str, parameter_report, resolve_device, safe_dtype_for_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or smoke-test an LLM lab model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run on random tokens instead of dataset files.")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from a checkpoint path, or from checkpoint_dir/latest.pt when passed without a value.",
    )
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint writes.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.data.tokenizer_path is not None and not args.smoke:
        tokenizer = ByteLevelBPETokenizer.load(cfg.data.tokenizer_path)
        if tokenizer.vocab_size > cfg.model.vocab_size:
            raise ValueError(
                f"tokenizer vocab_size {tokenizer.vocab_size} exceeds model vocab_size {cfg.model.vocab_size}"
            )
    device = resolve_device(args.device or cfg.training.device)
    dtype = safe_dtype_for_device(dtype_from_str(args.dtype or cfg.training.dtype), device)
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    model = build_model(cfg.model, device=device, dtype=dtype)
    mtp_layers = _enabled_mtp_layers(
        model,
        cfg.model.mtp_layers,
        cfg.model.mtp_loss_weight,
    )
    active = model.active_parameters_per_token() if hasattr(model, "active_parameters_per_token") else None
    params = parameter_report(model, active_parameters_per_token=active)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=(cfg.training.beta1, cfg.training.beta2),
        eps=cfg.training.adam_eps,
        weight_decay=cfg.training.weight_decay,
    )

    max_iters = args.max_iters or (1 if args.smoke else cfg.training.max_iters)
    warmup_iters, cosine_cycle_iters = _effective_lr_schedule(
        config_max_iters=cfg.training.max_iters,
        max_iters=max_iters,
        config_warmup_iters=cfg.training.warmup_iters,
        config_cosine_cycle_iters=cfg.training.cosine_cycle_iters,
    )
    train_dataset, val_dataset = _load_datasets(cfg, smoke=args.smoke)
    checkpoint_dir = Path(cfg.runtime.checkpoint_dir)
    run_dir = Path(cfg.runtime.run_dir)
    metrics_path = run_dir / "metrics.jsonl"

    start_step = 0
    if args.resume is not None:
        resume_path = checkpoint_dir / "latest.pt" if args.resume == "latest" else Path(args.resume)
        start_step = _load_checkpoint(resume_path, model, optimizer, device)
        _write_metric(metrics_path, {"type": "resume", "step": start_step, "checkpoint": str(resume_path)})
    else:
        _prepare_fresh_metrics(metrics_path)
        _write_metric(
            metrics_path,
            {
                "type": "run",
                "name": cfg.name,
                "architecture": cfg.model.architecture,
                "git_commit": _git_commit(),
                "tokenizer_path": cfg.data.tokenizer_path,
                "train_data": cfg.data.train_data,
                "val_data": cfg.data.val_data,
                "total_parameters": params.total_parameters,
                "active_parameters_per_token": params.active_parameters_per_token,
                "batch_size": cfg.training.batch_size,
                "context_length": cfg.model.context_length,
                "grad_accum_steps": cfg.training.grad_accum_steps,
                "max_iters": max_iters,
                "warmup_iters": warmup_iters,
                "cosine_cycle_iters": cosine_cycle_iters,
                "mtp_layers": mtp_layers,
                "mtp_loss_weight": cfg.model.mtp_loss_weight,
            },
        )
    if start_step > max_iters:
        raise ValueError(f"resume checkpoint step {start_step} is greater than max_iters {max_iters}")

    model.train()
    start = time.perf_counter()
    for step in range(start_step, max_iters):
        lr = cosine_lr(
            step,
            cfg.training.learning_rate,
            cfg.training.min_learning_rate,
            warmup_iters,
            cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        accum_losses = []
        accum_next_token_losses = []
        accum_mtp_losses = []
        for _ in range(cfg.training.grad_accum_steps):
            x, targets = get_batch_with_future_targets(
                train_dataset,
                cfg.training.batch_size,
                cfg.model.context_length,
                device,
                num_targets=1 + mtp_layers,
            )
            loss, next_token_loss, mtp_loss = _training_loss(
                model,
                x,
                targets,
                mtp_layers=mtp_layers,
                mtp_loss_weight=cfg.model.mtp_loss_weight,
            )
            (loss / cfg.training.grad_accum_steps).backward()
            accum_losses.append(float(loss.detach().cpu()))
            accum_next_token_losses.append(float(next_token_loss.detach().cpu()))
            if mtp_loss is not None:
                accum_mtp_losses.append(float(mtp_loss.detach().cpu()))
        if cfg.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()
        completed_step = step + 1
        elapsed = time.perf_counter() - start
        train_loss = float(np.mean(accum_losses))
        next_token_loss = float(np.mean(accum_next_token_losses))
        mtp_loss = float(np.mean(accum_mtp_losses)) if accum_mtp_losses else None
        tokens_per_step = cfg.training.batch_size * cfg.model.context_length * cfg.training.grad_accum_steps
        tokens_seen = (completed_step - start_step) * tokens_per_step
        train_record = {
            "type": "train",
            "step": completed_step,
            "lr": lr,
            "loss": train_loss,
            "next_token_loss": next_token_loss,
            "tokens": completed_step * tokens_per_step,
            "grad_accum_steps": cfg.training.grad_accum_steps,
            "tokens_per_second": tokens_seen / max(elapsed, 1e-9),
            "elapsed_seconds": elapsed,
        }
        if mtp_loss is not None:
            train_record["mtp_loss"] = mtp_loss
        _write_metric(metrics_path, train_record)
        if completed_step % cfg.runtime.log_interval == 0 or completed_step == max_iters:
            if mtp_loss is None:
                print(f"step={completed_step} lr={lr:.6g} loss={train_loss:.6f}")
            else:
                print(
                    f"step={completed_step} lr={lr:.6g} loss={train_loss:.6f} "
                    f"next_token_loss={next_token_loss:.6f} mtp_loss={mtp_loss:.6f}"
                )

        if val_dataset is not None and (
            completed_step % cfg.runtime.eval_interval == 0 or completed_step == max_iters
        ):
            val_loss = evaluate(
                model,
                val_dataset,
                batch_size=cfg.training.batch_size,
                context_length=cfg.model.context_length,
                device=device,
                eval_batches=cfg.runtime.eval_batches,
            )
            _write_metric(
                metrics_path,
                {
                    "type": "eval",
                    "step": completed_step,
                    "val_loss": val_loss,
                },
            )
            print(f"step={completed_step} val_loss={val_loss:.6f}")

        if not args.no_checkpoint and (
            completed_step % cfg.runtime.checkpoint_interval == 0 or completed_step == max_iters
        ):
            save_checkpoint(
                checkpoint_dir,
                model=model,
                optimizer=optimizer,
                cfg_dict=cfg.to_dict(),
                step=completed_step,
            )

    elapsed = time.perf_counter() - start
    tokens = (
        (max_iters - start_step)
        * cfg.training.batch_size
        * cfg.model.context_length
        * cfg.training.grad_accum_steps
    )
    print(f"tokens_per_second={tokens / max(elapsed, 1e-9):.2f}")


def _enabled_mtp_layers(model: torch.nn.Module, mtp_layers: int, mtp_loss_weight: float) -> int:
    if mtp_layers <= 0 or mtp_loss_weight <= 0.0:
        return 0
    if not hasattr(model, "forward_with_mtp"):
        raise ValueError("mtp_loss_weight > 0 requires a model that implements forward_with_mtp")
    return mtp_layers


def _training_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    targets: list[torch.Tensor],
    *,
    mtp_layers: int,
    mtp_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if mtp_layers == 0:
        logits = model(x)
        next_token_loss = cross_entropy(logits, targets[0])
        return next_token_loss, next_token_loss, None

    logits, mtp_logits = model.forward_with_mtp(x)
    if len(mtp_logits) < mtp_layers:
        raise RuntimeError(f"model returned {len(mtp_logits)} MTP logits, expected {mtp_layers}")
    next_token_loss = cross_entropy(logits, targets[0])
    mtp_losses = [
        cross_entropy(logits_k, target_k)
        for logits_k, target_k in zip(mtp_logits[:mtp_layers], targets[1:], strict=True)
    ]
    mtp_loss = torch.stack(mtp_losses).mean()
    return next_token_loss + mtp_loss_weight * mtp_loss, next_token_loss, mtp_loss


def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    device: torch.device,
    eval_batches: int,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = get_batch(dataset, batch_size, context_length, device)
            logits = model(x)
            losses.append(float(cross_entropy(logits, y).detach().cpu()))
    model.train()
    return float(np.mean(losses))


def save_checkpoint(
    checkpoint_dir: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg_dict: dict[str, object],
    step: int,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg_dict,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_random_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    step_path = checkpoint_dir / f"step_{step:08d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(payload, step_path)
    torch.save(payload, latest_path)
    return step_path


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if device.type == "cuda" and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return int(checkpoint["step"])


def _load_datasets(cfg, *, smoke: bool) -> tuple[np.ndarray, np.ndarray | None]:
    if smoke:
        train = np.random.randint(0, cfg.model.vocab_size, size=cfg.model.context_length * 8 + 1, dtype=np.int64)
        val = np.random.randint(0, cfg.model.vocab_size, size=cfg.model.context_length * 8 + 1, dtype=np.int64)
        return train, val
    if cfg.data.train_data is None:
        raise ValueError("train_data must be set unless --smoke is used")
    train = np.load(cfg.data.train_data, mmap_mode="r")
    val = np.load(cfg.data.val_data, mmap_mode="r") if cfg.data.val_data is not None else None
    return train, val


def _effective_lr_schedule(
    *,
    config_max_iters: int,
    max_iters: int,
    config_warmup_iters: int,
    config_cosine_cycle_iters: int,
) -> tuple[int, int]:
    if config_cosine_cycle_iters != config_max_iters or max_iters == config_max_iters:
        return min(config_warmup_iters, max_iters), max(config_cosine_cycle_iters, config_warmup_iters)

    scale = max_iters / config_max_iters
    warmup_iters = int(round(config_warmup_iters * scale))
    if config_warmup_iters > 0:
        warmup_iters = max(1, warmup_iters)
    warmup_iters = min(warmup_iters, max_iters)
    return warmup_iters, max_iters


def _prepare_fresh_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _write_metric(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    main()
