"""CPU/GPU training loop for the CTC recognizer."""

from __future__ import annotations

import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from wormocr.checkpoints import load_checkpoint, public_config, save_checkpoint
from wormocr.config import dump_config
from wormocr.data.dataset import LabelDataset, OnlineSynthDataset, collate_lines
from wormocr.evaluation.metrics import MetricTotals
from wormocr.exceptions import ConfigError, DataError, WormOCRError
from wormocr.inference.decode import greedy_decode
from wormocr.models.crnn import build_model, count_parameters
from wormocr.runtime import resolve_device
from wormocr.vocabulary import Vocabulary

logger = logging.getLogger("wormocr.train")


@dataclass
class TrainResult:
    last_checkpoint: str
    best_checkpoint: str
    steps: int
    best_cer: float | None
    maturity: str
    final_loss: float | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate(step: int, total_steps: int, warmup: int, base_lr: float, min_ratio: float) -> float:
    if total_steps <= 0:
        return base_lr
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    progress = (step - warmup) / float(max(1, total_steps - warmup))
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def maturity_for(mode: str, steps: int, previous: str | None = None) -> str:
    if mode == "smoke":
        return "smoke-tested"
    if mode == "debug":
        return "debug-trained"
    if mode == "baseline":
        return "trained" if steps >= 500 else "debug-trained"
    if steps >= 200:
        return "trained"
    if steps >= 20:
        return "debug-trained"
    return previous or "smoke-tested"


def build_vocabulary(config: dict) -> Vocabulary:
    return Vocabulary(config.get("vocab", {}).get("charset"))


def build_datasets(config: dict) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset | None, Vocabulary]:
    vocabulary = build_vocabulary(config)
    data = config["data"]
    if data.get("labels"):
        root = data.get("image_root") or str(Path(data["labels"]).parent)
        train: torch.utils.data.Dataset = LabelDataset(data["labels"], root, config, vocabulary)
        val: torch.utils.data.Dataset | None = None
        if data.get("val_labels"):
            val_root = data.get("val_image_root") or str(Path(data["val_labels"]).parent)
            val = LabelDataset(data["val_labels"], val_root, config, vocabulary)
        return train, val, vocabulary
    train = OnlineSynthDataset(
        config,
        int(data["train_samples"]),
        int(config["seed"]),
        bool(data.get("augment", True)),
    )
    val_samples = int(data.get("val_samples") or 0)
    if val_samples <= 0:
        return train, None, vocabulary
    val_seed = data.get("val_seed")
    if val_seed is None:
        val_seed = int(config["seed"]) + 1_000_003
    val = OnlineSynthDataset(config, val_samples, int(val_seed), augment=False)
    return train, val, vocabulary


def make_loader(dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        collate_fn=collate_lines,
        generator=generator,
    )


def compute_loss(model, images, texts, widths, vocabulary: Vocabulary):
    log_probs = model(images)
    time_steps = int(log_probs.shape[1])
    input_lengths = model.output_lengths(widths).clamp(min=1, max=time_steps)
    encoded = [vocabulary.encode(text) for text in texts]
    target_lengths_list = [len(item) for item in encoded]
    for text, target, available in zip(texts, target_lengths_list, input_lengths.tolist()):
        if target == 0:
            raise DataError("refusing an empty training label")
        if target > available:
            raise DataError(
                f"label {text!r} has {target} characters but the line only has {available} "
                "CTC time steps. Raise data.max_width or shorten data.max_chars."
            )
    targets = torch.tensor([index for item in encoded for index in item], dtype=torch.long)
    target_lengths = torch.tensor(target_lengths_list, dtype=torch.long)
    loss = nn.functional.ctc_loss(
        log_probs.permute(1, 0, 2).float(),
        targets,
        input_lengths.cpu(),
        target_lengths.cpu(),
        blank=vocabulary.blank,
        zero_infinity=True,
        reduction="mean",
    )
    return loss, log_probs, input_lengths


@torch.no_grad()
def evaluate_model(
    model,
    loader: DataLoader,
    vocabulary: Vocabulary,
    device: torch.device,
    collect: bool = False,
) -> tuple[MetricTotals, list[tuple[str, str]]]:
    was_training = model.training
    model.eval()
    totals = MetricTotals()
    previews: list[tuple[str, str]] = []
    for batch in loader:
        images = batch["images"].to(device)
        loss, log_probs, lengths = compute_loss(model, images, batch["texts"], batch["widths"], vocabulary)
        totals.update_loss(float(loss.detach().cpu()))
        for text, indices in zip(batch["texts"], greedy_decode(log_probs, lengths, vocabulary.blank)):
            hypothesis = vocabulary.decode(indices)
            totals.update(text, hypothesis)
            if collect or len(previews) < 8:
                previews.append((text, hypothesis))
    if was_training:
        model.train()
    return totals, previews


def _optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay: list[nn.Parameter] = []
    quiet: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias"):
            quiet.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": quiet, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def _batchnorm_eval_if_tiny(model: nn.Module, batch_size: int) -> list[nn.BatchNorm2d]:
    if batch_size >= 2:
        return []
    paused: list[nn.BatchNorm2d] = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and module.training:
            paused.append(module)
            module.eval()
    return paused


def _checkpoint_payload(
    model,
    optimizer,
    config,
    vocabulary: Vocabulary,
    epoch: int,
    step: int,
    maturity: str,
    best_cer: float | None,
    history: list[dict],
    total_steps: int,
) -> dict:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": public_config(config),
        "charset": vocabulary.charset,
        "epoch": epoch,
        "step": step,
        "maturity": maturity,
        "best_cer": best_cer,
        "history": history,
        "total_steps": total_steps,
    }


def train(config: dict, mode: str = "baseline", resume_from: str | None = None) -> TrainResult:
    _ensure_logging()
    if mode == "resume" and not resume_from:
        raise ConfigError("resume mode needs --checkpoint pointing at last.pt or best.pt")
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device"))
    dataset, val_dataset, vocabulary = build_datasets(config)
    model = build_model(config, vocabulary.size).to(device)
    train_cfg = config["train"]
    optimizer = _optimizer(model, float(train_cfg["lr"]), float(train_cfg["weight_decay"]))
    start_epoch = 0
    step = 0
    best_cer: float | None = None
    history: list[dict] = []
    previous_maturity = None
    if resume_from:
        payload = load_checkpoint(resume_from, map_location=device)
        if payload["charset"] != vocabulary.charset:
            raise WormOCRError("resume checkpoint charset does not match this config")
        model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload.get("epoch", 0))
        step = int(payload.get("step", 0))
        best_cer = payload.get("best_cer")
        history = list(payload.get("history") or [])
        previous_maturity = payload.get("maturity")
        logger.info("resumed from %s at step %s", resume_from, step)

    workers = int(train_cfg.get("num_workers") or 0)
    loader = make_loader(dataset, int(train_cfg["batch_size"]), True, int(config["seed"]), workers)
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = make_loader(
            val_dataset,
            int(train_cfg["batch_size"]),
            False,
            int(config["seed"]) + 17,
            workers,
        )
    epochs = int(train_cfg["epochs"])
    planned = max(1, epochs * max(1, len(loader)))
    if train_cfg.get("max_steps"):
        planned = min(planned, int(train_cfg["max_steps"]))
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.yaml").write_text(dump_config(config), encoding="utf-8")
    logger.info(
        "training %s on %s | parameters=%s | samples=%s | planned_steps=%s | maturity target=%s",
        config["model"].get("name", "recognizer"),
        device,
        count_parameters(model),
        len(dataset),
        planned,
        mode,
    )
    log_path = checkpoint_dir / "train_log.jsonl"
    log_file = log_path.open("a", encoding="utf-8")
    use_amp = bool(train_cfg.get("amp")) and device.type == "cuda"
    if train_cfg.get("amp") and not use_amp:
        logger.info("mixed precision was requested but CUDA is unavailable; training in float32")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    final_loss = None
    stale_epochs = 0
    patience = int(train_cfg.get("early_stopping_patience") or 0)
    max_steps = int(train_cfg["max_steps"]) if train_cfg.get("max_steps") else None
    max_minutes = train_cfg.get("max_minutes")
    started = time.perf_counter()
    model.train()
    completed_epoch = start_epoch
    try:
        for epoch in range(start_epoch, start_epoch + epochs):
            stopped_early = False
            for batch in loader:
                if max_steps is not None and step >= max_steps:
                    stopped_early = True
                    break
                if max_minutes is not None and (time.perf_counter() - started) / 60.0 >= float(max_minutes):
                    logger.info("stopping because train.max_minutes=%s was reached", max_minutes)
                    max_steps = step
                    stopped_early = True
                    break
                lr = learning_rate(
                    step,
                    planned,
                    int(train_cfg.get("warmup_steps") or 0),
                    float(train_cfg["lr"]),
                    float(train_cfg.get("min_lr_ratio") or 0.1),
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                images = batch["images"].to(device)
                paused = _batchnorm_eval_if_tiny(model, images.shape[0])
                optimizer.zero_grad(set_to_none=True)
                try:
                    if use_amp and scaler is not None:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            loss, _log_probs, _lengths = compute_loss(
                                model, images, batch["texts"], batch["widths"], vocabulary
                            )
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss, _log_probs, _lengths = compute_loss(
                            model, images, batch["texts"], batch["widths"], vocabulary
                        )
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                        optimizer.step()
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        raise MemoryError(
                            "wormOCR ran out of memory. Lower train.batch_size or data.max_chars."
                        ) from exc
                    raise
                finally:
                    for module in paused:
                        module.train()
                if not torch.isfinite(loss):
                    raise WormOCRError("loss became non-finite. Lower train.lr and try again.")
                final_loss = float(loss.detach().cpu())
                if step % int(train_cfg.get("log_every") or 20) == 0:
                    record = {"step": step, "epoch": epoch, "loss": final_loss, "lr": lr}
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()
                    logger.info("step %s epoch %s loss %.4f lr %.6f", step, epoch, final_loss, lr)
                step += 1
            if stopped_early:
                completed_epoch = epoch
                best_cer = _validate_and_save(
                    model, optimizer, val_loader, vocabulary, device, config, epoch, step,
                    best_cer, history, planned, previous_maturity, mode, checkpoint_dir,
                )
                break
            completed_epoch = epoch + 1
            metrics = _validate_and_save(
                model, optimizer, val_loader, vocabulary, device, config, epoch, step,
                best_cer, history, planned, previous_maturity, mode, checkpoint_dir,
            )
            if val_loader is not None and metrics is not None and (best_cer is None or metrics < best_cer - 1e-6):
                stale_epochs = 0
            elif val_loader is not None:
                stale_epochs += 1
            best_cer = metrics if metrics is not None else best_cer
            save_every = int(train_cfg.get("save_every") or 0)
            if save_every > 0 and (epoch + 1) % save_every == 0:
                maturity = maturity_for(mode, step, previous_maturity)
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
                    _checkpoint_payload(
                        model, optimizer, config, vocabulary, epoch + 1, step, maturity, best_cer, history, planned
                    ),
                )
            if patience > 0 and stale_epochs >= patience:
                logger.info("early stopping after %s epochs without CER improvement", stale_epochs)
                break
    finally:
        log_file.close()
    maturity = maturity_for(mode, step, previous_maturity)
    last_path = checkpoint_dir / "last.pt"
    save_checkpoint(
        last_path,
        _checkpoint_payload(
                model, optimizer, config, vocabulary, completed_epoch, step, maturity, best_cer, history, planned
            ),
        )
    best_path = checkpoint_dir / "best.pt"
    if not best_path.exists():
        save_checkpoint(
            best_path,
            _checkpoint_payload(
                model, optimizer, config, vocabulary, completed_epoch, step, maturity, best_cer, history, planned
            ),
        )
    _write_metrics(checkpoint_dir, best_cer, step, maturity, final_loss)
    logger.info("saved %s (%s)", best_path, maturity)
    return TrainResult(str(last_path), str(best_path), step, best_cer, maturity, final_loss)


def _validate_and_save(
    model,
    optimizer,
    val_loader,
    vocabulary,
    device,
    config,
    epoch,
    step,
    best_cer,
    history,
    planned,
    previous_maturity,
    mode,
    checkpoint_dir: Path,
) -> float | None:
    maturity = maturity_for(mode, step, previous_maturity)
    if val_loader is None:
        save_checkpoint(
            checkpoint_dir / "last.pt",
            _checkpoint_payload(
                model, optimizer, config, vocabulary, epoch, step, maturity, best_cer, history, planned
            ),
        )
        return best_cer
    totals, previews = evaluate_model(model, val_loader, vocabulary, device)
    metrics = totals.to_dict()
    metrics["step"] = step
    metrics["epoch"] = epoch
    history.append(metrics)
    logger.info(
        "val step %s CER %.4f WER %.4f exact %.3f loss %s",
        step,
        totals.corpus_cer,
        totals.corpus_wer,
        totals.exact_match,
        f"{totals.mean_loss:.4f}" if totals.mean_loss is not None else "n/a",
    )
    for reference, hypothesis in previews[:5]:
        logger.info("  ref=%r hyp=%r", reference, hypothesis)
    improved = best_cer is None or totals.corpus_cer < best_cer
    current_best = totals.corpus_cer if improved else best_cer
    payload = _checkpoint_payload(
        model, optimizer, config, vocabulary, epoch, step, maturity, current_best, history, planned
    )
    save_checkpoint(checkpoint_dir / "last.pt", payload)
    if improved:
        save_checkpoint(checkpoint_dir / "best.pt", payload)
        weights = dict(payload)
        weights.pop("optimizer_state", None)
        save_checkpoint(checkpoint_dir / "best_weights.pt", weights)
        logger.info("new best CER %.4f", totals.corpus_cer)
    return current_best


def _write_metrics(checkpoint_dir: Path, best_cer: float | None, step: int, maturity: str, final_loss: float | None) -> None:
    summary = {
        "best_cer": best_cer,
        "step": step,
        "maturity": maturity,
        "final_loss": final_loss,
    }
    (checkpoint_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _ensure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
