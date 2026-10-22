"""CPU/GPU training loop for the CTC recognizer."""

from __future__ import annotations

import copy
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
from wormocr.vocabulary import Vocabulary, ctc_time_needed

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


def schedule_span(start_step: int, epochs: int, batches_per_epoch: int, max_steps: int | None) -> int:
    """Steps this invocation should spread warmup and cosine across.

    A resumed run schedules from the steps it will still take. Using the global
    step against a short horizon drops the learning rate before characters align.
    """
    local = max(1, int(epochs) * max(1, int(batches_per_epoch)))
    if max_steps is not None:
        remaining = int(max_steps) - int(start_step)
        if remaining > 0:
            local = max(1, min(local, remaining))
    return local


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
    if mode in {"baseline", "resume"}:
        return "trained" if steps >= 500 else "debug-trained"
    if steps >= 20:
        return "debug-trained"
    return previous or "smoke-tested"


def build_vocabulary(config: dict) -> Vocabulary:
    return Vocabulary(config.get("vocab", {}).get("charset"))


def build_datasets(config: dict) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset | None, Vocabulary]:
    vocabulary = build_vocabulary(config)
    data = config["data"]
    val: torch.utils.data.Dataset | None = None
    if data.get("val_labels"):
        val_root = data.get("val_image_root") or str(Path(data["val_labels"]).parent)
        val = LabelDataset(data["val_labels"], val_root, config, vocabulary)
    if data.get("labels"):
        root = data.get("image_root") or str(Path(data["labels"]).parent)
        train: torch.utils.data.Dataset = LabelDataset(data["labels"], root, config, vocabulary)
        return train, val, vocabulary
    train = OnlineSynthDataset(
        config,
        int(data["train_samples"]),
        int(config["seed"]),
        bool(data.get("augment", True)),
    )
    if val is not None:
        return train, val, vocabulary
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
    workers = int(num_workers)
    extra: dict = {}
    if workers > 0:
        extra = {"persistent_workers": True, "prefetch_factor": 4}
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_lines,
        generator=generator,
        **extra,
    )


def compute_loss(model, images, texts, widths, vocabulary: Vocabulary):
    log_probs = model(images)
    time_steps = int(log_probs.shape[1])
    input_lengths = model.output_lengths(widths).clamp(min=1, max=time_steps)
    encoded = [vocabulary.encode(text) for text in texts]
    target_lengths_list = [len(item) for item in encoded]
    for text, target, available in zip(texts, target_lengths_list, input_lengths.tolist()):
        needed = ctc_time_needed(text)
        if target == 0:
            raise DataError("refusing an empty training label")
        if needed > available:
            raise DataError(
                f"label {text!r} needs {needed} CTC time steps but the line only has {available}. "
                "Raise data.max_width or shorten data.max_chars."
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
        log_probs = model(images)
        lengths = model.output_lengths(batch["widths"]).clamp(min=1, max=int(log_probs.shape[1]))
        feasible = [
            index
            for index, (text, available) in enumerate(zip(batch["texts"], lengths.tolist()))
            if text and ctc_time_needed(text) <= available
        ]
        if feasible:
            texts = [batch["texts"][index] for index in feasible]
            targets = torch.tensor([value for text in texts for value in vocabulary.encode(text)], dtype=torch.long)
            loss = nn.functional.ctc_loss(
                log_probs[feasible].permute(1, 0, 2).float(),
                targets,
                lengths[feasible].cpu(),
                torch.tensor([len(text) for text in texts], dtype=torch.long),
                blank=vocabulary.blank,
                zero_infinity=True,
                reduction="mean",
            )
            totals.update_loss(float(loss.detach().cpu()))
        for text, indices in zip(batch["texts"], greedy_decode(log_probs, lengths, vocabulary.blank)):
            hypothesis = vocabulary.decode(indices)
            totals.update(text, hypothesis)
            if collect or len(previews) < 8:
                previews.append((text, hypothesis))
    if was_training:
        model.train()
    return totals, previews


class _MemoryDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[dict]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[int(index)]


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


def _make_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(model)
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    return ema


@torch.no_grad()
def _update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for averaged, current in zip(ema.parameters(), model.parameters()):
        averaged.mul_(decay).add_(current.detach(), alpha=1.0 - decay)
    for averaged, current in zip(ema.buffers(), model.buffers()):
        averaged.copy_(current)


def _checkpoint_payload(
    export_model,
    train_model,
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
    payload = {
        "model_state": export_model.state_dict(),
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
    if train_model is not export_model:
        payload["train_state"] = train_model.state_dict()
    return payload


def train(config: dict, mode: str = "baseline", resume_from: str | None = None) -> TrainResult:
    _ensure_logging()
    if mode == "resume" and not resume_from:
        raise ConfigError("resume mode needs --checkpoint pointing at last.pt or best.pt")
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device"))
    train_cfg = config["train"]
    if train_cfg.get("torch_threads"):
        torch.set_num_threads(int(train_cfg["torch_threads"]))
    dataset, val_dataset, vocabulary = build_datasets(config)
    model = build_model(config, vocabulary.size).to(device)
    optimizer = _optimizer(model, float(train_cfg["lr"]), float(train_cfg["weight_decay"]))
    ema_decay = float(train_cfg.get("ema_decay") or 0.0)
    ema = _make_ema(model) if ema_decay > 0 else None
    start_epoch = 0
    step = 0
    best_cer: float | None = None
    history: list[dict] = []
    previous_maturity = None
    if resume_from:
        payload = load_checkpoint(resume_from, map_location=device)
        if payload["charset"] != vocabulary.charset:
            raise WormOCRError("resume checkpoint charset does not match this config")
        model.load_state_dict(payload.get("train_state") or payload["model_state"])
        if ema is not None:
            ema.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state"):
            optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload.get("epoch", 0))
        step = int(payload.get("step", 0))
        best_cer = payload.get("best_cer")
        history = list(payload.get("history") or [])
        previous_maturity = payload.get("maturity")
        logger.info("resumed from %s at step %s", resume_from, step)
    export_model = ema if ema is not None else model

    workers = int(train_cfg.get("num_workers") or 0)
    batch_size = int(train_cfg["batch_size"])
    loader = make_loader(dataset, batch_size, True, int(config["seed"]), workers)
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        cached = _MemoryDataset([val_dataset[index] for index in range(len(val_dataset))])
        val_loader = make_loader(cached, batch_size, False, int(config["seed"]) + 17, 0)
    epochs = int(train_cfg["epochs"])
    start_step = step
    max_steps = int(train_cfg["max_steps"]) if train_cfg.get("max_steps") else None
    local_steps = schedule_span(start_step, epochs, len(loader), max_steps)
    planned = start_step + local_steps
    eval_every = int(train_cfg.get("eval_every") or 0)
    patience = int(train_cfg.get("early_stopping_patience") or 0)
    max_minutes = train_cfg.get("max_minutes")
    log_every = int(train_cfg.get("log_every") or 20)
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.yaml").write_text(dump_config(config), encoding="utf-8")
    logger.info(
        "training %s on %s | parameters=%s | samples=%s | planned_steps=%s | workers=%s | ema=%s | mode=%s",
        config["model"].get("name", "recognizer"),
        device,
        count_parameters(model),
        len(dataset),
        local_steps,
        workers,
        ema_decay or "off",
        mode,
    )
    log_path = checkpoint_dir / "train_log.jsonl"
    log_file = log_path.open("a" if resume_from else "w", encoding="utf-8")
    use_amp = bool(train_cfg.get("amp")) and device.type == "cuda"
    if train_cfg.get("amp") and not use_amp:
        logger.info("mixed precision was requested but CUDA is unavailable; training in float32")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    final_loss: float | None = None
    stale = 0
    last_validated_step = -1
    started = time.perf_counter()
    model.train()
    completed_epoch = start_epoch

    def validate(epoch: int) -> None:
        nonlocal best_cer, stale, last_validated_step
        last_validated_step = step
        before = best_cer
        best_cer = _validate_and_save(
            export_model, model, optimizer, val_loader, vocabulary, device, config, epoch, step,
            best_cer, history, planned, previous_maturity, mode, checkpoint_dir,
        )
        if val_loader is None:
            return
        if before is None or (best_cer is not None and best_cer < before - 1e-6):
            stale = 0
        else:
            stale += 1

    def patience_spent() -> bool:
        if patience > 0 and stale >= patience:
            logger.info("early stopping after %s validations without CER improvement", stale)
            return True
        return False

    try:
        stopped = False
        for epoch in range(start_epoch, start_epoch + epochs):
            for batch in loader:
                if max_steps is not None and step >= max_steps:
                    stopped = True
                    break
                if max_minutes is not None and (time.perf_counter() - started) / 60.0 >= float(max_minutes):
                    logger.info("stopping because train.max_minutes=%s was reached", max_minutes)
                    stopped = True
                    break
                lr = learning_rate(
                    step - start_step,
                    local_steps,
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
                if ema is not None:
                    _update_ema(ema, model, min(ema_decay, (1.0 + step) / (10.0 + step)))
                final_loss = float(loss.detach().cpu())
                if step % log_every == 0:
                    record = {"step": step, "epoch": epoch, "loss": final_loss, "lr": lr}
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()
                    elapsed = time.perf_counter() - started
                    logger.info(
                        "step %s epoch %s loss %.4f lr %.6f elapsed %.0fs",
                        step, epoch, final_loss, lr, elapsed,
                    )
                step += 1
                if eval_every > 0 and step % eval_every == 0:
                    validate(epoch)
                    if patience_spent():
                        stopped = True
                        break
            if stopped:
                completed_epoch = epoch
                if last_validated_step != step:
                    validate(epoch)
                break
            completed_epoch = epoch + 1
            if last_validated_step != step:
                validate(epoch + 1)
            save_every = int(train_cfg.get("save_every") or 0)
            if save_every > 0 and (epoch + 1) % save_every == 0:
                save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
                    _checkpoint_payload(
                        export_model, model, optimizer, config, vocabulary, epoch + 1, step,
                        maturity_for(mode, step, previous_maturity), best_cer, history, planned,
                    ),
                )
            if patience_spent():
                break
    finally:
        log_file.close()
    maturity = maturity_for(mode, step, previous_maturity)
    last_path = checkpoint_dir / "last.pt"
    save_checkpoint(
        last_path,
        _checkpoint_payload(
            export_model, model, optimizer, config, vocabulary, completed_epoch, step,
            maturity, best_cer, history, planned,
        ),
    )
    best_path = checkpoint_dir / "best.pt"
    if not best_path.exists():
        save_checkpoint(
            best_path,
            _checkpoint_payload(
                export_model, model, optimizer, config, vocabulary, completed_epoch, step,
                maturity, best_cer, history, planned,
            ),
        )
    _write_metrics(checkpoint_dir, best_cer, step, maturity, final_loss)
    logger.info("saved %s (%s)", best_path, maturity)
    return TrainResult(str(last_path), str(best_path), step, best_cer, maturity, final_loss)


def _validate_and_save(
    export_model,
    train_model,
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
                export_model, train_model, optimizer, config, vocabulary, epoch, step,
                maturity, best_cer, history, planned,
            ),
        )
        return best_cer
    totals, previews = evaluate_model(export_model, val_loader, vocabulary, device)
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
        export_model, train_model, optimizer, config, vocabulary, epoch, step,
        maturity, current_best, history, planned,
    )
    save_checkpoint(checkpoint_dir / "last.pt", payload)
    if improved:
        save_checkpoint(checkpoint_dir / "best.pt", payload)
        weights = dict(payload)
        weights.pop("optimizer_state", None)
        weights.pop("train_state", None)
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
