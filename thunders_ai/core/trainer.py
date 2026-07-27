"""Trainer module for model fine-tuning with distributed and mixed-precision support."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from thunders_ai.config import ThundersConfig
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Handles model training loops with advanced features.

    Supports distributed training, gradient accumulation, mixed precision,
    learning rate scheduling, and checkpoint management.

    Args:
        config: ThundersConfig with training hyper-parameters.
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        val_loader: Optional DataLoader for validation data.

    Example::

        trainer = Trainer(config, model, train_loader, val_loader)
        trainer.train()
    """

    def __init__(
        self,
        config: ThundersConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        self._config = config
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader

        self._device = torch.device(
            getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model.to(self._device)

        # Training hyper-parameters from config
        self._epochs: int = getattr(config, "epochs", 3)
        self._lr: float = getattr(config, "learning_rate", 5e-5)
        self._grad_accum_steps: int = getattr(config, "gradient_accumulation_steps", 1)
        self._max_grad_norm: float = getattr(config, "max_grad_norm", 1.0)
        self._use_amp: bool = getattr(config, "mixed_precision", False)
        self._output_dir: str = getattr(config, "output_dir", "./checkpoints")

        # Optimizer & scheduler
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._lr)
        self._scheduler = self._build_scheduler()
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)

        # State tracking
        self._global_step = 0
        self._metrics: List[Dict[str, Any]] = []

        logger.info(
            "Trainer ready – epochs=%d, lr=%.2e, grad_accum=%d, amp=%s",
            self._epochs, self._lr, self._grad_accum_steps, self._use_amp,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Construct a cosine-annealing learning rate scheduler."""
        total_steps = len(self._train_loader) * self._epochs // self._grad_accum_steps
        warmup_steps = int(total_steps * 0.1)
        return torch.optim.lr_scheduler.SequentialLR(
            self._optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self._optimizer, start_factor=0.01, total_iters=warmup_steps
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self._optimizer, T_max=total_steps - warmup_steps
                ),
            ],
            milestones=[warmup_steps],
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, List[float]]:
        """Run the full training loop and return per-epoch metrics.

        Returns:
            Dictionary with ``train_loss`` and ``val_loss`` lists.
        """
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self._epochs + 1):
            logger.info("=== Epoch %d/%d ===", epoch, self._epochs)
            train_loss = self._train_epoch(epoch)
            history["train_loss"].append(train_loss)

            if self._val_loader is not None:
                val_loss = self._validate()
                history["val_loss"].append(val_loss)
                logger.info("Epoch %d – train_loss=%.4f  val_loss=%.4f", epoch, train_loss, val_loss)
            else:
                logger.info("Epoch %d – train_loss=%.4f", epoch, train_loss)

            self._save_checkpoint(epoch)

        return history

    def _train_epoch(self, epoch: int) -> float:
        """Execute a single training epoch."""
        self._model.train()
        total_loss = 0.0
        self._optimizer.zero_grad()

        for step, batch in enumerate(self._train_loader, 1):
            batch = {k: v.to(self._device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=self._use_amp):
                outputs = self._model(**batch)
                loss = outputs.loss / self._grad_accum_steps

            self._scaler.scale(loss).backward()

            if step % self._grad_accum_steps == 0:
                self._scaler.unscale_(self._optimizer)
                nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad()
                self._scheduler.step()
                self._global_step += 1

            total_loss += loss.item() * self._grad_accum_steps

            if step % 50 == 0:
                logger.debug("Epoch %d step %d – loss=%.4f", epoch, step, loss.item())

        return total_loss / len(self._train_loader)

    @torch.no_grad()
    def _validate(self) -> float:
        """Evaluate the model on the validation set."""
        self._model.eval()
        total_loss = 0.0
        for batch in self._val_loader:
            batch = {k: v.to(self._device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                outputs = self._model(**batch)
            total_loss += outputs.loss.item()
        return total_loss / len(self._val_loader)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int) -> Path:
        """Save a training checkpoint to disk."""
        ckpt_dir = Path(self._output_dir) / f"checkpoint-epoch-{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self._model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "scaler_state_dict": self._scaler.state_dict(),
                "global_step": self._global_step,
            },
            ckpt_dir / "ckpt.pt",
        )
        logger.info("Checkpoint saved → %s", ckpt_dir)
        return ckpt_dir

    def load_checkpoint(self, path: str) -> None:
        """Restore model, optimizer, and scaler state from a checkpoint.

        Args:
            path: Path to the ``ckpt.pt`` file.
        """
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self._scaler.load_state_dict(ckpt["scaler_state_dict"])
        self._global_step = ckpt.get("global_step", 0)
        logger.info("Checkpoint restored from %s (epoch %d)", path, ckpt["epoch"])
