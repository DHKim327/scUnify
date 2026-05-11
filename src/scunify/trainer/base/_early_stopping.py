"""Early stopping for LoRA training.

Monitors a validation metric (default: val_loss, direction=min) and stops
training when no improvement is observed for ``patience`` consecutive
epochs. Direction can be flipped to ``max`` for higher-is-better metrics
(e.g. val_pearson).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping based on a validation metric.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping (default 5).
    min_delta : float
        Minimum change to qualify as an improvement (default 0.0).
    direction : str
        ``"min"`` (default) — improvement = decrease (val_loss, val_mse_de).
        ``"max"`` — improvement = increase (val_pearson, val_acc).
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, direction: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        if direction not in ("min", "max"):
            raise ValueError(f"direction must be 'min' or 'max', got {direction!r}")
        self.direction = direction
        self.best_value: float = float("inf") if direction == "min" else float("-inf")
        self.counter: int = 0
        self.best_epoch: int = 0

    def step(self, value: float, epoch: int) -> bool:
        """Check whether to stop.

        Returns ``True`` when training should stop.
        """
        if self.direction == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
            return False  # continue training

        self.counter += 1
        if self.counter >= self.patience:
            logger.info(
                "Early stopping triggered at epoch %d "
                "(best=%.6f at epoch %d, patience=%d, direction=%s)",
                epoch,
                self.best_value,
                self.best_epoch,
                self.patience,
                self.direction,
            )
            return True  # stop

        return False  # continue

    # Back-compat alias for old call sites that referenced ``best_loss``
    @property
    def best_loss(self) -> float:
        return self.best_value
