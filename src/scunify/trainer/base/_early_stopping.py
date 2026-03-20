"""Early stopping for LoRA training.

Monitors validation loss and stops training when no improvement
is observed for ``patience`` consecutive epochs.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping based on validation loss.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping (default 5).
    min_delta : float
        Minimum change to qualify as an improvement (default 0.0).
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.best_epoch: int = 0

    def step(self, val_loss: float, epoch: int) -> bool:
        """Check whether to stop.

        Returns ``True`` when training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            return False  # continue training

        self.counter += 1
        if self.counter >= self.patience:
            logger.info(
                "Early stopping triggered at epoch %d "
                "(best=%.6f at epoch %d, patience=%d)",
                epoch,
                self.best_loss,
                self.best_epoch,
                self.patience,
            )
            return True  # stop

        return False  # continue
