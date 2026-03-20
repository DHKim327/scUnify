"""Training-specific progress tracking.

Separate from ``logger.py`` (inference) to avoid modifying existing code.
Adds Epoch and Loss columns to the Rich progress table.
"""

import time

import ray
from rich.console import Console
from rich.live import Live
from rich.table import Table


@ray.remote
class TrainingProgressActor:
    """Training progress actor — parallel to ProgressActor (inference)."""

    def __init__(self):
        self.rows: dict[tuple[str, int], dict] = {}

    def register(
        self,
        task_name: str,
        rank: int,
        total_batches: int | None,
        batch_size: int | None,
        total_epochs: int | None = None,
    ):
        self.rows[(task_name, rank)] = dict(
            task=task_name,
            rank=rank,
            gpu="-",
            batch_size=batch_size if batch_size is not None else "-",
            done=0,
            total=(total_batches if total_batches is not None else 0),
            epoch=0,
            total_epochs=(total_epochs if total_epochs is not None else 0),
            loss=None,
            val_loss=None,
            status="PENDING",
            ts=time.time(),
        )

    def update(
        self,
        task_name: str,
        rank: int,
        gpu_id: int,
        done: int,
        total: int,
        batch_size: int,
        epoch: int | None = None,
        total_epochs: int | None = None,
        loss: float | None = None,
        val_loss: float | None = None,
    ):
        row = self.rows.setdefault(
            (task_name, rank),
            dict(
                task=task_name,
                rank=rank,
                gpu="-",
                batch_size=batch_size,
                done=0,
                total=total,
                epoch=0,
                total_epochs=0,
                loss=None,
                val_loss=None,
                status="PENDING",
            ),
        )
        row.update(
            gpu=gpu_id,
            batch_size=batch_size,
            done=done,
            total=total,
            status="TRAINING",
            ts=time.time(),
        )
        if epoch is not None:
            row["epoch"] = epoch
        if total_epochs is not None:
            row["total_epochs"] = total_epochs
        if loss is not None:
            row["loss"] = loss
        if val_loss is not None:
            row["val_loss"] = val_loss

    def set_status(self, task_name: str, rank: int, status: str):
        r = self.rows.get((task_name, rank))
        if r:
            r["status"] = status
            r["ts"] = time.time()

    def finish(self, task_name: str, rank: int):
        r = self.rows.get((task_name, rank))
        if r:
            r["done"] = max(r["done"], r["total"])
            r["status"] = "DONE"
            r["ts"] = time.time()

    def snapshot(self):
        return self.rows.copy()


class TrainingProgressUI:
    """Rich Live table for training progress — parallel to ProgressUI."""

    def __init__(
        self,
        progress_actor,
        refresh_hz: int = 4,
        poll_interval: float = 0.25,
    ):
        self.actor = progress_actor
        self.refresh_hz = refresh_hz
        self.poll_interval = poll_interval
        self.console = Console()

    def _render_table(self, rows: dict) -> Table:
        tbl = Table(title="scUnify Training Progress", expand=True)
        tbl.add_column("Task")
        tbl.add_column("Worker")
        tbl.add_column("GPU ID")
        tbl.add_column("BS")
        tbl.add_column("Epoch")
        tbl.add_column("Batches")
        tbl.add_column("Progress")
        tbl.add_column("Loss")
        tbl.add_column("Val Loss")
        tbl.add_column("Status")

        for (_task, _rank), row in sorted(
            rows.items(), key=lambda kv: (kv[0][0], kv[0][1])
        ):
            done, total = row["done"], row["total"]
            bar_len = 20
            filled = int(bar_len * (done / max(1, total))) if total > 0 else 0
            bar = "█" * filled + "-" * (bar_len - filled)

            ep = row.get("epoch", 0)
            ep_total = row.get("total_epochs", 0)
            epoch_str = f"{ep + 1}/{ep_total}" if ep_total else f"{ep + 1}/?"

            loss_val = row.get("loss")
            loss_str = f"{loss_val:.4f}" if loss_val is not None else "-"

            vloss_val = row.get("val_loss")
            vloss_str = f"{vloss_val:.4f}" if vloss_val is not None else "-"

            tbl.add_row(
                row["task"],
                str(row["rank"]),
                str(row["gpu"]),
                str(row["batch_size"]),
                epoch_str,
                f"{done}/{total}" if total else f"{done}/?",
                bar if total else "·" * bar_len,
                loss_str,
                vloss_str,
                row.get("status", ""),
            )
        return tbl

    def run_until_complete(self):
        with Live(
            self._render_table({}),
            console=self.console,
            refresh_per_second=self.refresh_hz,
        ) as live:
            while True:
                rows = ray.get(self.actor.snapshot.remote())
                live.update(self._render_table(rows))
                if rows and all(
                    r.get("status") == "DONE" for r in rows.values()
                ):
                    break
                time.sleep(self.poll_interval)
