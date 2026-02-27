import time

import ray


@ray.remote
class ProgressActor:
    def __init__(self):
        # key: (task_name, rank)
        self.rows: dict[tuple[str, int], dict] = {}

    # Pre-register a task/rank entry
    def register(self, task_name: str, rank: int, total: int | None, batch_size: int | None):
        self.rows[(task_name, rank)] = dict(
            task=task_name,
            rank=rank,
            gpu="-",  # Not yet assigned
            batch_size=batch_size if batch_size is not None else "-",
            done=0,
            total=(total if total is not None else 0),
            status="PENDING",
            ts=time.time(),
        )

    # Helper to update status only (optional)
    def set_status(self, task_name: str, rank: int, status: str):
        r = self.rows.get((task_name, rank))
        if r:
            r["status"] = status
            r["ts"] = time.time()

    # Update progress and refresh status
    def update(self, task_name: str, rank: int, gpu_id: int, done: int, total: int, batch_size: int):
        row = self.rows.setdefault(
            (task_name, rank),
            dict(task=task_name, rank=rank, gpu="-", batch_size=batch_size, done=0, total=total, status="PENDING"),
        )
        row.update(gpu=gpu_id, batch_size=batch_size, done=done, total=total, status="RUNNING", ts=time.time())

    def finish(self, task_name: str, rank: int):
        r = self.rows.get((task_name, rank))
        if r:
            r["done"] = max(r["done"], r["total"])
            r["status"] = "FINISHED"
            r["ts"] = time.time()

    def snapshot(self):
        return self.rows.copy()


from rich.console import Console
from rich.live import Live
from rich.table import Table


class ProgressUI:
    def __init__(self, progress_actor, refresh_hz: int = 4, poll_interval: float = 0.25):
        self.actor = progress_actor
        self.refresh_hz = refresh_hz
        self.poll_interval = poll_interval
        self.console = Console()

    def _render_table(self, rows):
        tbl = Table(title="scUnify Inference Progress", expand=True)
        tbl.add_column("Task")
        tbl.add_column("Worker")
        tbl.add_column("GPU ID")
        tbl.add_column("Batch Size")
        tbl.add_column("Batches (done/total)")
        tbl.add_column("Progress")
        tbl.add_column("Status")

        for (task_name, rank), row in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            done, total = row["done"], row["total"]
            bar_len = 20
            filled = int(bar_len * (done / max(1, total))) if total > 0 else 0
            bar = "█" * filled + "·" * (bar_len - filled)
            tbl.add_row(
                row["task"],
                str(row["rank"]),
                str(row["gpu"]),
                str(row["batch_size"]),
                (f"{done}/{total}" if total else f"{done}/?"),
                (bar if total else "·" * bar_len),
                row.get("status", ""),
            )
        return tbl

    def run_until_complete(self):
        with Live(self._render_table({}), console=self.console, refresh_per_second=self.refresh_hz) as live:
            while True:
                rows = ray.get(self.actor.snapshot.remote())
                live.update(self._render_table(rows))
                if rows and all(r.get("status") == "FINISHED" for r in rows.values()):
                    break
                time.sleep(self.poll_interval)


import threading

try:
    import pynvml

    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False


class GPUMonitor:
    def __init__(self, device_index: int, interval: float = 0.5):
        self.device_index = int(device_index)
        self.interval = float(interval)
        self.util = []
        self.mem = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        if not _HAS_PYNVML:
            return
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        try:
            while not self._stop.is_set():
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.util.append(util.gpu)  # %
                self.mem.append(mem_info.used)  # bytes
                time.sleep(self.interval)
        finally:
            pynvml.nvmlShutdown()

    def start(self):
        if not _HAS_PYNVML:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not _HAS_PYNVML:
            return None, None, None
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if len(self.util) == 0:
            return 0.0, 0.0, 0
        util_mean = sum(self.util) / len(self.util)
        util_max = max(self.util)
        mem_max = max(self.mem)
        return util_mean, util_max, mem_max


class TimeLogger:
    def __init__(self):
        self._start: dict[str, float] = {}
        self._elapsed: dict[str, float] = {}

    def start(self, name: str):
        self._start[name] = time.time()

    def stop(self, name: str) -> float:
        if name not in self._start:
            return 0.0
        dt = time.time() - self._start.pop(name)
        self._elapsed[name] = self._elapsed.get(name, 0.0) + dt
        return dt

    def get(self, name: str, default: float = 0.0) -> float:
        return self._elapsed.get(name, default)

    def reset(self, name: str | None = None):
        if name is None:
            self._start.clear()
            self._elapsed.clear()
        else:
            self._start.pop(name, None)
            self._elapsed.pop(name, None)

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        if prefix:
            return {f"{prefix}{k}": v for k, v in self._elapsed.items()}
        return dict(self._elapsed)
