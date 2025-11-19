"""
Resource metrics collector for ML-Agents training runs.
It is for collectiong data:
- memory usage avg mb / peak mb
- cpu usage avg percent / peak percent   (trainer process CPU %, can exceed 100 on multi-core)
- gpu usage avg mb / peak mb   (NVIDIA only via NVML; empty on Macs without NVML)

"""
from _future_ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None  # type: ignore


class ResourceMonitor(threading.Thread):

    def _init_(self, ml_pid: Optional[int], interval_s: float = 1.0):
        super()._init_(daemon=True)
        self._interval_s = max(0.2, float(interval_s))
        self._stop = threading.Event()
        self._ml_proc = None
        if psutil and ml_pid:
            try:
                self._ml_proc = psutil.Process(ml_pid)
                _ = self._ml_proc.cpu_percent(interval=None)  
            except Exception:
                self._ml_proc = None
        if pynvml:
            try:
                pynvml.nvmlInit()
            except Exception:
                pass
        self._cpu_total: float = 0.0
        self._cpu_count: int = 0
        self._cpu_peak: float = 0.0
        self._mem_total: float = 0.0
        self._mem_count: int = 0
        self._mem_peak: float = 0.0
        self._gpu_mem_total: float = 0.0
        self._gpu_mem_count: int = 0
        self._gpu_mem_peak: float = 0.0
        self._rows: List[Dict[str, Optional[float]]] = []

    def run(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            self._sample_once()
            remaining = self._interval_s - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)
        
        if pynvml:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()

    def _sample_once(self) -> None:
        cpu = None
        mem = None
        if self._ml_proc and psutil:
            try:
                cpu = float(self._ml_proc.cpu_percent(interval=None))
                mem = float(self._ml_proc.memory_info().rss) / (1024 * 1024)
            except Exception:
                cpu = None
                mem = None
        gpu_mem_used = None
        if pynvml:
            try:
                n = pynvml.nvmlDeviceGetCount()
                if n > 0:
                    total = 0.0
                    for i in range(n):
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        m = pynvml.nvmlDeviceGetMemoryInfo(h)
                        total += float(m.used) / (1024 * 1024)
                    gpu_mem_used = total
            except Exception:
                gpu_mem_used = None
        
        if isinstance(cpu, (int, float)):
            self._cpu_total += float(cpu)
            self._cpu_count += 1
            if cpu > self._cpu_peak:
                self._cpu_peak = float(cpu)
        if isinstance(mem, (int, float)):
            self._mem_total += float(mem)
            self._mem_count += 1
            if mem > self._mem_peak:
                self._mem_peak = float(mem)
        if isinstance(gpu_mem_used, (int, float)):
            self._gpu_mem_total += float(gpu_mem_used)
            self._gpu_mem_count += 1
            if gpu_mem_used > self._gpu_mem_peak:
                self._gpu_mem_peak = float(gpu_mem_used)

        
        row = {
            "memory usage avg mb": (self._mem_total / self._mem_count) if self._mem_count > 0 else None,
            "memory usage peak mb": self._mem_peak if self._mem_count > 0 else None,
            "cpu usage avg percent": (self._cpu_total / self._cpu_count) if self._cpu_count > 0 else None,
            "cpu usage peak percent": self._cpu_peak if self._cpu_count > 0 else None,
            "gpu usage avg mb": (self._gpu_mem_total / self._gpu_mem_count) if self._gpu_mem_count > 0 else None,
            "gpu usage peak mb": self._gpu_mem_peak if self._gpu_mem_count > 0 else None,
        }
        self._rows.append(row)

    def rows(self) -> List[Dict[str, Optional[float]]]:
        return list(self._rows)

    def current_metrics(self) -> Dict[str, Optional[float]]:
        """
        Return the latest aggregated metrics without modifying internal buffers.
        """
        mem_avg = (self._mem_total / self._mem_count) if self._mem_count > 0 else None
        mem_peak = self._mem_peak if self._mem_count > 0 else None
        cpu_avg = (self._cpu_total / self._cpu_count) if self._cpu_count > 0 else None
        cpu_peak = self._cpu_peak if self._cpu_count > 0 else None
        gpu_avg = (self._gpu_mem_total / self._gpu_mem_count) if self._gpu_mem_count > 0 else None
        gpu_peak = self._gpu_mem_peak if self._gpu_mem_count > 0 else None
        return {
            "memory usage avg mb": mem_avg,
            "memory usage peak mb": mem_peak,
            "cpu usage avg percent": cpu_avg,
            "cpu usage peak percent": cpu_peak,
            "gpu usage avg mb": gpu_avg,
            "gpu usage peak mb": gpu_peak,
        }