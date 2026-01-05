"""
Resource metrics collector for ML-Agents training runs.
It is for collectiong data:
- memory usage avg mb / peak mb
- cpu usage avg percent / peak percent (can exceed 100 on multi-core)
- gpu usage avg mb / peak mb (NVIDIA only via NVML; empty on Macs without NVML)

"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

try:
    import pynvml  # type: ignore
except Exception:
    pynvml = None  # type: ignore


class ResourceMonitor(threading.Thread):

    def __init__(self, ml_pid: Optional[int], interval_s: float = 1.0):
        super().__init__(daemon=True)
        self._interval_s = max(0.2, float(interval_s))
        self._stop_event = threading.Event()
        self._ml_proc = None
        self._ml_pid = ml_pid
        self._monitor_children = True  # Monitor child processes too
        self._initialized_procs = (
            set()
        )  # Track which processes we've initialized for CPU
        self._nvml_initialized = False
        if psutil and ml_pid:
            try:
                self._ml_proc = psutil.Process(ml_pid)
                # Initialize CPU percent calculation (first call returns 0.0)
                _ = self._ml_proc.cpu_percent(interval=None)
                self._initialized_procs.add(ml_pid)
                print(f"ResourceMonitor: Monitoring process {ml_pid}")
            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess,
            ) as e:
                print(f"Warning: Could not attach to process {ml_pid}: {e}")
                self._ml_proc = None
            except Exception as e:
                print(f"Warning: Error related to process {ml_pid}:{e}")
                self._ml_proc = None
        else:
            if not psutil:
                print(
                    "Warning: psutil unavailable,"
                    " CPU and memory monitoring disabled"
                )
            if not ml_pid:
                print("Warning: No process ID, resource monitoring disabled")
        if pynvml:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
                print("ResourceMonitor: GPU monitoring initialized")
            except Exception as e:
                print(f"Warning: Could not initialize GPU monitoring: {e}")
                self._nvml_initialized = False
        else:
            print("Warning: pynvml not available, GPU monitoring disabled")
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
        while not self._stop_event.is_set():
            t0 = time.time()
            self._sample_once()
            remaining = self._interval_s - (time.time() - t0)
            if remaining > 0:
                time.sleep(remaining)

        if pynvml and self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()

    def _get_process_tree_stats(self, proc):
        """Get CPU and memory stats for a process and all its children."""
        cpu_total = 0.0
        mem_total = 0.0

        try:
            # Initialize parent process if not already initialized
            proc_pid = proc.pid
            parent_newly_init = False
            if proc_pid not in self._initialized_procs:
                try:
                    _ = proc.cpu_percent(interval=None)
                    self._initialized_procs.add(proc_pid)
                    parent_newly_init = True
                except Exception:
                    pass

            # Get parent process stats
            # For newly initialized processes,
            # use a blocking call to get immediate reading
            # For already initialized processes, use non-blocking call
            if parent_newly_init:
                # Use blocking call to get immediate CPU reading
                # (blocks for 0.1s)
                try:
                    cpu_val = float(proc.cpu_percent(interval=0.1))
                    cpu_total += cpu_val
                except Exception:
                    pass
            else:
                # Use non-blocking call (returns CPU since last call)
                try:
                    cpu_val = float(proc.cpu_percent(interval=None))
                    cpu_total += cpu_val
                except Exception:
                    pass
            mem_total += float(proc.memory_info().rss) / (1024 * 1024)

            # Get all child processes
            if self._monitor_children:
                try:
                    children = proc.children(recursive=True)
                    if (
                        children and self._cpu_count == 0
                    ):  # Debug: print once on first sample
                        print(
                            f"ResourceMonitor: Found {len(children)}"
                            "child process(es) to monitor"
                        )
                    for child in children:
                        try:
                            if child.is_running():
                                child_pid = child.pid
                                child_newly_init = False
                                # Initialize child process
                                # if not already initialized
                                if child_pid not in self._initialized_procs:
                                    try:
                                        _ = child.cpu_percent(interval=None)
                                        self._initialized_procs.add(child_pid)
                                        child_newly_init = True
                                    except Exception:
                                        pass
                                # Get child stats
                                if child_newly_init:
                                    # blocking call for newly initialized child
                                    try:
                                        child_cpu_val = float(
                                            child.cpu_percent(interval=0.1)
                                        )
                                        cpu_total += child_cpu_val
                                    except Exception:
                                        pass
                                else:
                                    # non-blocking call for 
                                    # already initialized child
                                    try:
                                        child_cpu_val = float(
                                            child.cpu_percent(interval=None)
                                        )
                                        cpu_total += child_cpu_val
                                    except Exception:
                                        pass
                                mem_total += float(child.memory_info().rss) / (
                                    1024 * 1024
                                )
                        except (
                            psutil.NoSuchProcess,
                            psutil.AccessDenied,
                            psutil.ZombieProcess,
                        ):
                            # Child process may have terminated, skip it
                            continue
                        except Exception:
                            # Other errors, skip this child
                            continue
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Can't get children
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied,
                psutil.ZombieProcess):
            return None, None
        except Exception:
            return None, None

        return cpu_total, mem_total

    def _sample_once(self) -> None:
        cpu = None
        mem = None
        if self._ml_proc and psutil:
            try:
                # Check if process still exists
                if not self._ml_proc.is_running():
                    self._ml_proc = None
                else:
                    # Get stats for process and all children
                    cpu, mem = self._get_process_tree_stats(self._ml_proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied,
                    psutil.ZombieProcess):
                # Process no longer exists or we lost access
                self._ml_proc = None
                cpu = None
                mem = None
        gpu_mem_used = None
        if pynvml and self._nvml_initialized:
            try:
                n = pynvml.nvmlDeviceGetCount()
                if n > 0:
                    total = 0.0
                    for i in range(n):
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        m = pynvml.nvmlDeviceGetMemoryInfo(h)
                        total += float(m.used) / (1024 * 1024)
                    gpu_mem_used = total
            except pynvml.NVMLError as e:
                # NVML-specific error
                if self._gpu_mem_count == 0:  # Only print once
                    print(f"Warning: NVML error during GPU monitoring: {e}")
                gpu_mem_used = None
            except Exception as e:
                # Other errors
                if self._gpu_mem_count == 0:  # Only print once
                    print(
                        f"Warning: Unexpected error during GPU monitoring: {e}"
                        )
                gpu_mem_used = None

        if isinstance(cpu, (int, float)):
            self._cpu_total += float(cpu)
            self._cpu_count += 1
            if cpu > self._cpu_peak:
                self._cpu_peak = float(cpu)
            # Debug: print first few CPU readings to verify it's working
            if self._cpu_count <= 3:
                print(f"ResourceMonitor: CPU sample {self._cpu_count}:"
                      " {cpu:.2f}%")
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
            "memory usage avg mb": (
                (self._mem_total / self._mem_count)
                if self._mem_count > 0 else None
            ),
            "memory usage peak mb": self._mem_peak
            if self._mem_count > 0 else None,
            "cpu usage avg percent": (
                (self._cpu_total / self._cpu_count)
                if self._cpu_count > 0 else None
            ),
            "cpu usage peak percent": self._cpu_peak
            if self._cpu_count > 0 else None,
            "gpu usage avg mb": (
                (self._gpu_mem_total / self._gpu_mem_count)
                if self._gpu_mem_count > 0
                else None
            ),
            "gpu usage peak mb": (
                self._gpu_mem_peak if self._gpu_mem_count > 0 else None
            ),
        }
        self._rows.append(row)

    def rows(self) -> List[Dict[str, Optional[float]]]:
        return list(self._rows)

    def calculate_resource_usage_data(self) -> Dict[str, Optional[float]]:
        """
        Return the latest aggregated metrics
        without modifying internal buffers.
        """
        mem_avg = (
            self._mem_total / self._mem_count
            if self._mem_count > 0 else None)
        mem_peak = self._mem_peak if self._mem_count > 0 else None
        cpu_avg = (
            self._cpu_total / self._cpu_count
            if self._cpu_count > 0 else None)
        cpu_peak = self._cpu_peak if self._cpu_count > 0 else None
        gpu_avg = (
            (self._gpu_mem_total / self._gpu_mem_count)
            if self._gpu_mem_count > 0
            else None
        )
        gpu_peak = self._gpu_mem_peak if self._gpu_mem_count > 0 else None
        return {
            "memory usage avg mb": mem_avg,
            "memory usage peak mb": mem_peak,
            "cpu usage avg percent": cpu_avg,
            "cpu usage peak percent": cpu_peak,
            "gpu usage avg mb": gpu_avg,
            "gpu usage peak mb": gpu_peak,
        }
