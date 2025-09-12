#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_top.py - terminal simplified nvitop
Dependency: pip install pynvml psutil rich pandas
Run: python3 gpu_top.py
Ctrl+C exit
"""

import time
import sys
from typing import Tuple

import psutil
import pandas as pd

from datetime import datetime

try:
    import pynvml as N
except Exception:
    print("Please install dependencies first: pip install pynvml psutil rich pandas")
    sys.exit(1)

from rich.console import Console, Group
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.progress_bar import ProgressBar
from rich.text import Text

# ===== Config =====
ACTIVE_ONLY   = False      # True: Only show GPUs with processes
SORT_PROCS_BY = "mem"      # "mem" | "pid"
SHOW_CMDLINE  = True
MAX_CMD_LEN   = 30
REFRESH_SEC   = 1.0        # Refresh interval (seconds)
USE_FULL_SCREEN = True     # Live(..., screen=True); if blank screen, set to False
# ====================


def bytes2human(n: int) -> str:
    units = ["B","KB","MB","GB","TB","PB"]
    s = 0
    n = int(n or 0)
    x = float(n)
    while x >= 1024 and s < len(units)-1:
        x /= 1024.0
        s += 1
    return f"{x:.1f}{units[s]}"


def _safe(callable_, default=None, *args, **kwargs):
    try:
        return callable_(*args, **kwargs)
    except Exception:
        return default


def _get_nvml_string(func, handle):
    v = _safe(func, None, handle)
    if isinstance(v, (bytes, bytearray)):
        return v.decode(errors="ignore")
    return v if v is not None else "?"


def _get_running_processes(handle):
    # Compatible with different NVML versions
    procs = _safe(N.nvmlDeviceGetComputeRunningProcesses_v3, None, handle)
    if procs is None:
        procs = _safe(N.nvmlDeviceGetComputeRunningProcesses_v2, None, handle)
    if procs is None:
        procs = _safe(N.nvmlDeviceGetComputeRunningProcesses, None, handle)
    return procs or []


def _get_process_util_map(handle):
    """Return {pid: {"sm": smUtil(%), "mem": memUtil(%)}}."""
    util_map = {}
    samples = _safe(N.nvmlDeviceGetProcessUtilization, None, handle, 0)
    if not samples:
        return util_map
    latest = {}
    for s in samples:
        pid = getattr(s, "pid", None)
        ts  = getattr(s, "timeStamp", 0)
        sm  = getattr(s, "smUtil", None)
        mm  = getattr(s, "memUtil", None)
        if pid is None:
            continue
        if (pid not in latest) or (ts > latest[pid]["ts"]):
            latest[pid] = {"ts": ts, "sm": sm, "mem": mm}
    for pid, v in latest.items():
        util_map[pid] = {"sm": v["sm"], "mem": v["mem"]}
    return util_map


def _username(pid):
    try:
        return psutil.Process(pid).username()
    except Exception:
        return "?"


def _cmdline(pid, max_len=10, show_cmd=True):
    try:
        p = psutil.Process(pid)
        if show_cmd:
            parts = p.cmdline()
            text = " ".join(parts) if parts else p.name()
        else:
            text = p.name()
        if not text:
            text = "?"
        if len(text) > max_len:
            return text[:max_len-3] + "..."
        return text
    except Exception:
        return "?"


def gpu_snapshot(active_only=False, sort_by="mem", show_cmd=True, max_cmd_len=10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Take one GPU snapshot, return (df_gpus, df_procs)"""
    try:
        N.nvmlInit()
    except Exception as e:
        # Return empty table, caller will handle
        return pd.DataFrame(), pd.DataFrame()

    try:
        count = _safe(N.nvmlDeviceGetCount, 0)
        g_rows = []
        p_rows = []

        for idx in range(count):
            h = _safe(N.nvmlDeviceGetHandleByIndex, None, idx)
            if h is None:
                continue

            name   = _get_nvml_string(N.nvmlDeviceGetName, h)

            mem    = _safe(N.nvmlDeviceGetMemoryInfo, None, h)
            used, total = (mem.used, mem.total) if mem else (0, 0)

            util   = _safe(N.nvmlDeviceGetUtilizationRates, None, h)
            util_gpu = util.gpu if util else 0
            util_mem = util.memory if util else 0

            temp   = _safe(N.nvmlDeviceGetTemperature, None, h, N.NVML_TEMPERATURE_GPU)
            fan    = _safe(N.nvmlDeviceGetFanSpeed, None, h)
            pstate = _safe(N.nvmlDeviceGetPerformanceState, None, h)

            power        = _safe(N.nvmlDeviceGetPowerUsage, None, h)             # mW
            power_limit  = _safe(N.nvmlDeviceGetEnforcedPowerLimit, None, h)     # mW
            power_w      = f"{power/1000:.0f}" if (power is not None) else "NA"
            power_lim_w  = f"{power_limit/1000:.0f}" if (power_limit is not None) else "NA"
            power_str    = f"{power_w}/{power_lim_w} W" if (power is not None and power_limit is not None) else "NA"

            procs = _get_running_processes(h)
            proc_count = len(procs)

            proc_util_map = _get_process_util_map(h)

            g_rows.append({
                "GPU": idx,
 #               "Name": name,
                "util_gpu": int(util_gpu) if isinstance(util_gpu, (int,float)) else 0,
                "util_mem": int(util_mem) if isinstance(util_mem, (int,float)) else 0,
                "mem_used_bytes": int(used) if isinstance(used, (int,float)) else 0,
                "mem_total_bytes": int(total) if isinstance(total, (int,float)) else 0,
                "Memory(Used/Total)": f"{bytes2human(used)}/{bytes2human(total)}",
                "Temp": (f"{temp}°C" if temp is not None else "NA"),
                "Power": power_str,
                # "P-St": (f"P{pstate}" if pstate is not None else "NA"),
                "Fan": (f"{fan}%" if fan is not None else "NA"),
                "Processes": proc_count,
            })

            for pr in procs:
                pid = getattr(pr, "pid", None)
                mem_used = getattr(pr, "usedGpuMemory", None)  # bytes
                mem_pct = None
                if isinstance(mem_used, int) and isinstance(total, int) and total > 0:
                    mem_pct = 100.0 * mem_used / total

                sm_util = None
                if pid is not None and pid in proc_util_map:
                    smv = proc_util_map[pid].get("sm", None)
                    if isinstance(smv, (int,float)):
                        sm_util = float(smv)

                p_rows.append({
                    "GPU": idx,
                    "PID": pid if pid is not None else "?",
                    "User": _username(pid) if pid is not None else "?",
                    "Mem": bytes2human(mem_used) if isinstance(mem_used, int) else "NA",
                    "Mem %": (f"{mem_pct:.1f}" if isinstance(mem_pct, float) else "NA"),
                    "GPU %": (f"{sm_util:.0f}" if isinstance(sm_util, float) else "NA"),
                    "Cmd": _cmdline(pid, max_cmd_len, show_cmd) if pid is not None else "?",
                    "_mem_sort": mem_used if isinstance(mem_used, int) else -1
                })

        df_gpus  = pd.DataFrame(g_rows).sort_values("GPU").reset_index(drop=True)
        df_procs = pd.DataFrame(p_rows)

        if active_only and not df_gpus.empty:
            active_set = set(df_gpus.loc[df_gpus["Processes"] > 0, "GPU"])
            df_gpus  = df_gpus[df_gpus["GPU"].isin(active_set)].reset_index(drop=True)
            if not df_procs.empty:
                df_procs = df_procs[df_procs["GPU"].isin(active_set)].reset_index(drop=True)

        if not df_procs.empty:
            if sort_by == "pid":
                df_procs = df_procs.sort_values(["GPU", "PID"]).reset_index(drop=True)
            else:
                df_procs = df_procs.sort_values(by=["_mem_sort","GPU","PID"], ascending=[False, True, True]).reset_index(drop=True)
            if "_mem_sort" in df_procs.columns:
                df_procs = df_procs.drop(columns=["_mem_sort"])

        return df_gpus, df_procs

    finally:
        try:
            N.nvmlShutdown()
        except Exception:
            pass


def make_bar(value: float, width: int = 30, color: str = "green"):
    """Return a renderable ProgressBar (value: 0..100)"""
    if value is None:
        return Text("NA")
    v = int(round(max(0, min(100, value))))
    return ProgressBar(total=100, completed=v, width=width, complete_style=color)


def make_tables():
    df_gpus, df_procs = gpu_snapshot(ACTIVE_ONLY, SORT_PROCS_BY, SHOW_CMDLINE, MAX_CMD_LEN)

    # If NVML cannot init or no GPU
    if df_gpus.empty:
        t_empty = Table(expand=True)
        t_empty.add_column("Info")
        t_empty.add_row("No GPU detected or NVML init failed")
        return t_empty, t_empty, t_empty, t_empty, df_gpus, df_procs

    # === GPU Overview (with bars)===
    t1 = Table(expand=True, show_lines=False)
    t1.add_column("Idx", justify="center", no_wrap=True)
#    t1.add_column("Name", justify="left", no_wrap=False, ratio=2)
    t1.add_column("GPU Util", justify="left")
    t1.add_column("Memory", justify="left")
    t1.add_column("GPU %", justify="center")
    t1.add_column("Mem %", justify="center")
    t1.add_column("Temp", justify="center")
    t1.add_column("Power", justify="center")
    # t1.add_column("P-St", justify="center")
    t1.add_column("Proc", justify="center")

    for _, row in df_gpus.iterrows():
        idx = row["GPU"]
        util_gpu = int(row.get("util_gpu", 0))
        mem_used = int(row.get("mem_used_bytes", 0))
        mem_total = int(row.get("mem_total_bytes", 1)) or 1
        mem_ratio = 100.0 * mem_used / mem_total if mem_total > 0 else 0.0

        color = "green" if util_gpu < 50 else "yellow" if util_gpu < 80 else "red"
        util_bar = make_bar(util_gpu, width=24, color=color)
        mem_bar = make_bar(mem_ratio, width=24, color="cyan")

        t1.add_row(
            str(idx),
#            str(row.get("Name", "?")),
            util_bar,
            mem_bar,
            str(util_gpu),
            str(int(mem_ratio)),
            str(row.get("Temp", "NA")),
            str(row.get("Power", "NA")),
            # str(row.get("P-State", "NA")),
            str(row.get("Processes", 0)),
        )

    # === Process Table ===
    t2 = Table(expand=True)
    if df_procs.empty:
        t2.add_column("Info")
        t2.add_row("No GPU processes")
    else:
        for col in df_procs.columns:
            t2.add_column(str(col), overflow="fold")
        for _, row in df_procs.iterrows():
            cells = [str(row[c]) for c in df_procs.columns]
            t2.add_row(*cells)
    
    return t1, t2, df_gpus, df_procs



def build_layout(t1, t2, df_gpus, df_procs):
    # header
    total_gpus = len(df_gpus) if hasattr(df_gpus, "__len__") else 0
    total_procs = len(df_procs) if hasattr(df_procs, "__len__") else 0
    header_text = Text(f" GPUs: {total_gpus}    GPU Procs: {total_procs}    Refresh Second: {REFRESH_SEC}s    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ", style="bold white on dark_blue")
    header = Panel(header_text, style="bold", padding=(0,0))

    # left_group = Group(t1, Panel(t3, title="Each GPU Memory Usage"), Panel(t4, title="Each GPU Utilization"))
    left_panel = Panel(t1, title="GPU Overview")

    right_panel = Panel(t2, title="Processes")

    layout = Layout()
    layout.split_column(
        Layout(header, size=3),
        Layout(name="main")
    )
    layout["main"].split_row(
        Layout(left_panel, ratio=2),
        Layout(right_panel, ratio=3)
    )
    return layout


def main():
    console = Console()
    try:
        with Live(console=console, refresh_per_second=1, screen=USE_FULL_SCREEN) as live:
            while True:
                t1, t2, df_gpus, df_procs = make_tables()
                # make_tables may return a single table duplicated when NVML failure
                if isinstance(t1, Table) and t1.columns and t1.columns[0].header == "Info" and df_gpus.empty:
                    # NVML unavailable or no GPU
                    layout = Layout()
                    layout.split_column(
                        Layout(Panel(t1, title="Info"), size=6)
                    )
                    live.update(layout)
                    time.sleep(REFRESH_SEC)
                    continue

                layout = build_layout(t1, t2, df_gpus, df_procs)
                live.update(layout)
                time.sleep(REFRESH_SEC)
    except KeyboardInterrupt:
        console.print("\n[bold]Esc nvitop[/bold]")
    except Exception as e:
        console.print(f"[red]Error occurred: {e}[/red]")


if __name__ == "__main__":
    main()
