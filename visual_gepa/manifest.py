"""Per-experiment provenance manifest writer.

Per codex retrospective R6 — adds a machine-readable manifest alongside
the result JSON so paper-grade reproducibility doesn't depend on
human-written LOG entries.

Manifest schema (str-only fields, JSON-serializable):
  experiment_id     : str (e.g. "B2_mini_seed42_2026-05-27T08:00:00Z")
  block             : str (e.g. "B2", "B1.baseline", "B0.smoke")
  started_at        : ISO timestamp
  finished_at       : ISO timestamp (filled by finish())
  elapsed_s         : float
  git_sha           : str  (short)
  git_branch        : str
  git_dirty         : bool (uncommitted changes present)
  host_id           : str  (hostname or vast.ai instance IP:port)
  gpu_name          : str
  gpu_memory_gb     : int
  python_version    : str
  torch_version     : str
  vllm_version      : str
  vllm_cmd          : str  (verbatim — env vars + flags)
  model_path        : str
  model_md5         : str  (lazy — compute only if requested)
  qcow2_path        : str (if OSWorld)
  qcow2_size_bytes  : int  (sanity check vs Ubuntu.qcow2.zip published size)
  qcow2_md5         : str  (lazy)
  config_path       : str  (configs/osworld_b1_5.json etc.)
  config_md5        : str
  rng_seed          : int
  result_path       : str
  notes             : str  (free-form, any extra context)

Usage:
    from visual_gepa.manifest import Manifest
    m = Manifest(experiment_id="B2_mini_2026-05-27", block="B2")
    m.start(host_id=..., gpu_name=..., vllm_cmd=..., config_path=..., seed=42)
    # ... run experiment ...
    m.finish(result_path="results/B2_mini.json", notes="all 5 tasks ran")
    m.write("results/B2_mini_manifest.json")
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _git(*args: str, cwd: str | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def _file_md5(path: str | Path, max_bytes: int | None = None) -> str:
    """Compute md5 of a file. If max_bytes given, only hash the first N bytes."""
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.md5()
    with p.open("rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            h.update(f.read(max_bytes))
    return h.hexdigest()


def _gpu_info() -> tuple[str, int]:
    """Return (gpu_name, total_memory_gb) via nvidia-smi if available."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        name, mem_mib = [x.strip() for x in out.split(",")]
        return name, int(int(mem_mib) // 1024)
    except Exception:
        return "", 0


def _torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except Exception:
        return ""


def _vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "")
    except Exception:
        return ""


@dataclass
class Manifest:
    experiment_id: str
    block: str
    started_at: str = ""
    finished_at: str = ""
    elapsed_s: float = 0.0
    git_sha: str = ""
    git_branch: str = ""
    git_dirty: bool = False
    host_id: str = ""
    gpu_name: str = ""
    gpu_memory_gb: int = 0
    python_version: str = ""
    torch_version: str = ""
    vllm_version: str = ""
    vllm_cmd: str = ""
    model_path: str = ""
    model_md5: str = ""
    qcow2_path: str = ""
    qcow2_size_bytes: int = 0
    qcow2_md5: str = ""
    config_path: str = ""
    config_md5: str = ""
    rng_seed: int = 0
    result_path: str = ""
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default=0.0, repr=False)

    def start(
        self,
        host_id: str | None = None,
        vllm_cmd: str = "",
        model_path: str = "",
        qcow2_path: str = "",
        config_path: str = "",
        seed: int = 0,
        compute_model_md5: bool = False,
        compute_qcow2_md5: bool = False,
    ) -> None:
        self._t0 = time.perf_counter()
        self.started_at = datetime.datetime.utcnow().isoformat() + "Z"
        self.host_id = host_id or socket.gethostname()
        self.gpu_name, self.gpu_memory_gb = _gpu_info()
        self.python_version = platform.python_version()
        self.torch_version = _torch_version()
        self.vllm_version = _vllm_version()
        self.vllm_cmd = vllm_cmd
        self.model_path = model_path
        self.qcow2_path = qcow2_path
        if qcow2_path and Path(qcow2_path).exists():
            self.qcow2_size_bytes = Path(qcow2_path).stat().st_size
            if compute_qcow2_md5:
                self.qcow2_md5 = _file_md5(qcow2_path)
        if compute_model_md5 and model_path:
            # Model dir, not single file; fingerprint by hashing config.json + tokenizer.json sizes
            try:
                marks = []
                for fname in ("config.json", "tokenizer.json"):
                    p = Path(model_path) / fname
                    if p.exists():
                        marks.append(f"{fname}:{p.stat().st_size}")
                self.model_md5 = hashlib.md5("|".join(marks).encode()).hexdigest()[:16] or ""
            except Exception:
                self.model_md5 = ""
        self.config_path = config_path
        if config_path:
            self.config_md5 = _file_md5(config_path)
        self.rng_seed = seed
        self.git_sha = _git("rev-parse", "--short", "HEAD") or _git(
            "rev-parse", "--short", "HEAD", cwd=str(Path(__file__).resolve().parent.parent)
        )
        self.git_branch = _git("rev-parse", "--abbrev-ref", "HEAD") or _git(
            "rev-parse", "--abbrev-ref", "HEAD", cwd=str(Path(__file__).resolve().parent.parent)
        )
        dirty = _git("status", "--porcelain") or _git(
            "status", "--porcelain", cwd=str(Path(__file__).resolve().parent.parent)
        )
        self.git_dirty = bool(dirty)

    def finish(self, result_path: str = "", notes: str = "") -> None:
        self.finished_at = datetime.datetime.utcnow().isoformat() + "Z"
        self.elapsed_s = round(time.perf_counter() - self._t0, 3) if self._t0 else 0.0
        self.result_path = result_path
        self.notes = notes

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("_t0", None)
        return d

    def write(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
