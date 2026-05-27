"""Preflight checks before any multi-task / multi-rollout B-block run.

Per codex retrospective R5 + top-action 1. Halts BEFORE budget is spent
if any of the following are broken:

  1. qcow2 file size matches the zip's expected member size (catches
     truncated Ubuntu.qcow2 — root cause of the v2 5/5 crash on 2026-05-26)
  2. vllm /v1/models returns the right served model id
  3. vllm /v1/chat/completions accepts a real (tiny) chat request and returns
     a non-empty response — distinguishes "boot succeeded" from "serves
     traffic" (vllm has been observed to pass /v1/models then crash on
     first request)
  4. OSWorld DesktopEnv can boot one container + reach the VM screenshot
     endpoint (the path that times out at 300s when qcow2 is corrupt)

Exit code 0 = all passed; 2 = at least one gate failed (halt downstream).

Usage:
    python scripts/preflight.py \\
      --vllm-endpoint http://127.0.0.1:8000/v1 \\
      --vllm-model /root/models/Qwen3.5-9B \\
      --qcow2 /root/visual-gepa/osworld_cache/docker_vm_data/Ubuntu.qcow2 \\
      --qcow2-zip /root/visual-gepa/osworld_cache/docker_vm_data/Ubuntu.qcow2.zip \\
      --osworld-task third_party/OSWorld/evaluation_examples/examples/libreoffice_calc/1954cced-e748-45c4-9c26-9855b97fbc5e.json \\
      --osworld-cache-dir /root/visual-gepa/osworld_cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


def gate_qcow2_size(qcow2: str, qcow2_zip: str) -> tuple[bool, str]:
    if not qcow2 or not qcow2_zip:
        return True, "(skipped: paths not provided)"
    pq = Path(qcow2)
    pz = Path(qcow2_zip)
    if not pz.exists():
        return False, f"qcow2_zip missing: {pz}"
    try:
        with zipfile.ZipFile(pz) as zf:
            # Find the qcow2 member
            members = [m for m in zf.infolist() if m.filename.endswith(".qcow2")]
            if not members:
                return False, f"no .qcow2 member in {pz}"
            expected = members[0].file_size
    except Exception as e:
        return False, f"zip read error: {type(e).__name__}: {e}"

    if not pq.exists():
        return False, f"qcow2 file missing: {pq} (re-extract from {pz})"
    actual = pq.stat().st_size
    if actual != expected:
        return False, (
            f"qcow2 SIZE MISMATCH: actual={actual:,} bytes, expected={expected:,} bytes "
            f"({actual/expected*100:.1f}% of expected) → likely truncated; re-extract:  "
            f"rm {pq} && unzip {pz} -d {pq.parent}"
        )
    return True, f"qcow2 size OK ({actual:,} bytes = expected)"


def gate_vllm_models(endpoint: str, model: str) -> tuple[bool, str]:
    url = endpoint.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode())
        ids = [m.get("id", "") for m in data.get("data", [])]
        if model in ids:
            return True, f"vllm serves {model}"
        return False, f"vllm /v1/models does NOT include {model!r}; serves {ids}"
    except urllib.error.URLError as e:
        return False, f"vllm /v1/models unreachable: {e}"
    except Exception as e:
        return False, f"vllm /v1/models error: {type(e).__name__}: {e}"


def gate_vllm_chat(endpoint: str, model: str) -> tuple[bool, str]:
    """Real chat request — distinguishes 'boot OK' from 'serves traffic'."""
    url = endpoint.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 8,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode())
    except Exception as e:
        return False, f"vllm chat failed: {type(e).__name__}: {str(e)[:200]}"
    dt = time.perf_counter() - t0
    text = (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
    if not text:
        return False, f"vllm chat returned empty content (latency={dt:.1f}s)"
    return True, f"vllm chat returned {text!r} in {dt:.1f}s"


def gate_osworld_env(task_path: str, cache_dir: str | None = None, timeout: int = 240) -> tuple[bool, str]:
    """Boot a real DesktopEnv container + reset on one task; close on success."""
    if not task_path:
        return True, "(skipped: --osworld-task not provided)"
    try:
        # Lazy import — only available on the server with OSWorld installed
        from desktop_env.desktop_env import DesktopEnv
        from desktop_env.providers.docker import manager as docker_manager
    except Exception as e:
        return False, f"desktop_env import failed: {type(e).__name__}: {e}"

    if cache_dir:
        vms_dir = str(Path(cache_dir) / "docker_vm_data")
        Path(vms_dir).mkdir(parents=True, exist_ok=True)
        docker_manager.VMS_DIR = vms_dir

    task = json.loads(Path(task_path).read_text())
    env = None
    t0 = time.perf_counter()
    try:
        env = DesktopEnv(
            provider_name="docker",
            os_type="Ubuntu",
            headless=True,
            cache_dir=cache_dir or "osworld_cache",
        )
        obs = env.reset(task_config=task)
        dt = time.perf_counter() - t0
        if not isinstance(obs, dict) or "screenshot" not in obs:
            return False, f"env.reset returned bad obs: {type(obs).__name__}"
        return True, (
            f"OSWorld boot + reset OK in {dt:.1f}s "
            f"(task={task.get('id', '?')[:36]}, screenshot bytes={len(obs.get('screenshot', b''))})"
        )
    except Exception as e:
        return False, f"OSWorld env failed in {time.perf_counter()-t0:.1f}s: {type(e).__name__}: {str(e)[:200]}"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-endpoint", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--vllm-model", default=os.environ.get("VLLM_MODEL_NAME", "/root/models/Qwen3.5-9B"))
    ap.add_argument("--qcow2", default="")
    ap.add_argument("--qcow2-zip", default="")
    ap.add_argument("--osworld-task", default="")
    ap.add_argument("--osworld-cache-dir", default=None)
    ap.add_argument("--skip-osworld", action="store_true", help="Skip the slow VM-boot gate")
    args = ap.parse_args()

    gates = []
    print("=== preflight ===")

    print("\n[1/4] qcow2 size matches zip-member size?")
    ok, msg = gate_qcow2_size(args.qcow2, args.qcow2_zip)
    gates.append(("qcow2_size", ok, msg))
    print(f"  {'✓' if ok else '✗'}  {msg}")

    print("\n[2/4] vllm /v1/models?")
    ok, msg = gate_vllm_models(args.vllm_endpoint, args.vllm_model)
    gates.append(("vllm_models", ok, msg))
    print(f"  {'✓' if ok else '✗'}  {msg}")

    print("\n[3/4] vllm /v1/chat/completions (real request)?")
    if gates[-1][1]:
        ok, msg = gate_vllm_chat(args.vllm_endpoint, args.vllm_model)
    else:
        ok, msg = False, "(skipped: /v1/models failed)"
    gates.append(("vllm_chat", ok, msg))
    print(f"  {'✓' if ok else '✗'}  {msg}")

    if args.skip_osworld:
        print("\n[4/4] OSWorld env boot — SKIPPED")
        gates.append(("osworld_env", True, "(skipped via --skip-osworld)"))
    else:
        print("\n[4/4] OSWorld env boot + reset (up to 240s)?")
        ok, msg = gate_osworld_env(args.osworld_task, args.osworld_cache_dir)
        gates.append(("osworld_env", ok, msg))
        print(f"  {'✓' if ok else '✗'}  {msg}")

    print("\n=== summary ===")
    all_ok = all(g[1] for g in gates)
    for name, ok, msg in gates:
        print(f"  {'✓' if ok else '✗'}  {name}: {msg}")

    if not all_ok:
        print("\n🚨 PREFLIGHT FAILED — do not launch the multi-task run.")
        return 2
    print("\n✓ all gates passed — safe to launch downstream run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
