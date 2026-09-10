"""GPU / Lambda readiness checks for module3-engine (notebook + scripts/lambda_check.sh)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

CLASS5_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or (CLASS5_ROOT / ".env")
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value


def run_gpu_checks(*, load_env: bool = True) -> dict:
    """Return a report dict; does not load TinyLlama (fast check)."""
    if load_env:
        load_dotenv()

    report: dict = {
        "smol_vllm": False,
        "torch": False,
        "cuda_available": False,
        "cuda_device": None,
        "transformers": False,
        "nvidia_smi": False,
        "gpu_name": None,
        "hf_token_set": bool(os.environ.get("HF_TOKEN")),
        "lambda_ssh_set": bool(os.environ.get("LAMBDA")),
        "real_model_ready": False,
        "errors": [],
    }

    try:
        from smol_vllm import LLMEngine  # noqa: F401

        report["smol_vllm"] = True
    except ImportError as exc:
        report["errors"].append(f"smol_vllm import failed: {exc}")

    try:
        import torch

        report["torch"] = True
        report["cuda_available"] = bool(torch.cuda.is_available())
        if report["cuda_available"]:
            report["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        report["errors"].append("torch not installed — pip install -e smol-vllm[tinyllama-1.1b]")

    try:
        import transformers  # noqa: F401

        report["transformers"] = True
    except ImportError:
        report["errors"].append("transformers not installed")

    if shutil.which("nvidia-smi"):
        report["nvidia_smi"] = True
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                timeout=10,
            ).strip()
            if out:
                report["gpu_name"] = out.splitlines()[0]
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            pass

    report["real_model_ready"] = (
        report["smol_vllm"]
        and report["torch"]
        and report["cuda_available"]
        and report["transformers"]
    )
    return report


def print_gpu_report(report: dict | None = None) -> dict:
    report = report or run_gpu_checks()
    ok = lambda v: "OK" if v else "—"

    print("=" * 60)
    print("Class 5 GPU / Lambda readiness")
    print("=" * 60)
    print(f"  smol_vllm import     {ok(report['smol_vllm'])}")
    print(f"  torch                {ok(report['torch'])}")
    print(f"  CUDA available       {ok(report['cuda_available'])}")
    if report["cuda_device"]:
        print(f"  CUDA device          {report['cuda_device']}")
    if report["gpu_name"]:
        print(f"  nvidia-smi GPU       {report['gpu_name']}")
    print(f"  transformers         {ok(report['transformers'])}")
    print(f"  HF_TOKEN in env      {ok(report['hf_token_set'])}")
    print(f"  LAMBDA in .env       {ok(report['lambda_ssh_set'])}  (Mac rsync/tunnel)")
    print(f"  Real model ready     {ok(report['real_model_ready'])}")
    if report["errors"]:
        print("\n  Notes:")
        for err in report["errors"]:
            print(f"    • {err}")
    if report["real_model_ready"]:
        print("\n  → Part D (GPU) cells can run. First load takes ~30s (TinyLlama).")
    else:
        print("\n  → Parts A–C + B use FakeModel on CPU. Run Part D on Lambda (LAMBDA.md).")
    print("=" * 60)
    return report
