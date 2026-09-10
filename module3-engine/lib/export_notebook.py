"""Export class5.ipynb with all saved outputs → PDF (webpdf, no Pandoc)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CLASS5_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NOTEBOOK = CLASS5_ROOT / "class5.ipynb"


def _ensure_webpdf_deps() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "nbconvert", "playwright"],
        check=False,
    )
    subprocess.run(["playwright", "install", "chromium"], check=False)


def save_pdf(output_name: str = "class5_with_outputs") -> Path:
    """Export notebook to PDF preserving existing cell outputs."""
    nb = DEFAULT_NOTEBOOK
    pdf = CLASS5_ROOT / f"{output_name}.pdf"
    _ensure_webpdf_deps()

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "webpdf",
            "--output",
            output_name,
            str(nb),
        ],
        capture_output=True,
        text=True,
        cwd=CLASS5_ROOT,
    )
    if result.returncode != 0 or not pdf.is_file():
        hint = (result.stderr or result.stdout or "").strip().splitlines()
        msg = hint[-1] if hint else "unknown error"
        raise RuntimeError(
            f"PDF export failed: {msg}\n"
            "On Lambda (venv active):\n"
            "  playwright install chromium\n"
            "  sudo $(which python) -m playwright install-deps chromium"
        )
    return pdf


# backwards compat
save_for_download = save_pdf
