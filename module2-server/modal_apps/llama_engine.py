from __future__ import annotations

import modal
import subprocess

APP_NAME = "llama-engine"
GGUF_URL = (
    "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/"
    "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
)
LLAMA_CPP_IMAGE = "ghcr.io/ggml-org/llama.cpp:server-cuda"

app = modal.App(APP_NAME)

gguf_cache = modal.Volume.from_name("class2-gguf-cache", create_if_missing=True)

image = (
    # llama.cpp images set ENTRYPOINT to llama-server; Modal must run Python first.
    modal.Image.from_registry(LLAMA_CPP_IMAGE, add_python="3.11")
    .entrypoint([])
    .env(
        {
            "GGUF_URL": GGUF_URL,
            "GGUF_PATH": "/cache/models/model.gguf",
            "PATH": "$PATH:/app",
        }
    )
)


@app.function(
    image=image,
    gpu="T4",
    timeout=60 * 60,
    scaledown_window=10 * 60,
    max_containers=1,
    volumes={"/cache/models": gguf_cache},
    
)
@modal.web_server(8080, startup_timeout=60)
def serve():
    import os
    import shutil
    import urllib.request
    from pathlib import Path
    
    gguf_path = Path(os.environ.get("GGUF_PATH", "/cache/models/model.gguf"))
    if not gguf_path.exists():
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        url = os.environ.get("GGUF_URL", GGUF_URL)
        print(f"Downloading GGUF to {gguf_path} ...", flush=True)
        urllib.request.urlretrieve(url, str(gguf_path))
        print(f"GGUF ready ({gguf_path.stat().st_size // (1024 * 1024)} MB)", flush=True)

    # The llama.cpp server image installs the binary at /app/llama-server and invokes it via an
    # absolute-path ENTRYPOINT — /app is NOT on PATH, so shutil.which("llama-server") returns None.
    # Prefer the known install path; fall back to PATH / other locations for other images.
    candidates = [
        "/app/llama-server",
        "/usr/local/bin/llama-server",
        "/usr/bin/llama-server",
    ]
    llama_server = next((c for c in candidates if Path(c).exists()), None) or shutil.which("llama-server")
    if not llama_server:
        raise RuntimeError(
            "llama-server not found at /app/llama-server or on PATH. "
            f"PATH={os.environ.get('PATH', '')}"
        )

    # llama-server's shared libs (libllama-server-impl.so, etc.) sit next to the binary in /app,
    # but the image sets no LD_LIBRARY_PATH and the $ORIGIN rpath isn't resolving under exec — so
    # the loader fails with "cannot open shared object file". Point it at the binary's own dir.
    lib_dir = str(Path(llama_server).parent)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        p for p in (lib_dir, os.environ.get("LD_LIBRARY_PATH", "")) if p
    )

    # IMPORTANT: launch llama-server as a SUBPROCESS, not os.execvp(). execvp replaces this
    # process image, which destroys Modal's runtime + heartbeat thread — Modal then sees no
    # heartbeat and kills the container after 900s ("Runner heartbeat timeout"). With @web_server
    # we must keep the Modal runtime alive to proxy port 8080 and heartbeat, so we spawn the
    # server as a child and return; Modal waits for the port (startup_timeout) then proxies.
    cmd = [
        llama_server,
        "--host", "0.0.0.0",
        "--port", "8080",
        "-m", str(gguf_path),
        "-c", "2048",   # TinyLlama's trained context; 4096 just gets capped to this anyway
        "-ngl", "999",  # offload all layers to the T4
    ]
    print(
        f"Starting {' '.join(cmd)} (LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']})",
        flush=True,
    )
    subprocess.Popen(cmd, env=os.environ)
