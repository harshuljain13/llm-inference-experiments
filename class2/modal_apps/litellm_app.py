from __future__ import annotations

import os

import modal

APP_NAME = "litellm"

app = modal.App(APP_NAME)

NAIVE_SERVER_URL = os.environ.get("NAIVE_SERVER_URL", "")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # Pin to a combo verified to import cleanly together. Two traps:
    #   * the [proxy] extra is required (pulls uvicorn/backoff/etc. the proxy server needs);
    #   * a loose version range let pip resolve a newer FastAPI that REMOVED `get_flat_dependant`,
    #     which litellm 1.80 imports — so we pin fastapi to 0.140.0 which still has it.
    .pip_install("litellm[proxy]==1.80.0", "fastapi==0.140.0")
    .add_local_file(
        local_path="config/litellm_config.yaml",
        remote_path="/root/litellm_config.yaml",
        copy=True,
    )
    .env(
        {
            "NAIVE_SERVER_URL": NAIVE_SERVER_URL,
            "LITELLM_PORT": "4000",
        }
    )
)


@app.function(
    image=image,
    timeout=60 * 60,
    scaledown_window=10 * 60,
    max_containers=1,
)
@modal.web_server(4000, startup_timeout=180)
def serve():
    import subprocess

    # Don't use the `litellm` CLI: its proxy_cli.py does a bare `from proxy_server import ...`
    # that needs litellm/proxy on sys.path via a fragile CWD dance, which fails here with
    # "ModuleNotFoundError: No module named 'proxy_server'". Instead run the proxy's ASGI app
    # directly with uvicorn (fully-qualified import) and hand it the config via CONFIG_FILE_PATH,
    # which proxy_server reads on startup (proxy_server.py: get_secret_str("CONFIG_FILE_PATH")).
    env = os.environ.copy()
    env["CONFIG_FILE_PATH"] = "/root/litellm_config.yaml"
    subprocess.Popen(
        [
            "uvicorn",
            "litellm.proxy.proxy_server:app",
            "--host",
            "0.0.0.0",
            "--port",
            "4000",
        ],
        env=env,
        # stdout/stderr flow to Modal logs so startup failures are visible.
    )
