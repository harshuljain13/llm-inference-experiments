# Class 7 — CrewAI → limiter → gateway → two vLLM replicas

One GPU on Lambda. `--max-num-seqs 8` is the point — do not raise it.

```
module4-gateway/
  app.py          CrewAI client
  limiter.py      app-side rate limiter
  gateway/        Python package  (python -m gateway.main)
  setup/          Mac sync + Lambda launch
  tests/
  bench/
```

`gateway/` must stay a package at the lab root. Do not nest the lab inside `llm-gateway-lab/`.

---

# ON YOUR MAC

Rsync first. SSH into an empty `~/module4-gateway` is a dead end — the lab is not on the GPU until `sync_to_lambda.sh` finishes.

Open a **new** terminal. `pwd` must end with `module4-gateway`.

```
cd module4-gateway
pwd
cp .env.example .env
```

Put your values in `.env` (never commit it):

```
export LAMBDA=ubuntu@YOUR_LAMBDA_IP
export LAMBDA_SSH_KEY=$HOME/.ssh/id_ed25519_lambda
export HF_TOKEN=hf_xxxx
```

`LAMBDA_SSH_KEY` is the **private** key — the file with no `.pub` extension. `ls ~/.ssh` to find yours. If SSH complains the key is too open: `chmod 600 ~/.ssh/id_ed25519_lambda`.

```
bash setup/sync_to_lambda.sh
```

Wait until it prints `Synced`. Then:

```
bash setup/ssh.sh
```

`setup/ssh.sh` reads `.env` itself and drops you on the GPU in `~/module4-gateway`. The prompt is `ubuntu@...`, not `jarvis@...`.

On Lambda, check the copy landed:

```
ls
```

You must see `app.py`, `Makefile`, `gateway/`, `setup/`. If `ls` is empty, `exit` and run `bash setup/sync_to_lambda.sh` on the Mac again.

Every time you change code, rsync again from the Mac before you expect Lambda to see it:

```
cd module4-gateway
bash setup/sync_to_lambda.sh
```

---

# ON LAMBDA — setup (once)

Only after rsync. You are already in `~/module4-gateway` if you used `bash setup/ssh.sh`.

```
bash setup/lambda_setup.sh
```

Every new Lambda tab:

```
cd ~/module4-gateway && source .venv/bin/activate
```

---

# ON LAMBDA — run

```
bash setup/launch_replicas.sh
make smoke

python -m gateway.main --replicas http://127.0.0.1:8001,http://127.0.0.1:8002
```

`launch_replicas.sh` brings the replicas up **one at a time** — 8001 must answer `/v1/models` before 8002 starts. Two vLLM engines profiling GPU memory at the same time race each other and one dies at engine-core init. First launch is slow (model download + compile, several minutes per replica); that is normal.

Each replica logs to its own file, not to your terminal:

```
/tmp/llm-gateway-lab-8001.log
/tmp/llm-gateway-lab-8002.log
```

Leave `gateway.main` running in this tab. It holds the terminal.

### If a replica fails to start

`make smoke` printing `FAIL :8001` and `PASS :8002` means one engine died. The launcher already tailed that replica's log for you; the real error is in there, above the `Engine core initialization failed` traceback.

Clear the GPU and retry — a dead replica's sibling is still holding memory:

```
pkill -f "vllm serve" ; sleep 5 ; nvidia-smi
bash setup/launch_replicas.sh
```

`nvidia-smi` must show ~0 MiB in use before you relaunch. If it still fails: drop both replicas to `--gpu-memory-utilization 0.30`, or clear a stale compile cache with `rm -rf ~/.cache/vllm`.

---

# ON LAMBDA — second tab

The gateway occupies the first tab, so the client needs its own. Open a **new terminal on your Mac** and SSH in again:

```
cd module4-gateway
bash setup/ssh.sh
source .venv/bin/activate
```

`setup/ssh.sh` already lands you in `~/module4-gateway`. Then:

```
python app.py "What is KV cache?"

make test
make bench
```

`make bench` runs all four presets (baseline → route → queue → full) and writes `results.json` and `results.html` **on Lambda**, in `~/module4-gateway`.

---

# PULL RESULTS BACK TO YOUR MAC

The bench output lives on the GPU box. Copy it down before you terminate the instance — the instance is gone for good, and so are the results.

On your **Mac**, from `module4-gateway`:

```
set -a; source .env; set +a
scp -i "$LAMBDA_SSH_KEY" \
  "$LAMBDA:/home/ubuntu/module4-gateway/results.json" \
  "$LAMBDA:/home/ubuntu/module4-gateway/results.html" .
```

Then `open results.html`.

---

# TEAR DOWN

Copy `results.json` / `results.html` down first (above) — terminating the instance destroys them.

```
kill $(cat /tmp/llm-gateway-lab-8001.pid /tmp/llm-gateway-lab-8002.pid)
```

If that leaves anything behind: `pkill -f "vllm serve"`, then confirm with `nvidia-smi`.

Then terminate the instance in the Lambda console.
