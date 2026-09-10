# Class 5 on Lambda GPU (remote Jupyter)

Run the full notebook on a Lambda instance. Your Mac is the browser; the GPU instance runs Jupyter + TinyLlama for **Exp 5**.

Launch the instance in the [Lambda Cloud console](https://cloud.lambda.ai), SSH in, then follow the steps below. Terminate the instance in the console when you're done.

**Env (once on your Mac):**

```bash
cd module3-engine
cp .env.example .env
# edit .env — use YOUR values (never commit .env):
#   export LAMBDA=ubuntu@YOUR_LAMBDA_IP   ← IP from Lambda console, NO "ssh" prefix
#   export LAMBDA_SSH_KEY=$HOME/.ssh/YOUR_LAMBDA_KEY   ← private key registered at launch
#   export HF_TOKEN=hf_xxxx               ← no space after =
source .env
echo $LAMBDA   # should print ubuntu@129.x.x.x
ssh -i "$LAMBDA_SSH_KEY" -o BatchMode=yes "$LAMBDA" echo "SSH OK"   # must print OK
```

| Part | CPU OK? | GPU needed? |
|------|---------|-------------|
| A — Engine internals (FakeModel) | yes | no |
| C — Exps 1–4 + checkpoints | yes | no |
| C — Exp 5 (Fake vs Real) | FakeModel yes | **Real model yes** |
| B — CrewAI → engine | yes | no (FakeModel) |

---

## 1. Copy module3-engine to the instance

**From your Mac** (prompt must be `jarvis@Mac`, not `ubuntu@...`):

```bash
cd module3-engine
source .env
bash scripts/sync_to_lambda.sh
```

Or manually:

```bash
cd module3-engine
source .env
rsync -avz --progress -e "ssh -i $LAMBDA_SSH_KEY" \
  --exclude '.venv' --exclude '.ipynb_checkpoints' \
  --exclude '__pycache__' --exclude 'logs' --exclude '.env' \
  ./ $LAMBDA:~/module3-engine/
```

---

## 2. One-time setup on the instance

SSH in:

```bash
ssh -i "$LAMBDA_SSH_KEY" $LAMBDA
cd ~/module3-engine
bash scripts/lambda_setup.sh
```

This creates `.venv`, installs smol-vllm + CrewAI + Jupyter + TinyLlama deps, and prints `cuda available True` if the GPU is visible. If you copied `.env` to the instance, `HF_TOKEN` is picked up automatically for Exp 5.

**Verify GPU:**

```bash
bash scripts/lambda_check.sh   # expect: Real model ready OK
```

---

## 3. Start Jupyter on the instance

**On the Lambda instance** (SSH session):

```bash
cd ~/module3-engine
bash scripts/lambda_jupyter.sh
```

Leave this running. **No token** — Jupyter binds `127.0.0.1` only; the SSH tunnel is the auth.

If login still appears, an old server is running — on Lambda run `pkill -f jupyter` then restart this script.

---

## 4. Tunnel from your Mac

**New terminal on your Mac** (keep the SSH session above open):

```bash
ssh -i "$LAMBDA_SSH_KEY" -L 8888:127.0.0.1:8888 $LAMBDA
```

Open in your browser:

**http://127.0.0.1:8888/**

(no token — if that port is busy on your Mac, tunnel `8889:127.0.0.1:8888` and open `http://127.0.0.1:8889/`)

Run `class5.ipynb` top to bottom — Parts A–C + B on FakeModel; **Part D** is the GPU track (TinyLlama).

---

## 5. Quick verify (optional, on instance)

```bash
cd ~/module3-engine && source .venv/bin/activate && export PYTHONPATH=$PWD

# CPU path — all of Part A/C (FakeModel)
python -c "from smol_vllm import LLMEngine; print('OK')"

# GPU path — Exp 5 real model smoke test
python -c "
from smol_vllm import LLMEngine
import torch
print('cuda', torch.cuda.is_available())
e = LLMEngine(use_real_model=True, max_batch_size=2)
tok = e.model.tokenizer
ids = tok.encode('Hello from Lambda GPU!', add_special_tokens=False)
for t in e.generate(ids, max_tokens=12):
    print(tok.decode([t]), end='', flush=True)
print()
"
```

---

## 6. Teaching flow on Lambda

| Order | Notebook section | Notes |
|-------|------------------|-------|
| 1 | Setup cell | Should print `smol_vllm imported OK` |
| 2 | Part A | FakeModel — scheduler, prefill/decode |
| 3 | Part C | Exps 1–4 on CPU; **Exp 5** shows real GPU timings |
| 4 | Part B | CrewAI + FakeModel (CPU); or swap to real model later |

**Say on Exp 5:** "FakeModel taught the control plane. Now the same scheduler runs TinyLlama on CUDA — watch `[metrics]` prefill_ms jump."

---

## 7. CLI alternatives (no browser)

On the instance:

```bash
cd ~/module3-engine && source .venv/bin/activate && export PYTHONPATH=$PWD

smol-vllm-demo                              # all 5 experiments
python agent_demo.py "What is KV cache?"   # CrewAI Part B
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Permission denied (publickey)` | Wrong SSH key — set `LAMBDA_SSH_KEY` to the key registered in Lambda console at instance launch |
| `Could not resolve hostname your_lambda_ip` | `.env` still has placeholder — set `export LAMBDA=ubuntu@REAL_IP` from Lambda console |
| `Could not resolve hostname ssh` | Remove `ssh` from `LAMBDA` — use `ubuntu@IP` not `ssh ubuntu@IP` |
| rsync: `./module3-engine/` not found | You're already in `module3-engine/` — use `./` as source |
| Login page / invalid token | Old Jupyter still running — on Lambda: `pkill -f jupyter`, restart `lambda_jupyter.sh`. Or Mac port 8888 in use — tunnel via `-L 8889:127.0.0.1:8888` |
| Browser can't connect | Check SSH `-L` tunnel is running; Jupyter must bind `127.0.0.1:8888` |
| `cuda available False` | Run `nvidia-smi`; reinstall torch with CUDA wheels if needed |
| Exp 5 OOM | Lower `max_batch_size=1`, raise `num_gpu_blocks`, or use Qwen2-0.5B |
| `cannot import LLMEngine` | Vendor folder must be `smol-vllm/` not `smol_vllm/`; re-run `lambda_setup.sh` |
| CrewAI event-loop error in notebook | Use `await kickoff_crew(...)` in Part B; restart kernel after pull |
| Slow rsync | Exclude `.venv` — build venv on the instance |
| Ran rsync/scp on Lambda by mistake | Those commands run on **Mac** only — prompt should not be `ubuntu@...` |
| Code works on Mac but fails on Lambda | On Mac: `bash scripts/sync_to_lambda.sh`. On Lambda: restart Jupyter kernel (Kernel → Restart) |
| `playwright: command not found` | `source .venv/bin/activate` first; use `sudo $(which python) -m playwright install-deps chromium` |
| PDF export fails | On Lambda: `bash scripts/lambda_pdf_deps.sh` |
| `DynamicCache` / `from_legacy_cache` errors | Re-sync from Mac (`bash scripts/sync_to_lambda.sh`) — fixes are in `smol-vllm/smol_vllm/causal_model.py` |
| `gpu_mem=0/0` in Parts A–C | Expected — FakeModel is CPU-only; GPU memory shows in Part D |

---

## Tear down

Stop Jupyter (Ctrl+C in the instance terminal). Stop the SSH tunnel (Ctrl+C on Mac).

Lambda: terminate the instance from the Lambda Cloud console when done to avoid charges.

---

## Before sharing this repo (instructors)

- **Never commit `.env`** — it holds your Lambda IP, SSH key path, and HF token. It is in `.gitignore`.
- Each student copies `.env.example` → `.env` and fills in **their own** Lambda IP and keys.
- Run `bash scripts/check_share_safe.sh` before your first push to catch accidental leaks.
- Do not commit `class5_with_outputs.pdf` if it was run on your instance (hostname may appear in outputs).
