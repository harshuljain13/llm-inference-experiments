# SETUP (once)
cd module3-engine
cp .env.example .env          # fill LAMBDA and HF_TOKEN
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD
source .env
python -c "from smol_vllm import LLMEngine; print('smol_vllm OK')"
bash scripts/lambda_check.sh    # Mac: expect CPU-only; Lambda: Real model ready OK
jupyter notebook class5.ipynb

# EVERY NEW TAB
cd module3-engine && source .venv/bin/activate && export PYTHONPATH=$PWD

# NOTEBOOK — Part A + C + B (CPU) + Part D (GPU on Lambda)
jupyter notebook class5.ipynb

# CLI — all 5 experiments (same as notebook Part C)
smol-vllm-demo

# CLI — CrewAI demo (same as notebook Part B)
python agent_demo.py "What is paged attention?"
python agent_demo.py --single "What is paged attention?"

# QUICK REPL — FakeModel (no GPU)
python -c "
from smol_vllm import LLMEngine
e = LLMEngine(num_gpu_blocks=24, block_size=16, max_batch_size=4)
e.add_request(list(range(10, 30)), max_tokens=8)
for _ in range(20):
    outs = e.step()
    if all(o.finished for o in outs): break
print('done', outs[0].output_tokens)
"

# OPTIONAL — real model (needs GPU + HF token for gated models)
pip install -e smol-vllm[tinyllama-1.1b]
source .env
python -c "
from smol_vllm import LLMEngine
e = LLMEngine(use_real_model=True, max_batch_size=2)
tok = e.model.tokenizer
ids = tok.encode('Hello!', add_special_tokens=False)
for t in e.generate(ids, max_tokens=16):
    print(tok.decode([t]), end='', flush=True)
print()
"

# LAMBDA GPU — see LAMBDA.md
# Mac: cd module3-engine && source .env && bash scripts/sync_to_lambda.sh
# Mac: ssh -i "$LAMBDA_SSH_KEY" -L 8889:127.0.0.1:8888 $LAMBDA
# Lambda: bash scripts/lambda_jupyter.sh
# Mac: open http://127.0.0.1:8889/
# PDF: last notebook cell (once on Lambda: bash scripts/lambda_pdf_deps.sh)
