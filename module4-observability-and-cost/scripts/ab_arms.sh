# shellcheck shell=bash
# Arms for ./scripts/run_server_ab.sh — each HINT is a vLLM engine profile (see scripts/vllm_engine/).
# Run launchers on the GPU host from the repo root (or copy scripts/vllm_engine/ there).
#
# Docs: https://docs.vllm.ai/en/stable/configuration/engine_args/

AB_CREW_TECHNIQUE="${AB_CREW_TECHNIQUE:-baseline}"

# Sequential A/B: after each arm, set VLLM_SERVER_PROFILE below on the laptop .env and restart gateway.
AB_ARMS_COUNT=3

AB_ARM_1_SERVER_PROFILE="eng_baseline"
AB_ARM_1_TECHNIQUE="ab_baseline"
AB_ARM_1_HINT="$(cat <<'EOF'
bash scripts/vllm_engine/baseline.sh
EOF
)"

AB_ARM_2_SERVER_PROFILE="eng_chunked_prefill"
AB_ARM_2_TECHNIQUE="ab_chunked_prefill"
AB_ARM_2_HINT="$(cat <<'EOF'
bash scripts/vllm_engine/chunked_prefill.sh
EOF
)"

AB_ARM_3_SERVER_PROFILE="eng_prefix_caching"
AB_ARM_3_TECHNIQUE="ab_prefix_caching"
AB_ARM_3_HINT="$(cat <<'EOF'
bash scripts/vllm_engine/prefix_caching.sh
EOF
)"

# Add a 4th arm for speculative decoding (parallel needs VLLM_SERVE_PORT=8003):
# AB_ARMS_COUNT=4
# AB_ARM_4_SERVER_PROFILE="eng_spec_decode"
# AB_ARM_4_TECHNIQUE="ab_spec_decode"
# AB_ARM_4_HINT='export VLLM_SPECULATIVE_CONFIG_JSON='"'"'{"method":"eagle","model":"YOUR/DRAFT","num_speculative_tokens":3}'"'"'
# bash scripts/vllm_engine/speculative_decoding.sh'
