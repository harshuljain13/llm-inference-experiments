#!/bin/bash
set -euo pipefail
cd /app
ray start --head \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --num-cpus="${RAY_NUM_CPUS:-4}" \
  --num-gpus="${RAY_NUM_GPUS:-1}"
exec serve run serve_config.yaml
