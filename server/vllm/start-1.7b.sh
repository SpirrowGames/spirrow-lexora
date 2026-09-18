#!/bin/bash
source /home/sgadmin/services/vllm/venv/bin/activate
exec python -m vllm.entrypoints.openai.api_server \
  --model /home/sgadmin/services/vllm/models/Qwen3-1.7B \
  --served-model-name Qwen3-1.7B \
  --host 0.0.0.0 \
  --port 8001 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.12
