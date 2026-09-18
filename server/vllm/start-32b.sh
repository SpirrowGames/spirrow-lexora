#!/bin/bash
source /home/sgadmin/services/vllm/venv/bin/activate
exec python -m vllm.entrypoints.openai.api_server \
  --model /home/sgadmin/services/vllm/models/Qwen3-32B-AWQ \
  --served-model-name Qwen3-32B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --quantization awq \
  --gpu-memory-utilization 0.7
