#!/bin/bash
source /home/sgadmin/services/vllm/venv/bin/activate
exec python -m vllm.entrypoints.openai.api_server \
  --model /home/sgadmin/services/vllm/models/Qwen2.5-Coder-14B-Instruct-AWQ \
  --served-model-name Qwen2.5-Coder-14B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 8192 \
  --quantization awq \
  --gpu-memory-utilization 0.4
