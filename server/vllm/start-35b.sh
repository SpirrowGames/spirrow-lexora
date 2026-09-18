#!/bin/bash
source /home/sgadmin/services/vllm/venv/bin/activate
exec python -m vllm.entrypoints.openai.api_server \
  --model /home/sgadmin/services/vllm/models/Qwen3.5-35B-A3B-GPTQ-Int4 \
  --served-model-name Qwen3.5-35B-A3B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --quantization gptq_marlin \
  --dtype float16 \
  --gpu-memory-utilization 0.83 \
  --enforce-eager
