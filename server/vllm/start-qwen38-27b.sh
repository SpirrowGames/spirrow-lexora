#!/bin/bash
# Qwen3.8-27B (W4A16 / compressed-tensors) — 2026-08-18 に Qwen3-32B-AWQ から載せ替え。
#
# 切り戻し: /etc/systemd/system/vllm-32b.service.d/qwen38.conf を削除して
#           systemctl daemon-reload && systemctl restart vllm-32b (start-32b.sh に戻る)
#
# --served-model-name に旧名 Qwen3-32B を別名として残してある。Lexora のティア名
# ではなくモデル名を直接指定してくる呼び出し元 (7日で15件) を落とさないため。
# --max-num-seqs 8: Qwen3.8 は 64層中 48層が linear_attention (gated delta net) の
# ハイブリッド。GDN の conv state はブロック単位で確保され、既定の cudagraph capture
# (最大512バッチ) が state 行数を超えると causal_conv1d_update の
# `assert num_cache_lines >= batch` で起動時に落ちる。実測トラフィックは 7日で
# light 594 / naysayer 158 なので 8 並列で十分。
# --max-model-len 32768 / util 0.89: KV は full_attention 16層分だけだが、mamba page
# size に合わせて attention block が 784 tokens に膨らむため 1トークンあたり約 258KB
# と重い (実測2点: util 0.72 -> 2.93GiB/11,760 tok、util 0.78 -> 4.81GiB/19,600 tok)。
# 2026-08-18 に Forge (system+user の二重起動) を停止し bge-m3 を CPU に移して
# GPU 3.9GB を空け、util 0.89 = KV 約 8.3GiB まで引き上げた。
# **この設定は GPU を vLLM が実質専有する前提**。Forge を戻すなら util を 0.78
# (ctx 16384) に下げること。65536 以上は物理的に載らない (1M には KV 246GiB 必要)。
# --quantization は指定しない: config.json の quantization_config
# (compressed-tensors / pack-quantized) から自動判別される。awq を明示すると誤る。
# --reasoning-parser qwen3 (2026-09-18): 未指定だと thinking 系ティア (Lexora の
# medium / heavy) の応答本文が "<思考文>\n</think>\n\n<答え>" になり、思考文が
# content に混ざったまま呼び出し元へ届いていた。Qwen3.8 の chat template は
# <think> を prompt 側で開くので、出力には閉じタグだけが残るという壊れ方をする。
# パーサを付けると思考文は reasoning_content に分離され content が本文だけになる。
# パーサはリクエスト毎に chat_template_kwargs 付きで生成される
# (entrypoints/openai/chat_completion/serving.py) ため、light ティアが送る
# enable_thinking=False はそのまま尊重される = light の挙動は変わらない。
# 注意: thinking 有効で max_tokens が </think> より手前で尽きた場合、
# 従来は思考文が content に入っていたが、今後は content が null になる
# (truncated = 本文がまだ無い、というのがパーサの意味論)。
source /home/sgadmin/services/vllm/venv/bin/activate
exec python -m vllm.entrypoints.openai.api_server \
  --model /home/sgadmin/services/vllm/models/Qwen3.8-27B-W4A16-AWQ \
  --served-model-name Qwen3.8-27B Qwen3-32B \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.89 \
  --reasoning-parser qwen3
