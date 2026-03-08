#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export BASE_MODEL=unsloth/Llama-3.2-3B-Instruct-bnb-4bit
export OUTPUT_DIR=/workspace/outputs/netsuite-sql-lora
export TRAIN_FILE=/workspace/data/train.jsonl
export VAL_FILE=/workspace/data/val.jsonl
export MAX_SEQ_LENGTH=2048

python train.py
