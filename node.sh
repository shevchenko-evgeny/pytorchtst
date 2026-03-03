#! /usr/bin/env bash
TORCH_DISTRIBUTED_DEBUG=DETAIL
uv run torchrun --nnodes=2 --node_rank=1 --nproc_per_node=$(nvidia-smi -L | wc -l) --master_addr=$MASTER_ADDR test.py
