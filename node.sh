#! /usr/bin/env bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,EN
uv run torchrun --nnodes=2 --node_rank=1 --nproc_per_node=$(nvidia-smi -L | wc -l) --master_addr=$MASTER_ADDR test.py
