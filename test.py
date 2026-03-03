import torch.distributed as dist
import torch
import os

def init_env():
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"Local rank: {local_rank}")
    torch.cuda.set_device(local_rank)
    print("Init process group")
    dist.init_process_group(backend="nccl")
    print("Init process group DONE")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"World size: {world_size}, my rank: {rank}")



def main():
    init_env()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    main()
