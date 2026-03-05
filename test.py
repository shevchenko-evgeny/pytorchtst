import torch.distributed as dist
import torch
import argparse
import os

def init_env():
    print("Init process group")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    print(f"World size: {world_size}, global_rank: {rank}, local_rank: {local_rank}")
    torch.cuda.set_device(local_rank)
    return local_rank



def main():
    args = argparse.ArgumentParser(description="Benchmark GPU connectivities")
    args.add_argument("--tensor_elements_number", type=int, default=100000)
    args.add_argument("--warm_rounds", type=int, default=10)
    args.add_argument("--rounds", type=int, default=100)

    args = args.parse_args()

    global TENSOR_SIZE 
    global WARM_ROUNDS 
    global ROUNDS 

    TENSOR_SIZE = args.tensor_elements_number
    WARM_ROUNDS = args.warm_rounds
    ROUNDS = args.rounds

    local_rank = init_env()

    global DEVICE
    DEVICE = torch.device(f"cuda:{local_rank}")

    dist.destroy_process_group()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    main()
