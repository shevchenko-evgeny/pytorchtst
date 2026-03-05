import torch.distributed as dist
import torch
import argparse

def init_env():
    print("Init process group")
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    print(f"World size: {world_size}, my rank: {rank}")
    return rank



def main():
    args = argparse.ArgumentParser(description="Benchmark GPU connectivities")
    args.add_argument("--tensor_elements_number", type=int, default=100000)
    args.add_argument("--warm_rounds", type=int, default=10)
    args.add_argument("--rounds", type=int, default=100)

    args.parse_args()

    global TENSOR_SIZE 
    global WARM_ROUNDS 
    global ROUNDS 

    TENSOR_SIZE = args.tensor_elements_number
    WARM_ROUNDS = args.warm_rounds
    ROUNDS = args.rounds

    rank = init_env()
    dist.destroy_process_group()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    main()
