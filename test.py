import torch.distributed as dist
import torch
import argparse
import os

def init_env():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"World size: {world_size}, global_rank: {rank}, local_rank: {local_rank} \n")
    return local_rank

def topology():
    devices_num = torch.cuda.device_count()
    if dist.get_rank() == 0: 
        print(f"I see {devices_num} devices interconnected")



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

    try:
        dist.barrier(device_ids=[local_rank])
        topology()
        r = dist.get_rank()
        print(f"rank={r} local_rank={local_rank} BEFORE barrier \n", flush=True)
        dist.barrier(device_ids=[local_rank])
        print(f"rank={r} AFTER barrier \n", flush=True)

        pass
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()



if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    main()
