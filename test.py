import torch.distributed as dist
import torch
import argparse
import os
import time

def init_env():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"World size: {world_size}, global_rank: {rank}, local_rank: {local_rank} \n")
    return local_rank

def send_rcv():
    src_dst = [4, 5, 6, 7, -1, -1 ,-1, -1]
    rank = dist.get_rank()
    tensor = torch.full((2,2), rank, device=DEVICE, dtype=torch.float32)
    dst = src_dst[rank] 
    if dst > 0:
        print(f"Rank {rank} sending \n")
        dist.send(tensor, dst=dst)
        print(f"Rank {rank} sent \n")
    else: 
        print(f"Rank {rank} receiving \n")
        dist.recv(tensor)
        print(f"Rank {rank} received {tensor} \n")



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
        send_rcv()

        pass
    finally:
        if dist.is_available() and dist.is_initialized():
            print("destroying group \n")
            dist.destroy_process_group()
            time.sleep(1)



if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    main()
