import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from lec08_util import get_device, get_init_params, int_divide, spawn, summarize_tensor


def setup(rank: int, world_size: int):
    # Specify where master lives (rank 0), used to coordinate (actual data goes through NCCL)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"

    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def collective_communication_main(rank: int, world_size: int):
    setup(rank, world_size)

    # All reduce
    dist.barrier()
    tensor = torch.tensor([0.0, 1.0, 2.0, 3.0], device=get_device(rank)) + rank
    print(f"Rank {rank} [before all-reduce]: {tensor}", flush=True)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
    print(f"Rank {rank} [after all-reduce]: {tensor}", flush=True)

    # reduce_scatter
    dist.barrier()
    input = (
        torch.arange(world_size, dtype=torch.float32, device=get_device(rank)) + rank
    )  # Input
    output = torch.empty(1, device=get_device(rank))  # Allocate output

    print(
        f"Rank {rank} [before reduce-scatter]: input = {input}, output = {output}",
        flush=True,
    )
    dist.reduce_scatter_tensor(
        output=output, input=input, op=dist.ReduceOp.SUM, async_op=False
    )
    print(
        f"Rank {rank} [after reduce-scatter]: input = {input}, output = {output}",
        flush=True,
    )

    # All gather
    input = output
    output = torch.empty(world_size, device=get_device(rank))
    print(
        f"Rank {rank} [before all-gather]: input = {input}, output = {output}",
        flush=True,
    )
    dist.all_gather_into_tensor(output, input, async_op=False)
    print(
        f"Rank {rank} [after all-gather]: input = {input}, output = {output}",
        flush=True,
    )

    cleanup()


def generate_sample_data():
    batch_size = 128
    num_dims = 1024
    data = torch.randn(batch_size, num_dims)
    return data


def data_parallelism_main(
    rank: int, world_size: int, data: torch.tensor, num_layers: int, num_steps: int
):
    setup(rank, world_size)
    batch_size, num_dims = data.shape
    local_batch_size = int_divide(batch_size, world_size)
    start_index = rank * local_batch_size
    end_index = start_index + local_batch_size
    data = data[start_index:end_index].to(get_device(rank))

    params = [get_init_params(num_dims, num_dims, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    for step in range(num_steps):
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()
        loss.backward()

        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        optimizer.step()
        print(
            f"[data_parallelism] Rank {rank}: step = {step}, loss = {loss.item()}, params = {[summarize_tensor(params[i]) for i in range(num_layers)]}",
            flush=True,
        )
        optimizer.zero_grad()

    cleanup()


def data_parallelism():
    data = generate_sample_data()
    spawn(data_parallelism_main, world_size=4, data=data, num_layers=4, num_steps=2)


def tensor_parallelism_main(
    rank: int, world_size: int, data: torch.Tensor, num_layers: int
):
    setup(rank, world_size)
    batch_size, num_dim = data.shape
    data = data.to(get_device(rank))
    local_num_dim = int_divide(num_dim, world_size)
    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]
    # optimizer = torch.optim.AdamW(params, lr=1e-3)

    x = data
    for i in range(num_layers):
        x = x @ params[i]  # (batch_size, local_num_dims)
        x = F.gelu(x)

        print(f"x shape before all gather: {x.shape}, rank: {rank}")
        activations = [
            torch.empty(batch_size, local_num_dim, device=get_device(rank))
            for _ in range(world_size)
        ]
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)

        x = torch.cat(activations, dim=1)
        print(f"x shape after all gather: {x.shape}, rank: {rank}")

    cleanup()


def tensor_parallelism():
    data = generate_sample_data()
    spawn(tensor_parallelism_main, world_size=4, data=data, num_layers=1)


def cleanup():
    dist.destroy_process_group()


def main():
    # spawn(collective_communication_main, 4)
    # data_parallelism()
    tensor_parallelism()
    # not implementing pipeline parallelism for now.


if __name__ == "__main__":
    main()
