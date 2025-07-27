import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity


def get_device(index: int = 0) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = F.gelu(x)
        return x


def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    model = MLP(dim, num_layers=num_layers).to(get_device())
    x = torch.randn(batch_size, dim, device=get_device())

    def run():
        for _ in range(num_steps):
            y = model(x).mean()
            y.backward()

    return run


def run_operation1(dim: int, operation: Callable) -> Callable:
    x = torch.randn(dim, device=get_device())
    return lambda: operation(x)


def run_operation2(dim: int, operation: Callable) -> Callable:
    x = torch.randn(dim, dim, device=get_device())
    y = torch.randn(dim, dim, device=get_device())
    return lambda: operation(x, y)


def mean(x: list[float]) -> float:
    return sum(x) / len(x)


def benchmark(
    description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3
):
    for _ in range(num_warmups):
        run()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(num_trials):
        start = time.time()
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

    mean_time = mean(times)
    return mean_time


def benchmarking():
    """
    Measures e2e time of performing different operations.
    For measuring where the time is spend, we need profiling
    """
    dims = (1024, 2048, 4096, 8192, 8192 * 2)
    matmul_results = []
    for dim in dims:
        # @ inspect dim
        result = benchmark(
            f"matmul(dim={dim})", run_operation2(dim=dim, operation=lambda a, b: a @ b)
        )
        matmul_results.append((dim, result))  # @inspect matmul_results

    print(f"matmul results are {matmul_results}")

    dim = 256  # @inspect dim
    num_layers = 4  # @inspect num_layers
    batch_size = 256  # @inspect batch_size
    num_steps = 2  # @inspect num_steps

    mlp_base = benchmark(
        "run_mlp",
        run_mlp(
            dim=dim, num_layers=num_layers, batch_size=batch_size, num_steps=num_steps
        ),
    )
    print(f"mlp base time taken is {mlp_base}")  # @inspect mlp_base


def profile(
    description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False
):
    # Warmup
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Run the code with the profiler
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # Output stack trace for visualization
        with_stack=with_stack,
        # Needed to export stack trace for visualization
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
    ) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Print out table
    table = prof.key_averages().table(
        sort_by="cuda_time_total", max_name_column_width=80, row_limit=10
    )
    # Write stack trace visualization
    if with_stack:
        import os

        # Create the directory if it doesn't exist
        os.makedirs("var", exist_ok=True)
        text_path = f"var/stacks_{description}.txt"
        prof.export_stacks(text_path, "self_cuda_time_total")

    return table


def profiling():
    sleep_func = lambda: time.sleep(50 / 1000)
    sleep_profile = profile("sleep", sleep_func, with_stack=False)
    print("Sleep profile is \n")
    print(sleep_profile)
    print("-" * 30)

    add_function = lambda a, b: a + b
    add_profile = profile("add", run_operation2(dim=2048, operation=add_function))
    print("Add profile is \n")
    print(add_profile)
    print("-" * 30)

    matmul_func = lambda a, b: a @ b
    mult_profile = profile("mult", run_operation2(dim=2048, operation=matmul_func))
    print("Matmul profile is \n")
    print(mult_profile)
    print("-" * 30)

    print("MLP profile is \n")
    mlp_profile = profile(
        "mlp",
        run_mlp(dim=2048, num_layers=64, batch_size=1024, num_steps=2),
        with_stack=True,
    )
    print(mlp_profile)
    print("-" * 30)


def pytorch_gelu(x: torch.Tensor):
    return F.gelu(x, approximate="tanh")


def manual_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))


@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # Read data chunk
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # do the computation
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)
    y = 0.5 * x * (1 + tanh)
    # store data back
    tl.store(y_ptr + offsets, y, mask=mask)


def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()

    y = torch.empty_like(x)
    num_elements = x.numel()
    block_size = 1024
    num_blocks = triton.cdiv(num_elements, block_size)
    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)
    return y


def cuda_kernels():
    x = torch.randn(64, device=get_device())
    y1 = pytorch_gelu(x)
    y2 = manual_gelu(x)
    assert torch.allclose(y1, y2)

    manual_gelu_compiled = torch.compile(manual_gelu)
    y3 = manual_gelu_compiled(x)
    assert torch.allclose(y2, y3)

    y4 = triton_gelu(x)
    assert torch.allclose(y3, y4)

    pytorch_time = benchmark(
        "pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu)
    )  # @inspect pytorch_time
    manual_time = benchmark(
        "manual_gelu", run_operation1(dim=16384, operation=manual_gelu)
    )  # @inspect manual_time
    manual_time_compiled = benchmark(
        "manual_time_compiled",
        run_operation1(dim=16384, operation=manual_gelu_compiled),
    )
    triton_time = benchmark(
        "triton_time",
        run_operation1(dim=16384, operation=triton_gelu),
    )
    print(
        f"pytorch time is {pytorch_time}, manual time is {manual_time}, manual_time_compiled is {manual_time_compiled}, triton time is {triton_time}"
    )


def main():
    # benchmarking()
    # profiling()
    cuda_kernels()


if __name__ == "__main__":
    main()
