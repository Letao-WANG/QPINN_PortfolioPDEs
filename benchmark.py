# test_matmul_speed.py
import torch
import time

def benchmark(device, size=4096, warmup=5, runs=20):
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()  # MPS 需要同步

    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)

    # Warmup
    for _ in range(warmup):
        _ = torch.matmul(A, B)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(runs):
        _ = torch.matmul(A, B)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
    end = time.time()

    avg_time = (end - start) / runs
    gflops = (2 * size**3) / (avg_time * 1e9)  # 2 * n^3 FLOPs
    print(f"{device}: {avg_time*1000:.2f} ms/run, {gflops:.2f} GFLOPS")

if __name__ == "__main__":
    size = 4096
    print(f"Matrix size: {size}x{size}")

    # CPU
    benchmark(torch.device('cpu'), size)

    # CUDA (if available)
    if torch.cuda.is_available():
        benchmark(torch.device('cuda'), size)
    else:
        print("CUDA not available")

    # MPS (Apple Silicon, if available)
    if torch.backends.mps.is_available():
        benchmark(torch.device('mps'), size)
    else:
        print("MPS not available")
#%%
