import torch
import pennylane as qml
import time

print("========== Quick GPU Benchmark ==========")
use_cuda = torch.cuda.is_available()
print(f"torch.cuda.is_available(): {use_cuda}")
if use_cuda:
    print("CUDA device:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

n_wires = 5

# --- 测试 lightning.gpu ---
try:
    dev_gpu = qml.device("lightning.gpu" if use_cuda else "default.qubit", wires=n_wires)

    @qml.qnode(dev_gpu, interface="torch", diff_method="adjoint")
    def circuit(x):
        for i in range(n_wires):
            qml.RX(x, wires=i)
            qml.RY(x, wires=i)
        qml.broadcast(qml.CNOT, wires=range(n_wires), pattern="ring")
        return qml.expval(qml.PauliZ(0))

    x = torch.tensor(0.123, dtype=torch.float64, device="cuda" if use_cuda else "cpu")

    # 预热
    for _ in range(3):
        _ = circuit(x)

    # GPU benchmark
    start = time.time()
    for _ in range(50):
        _ = circuit(x)
    if use_cuda:
        torch.cuda.synchronize()
    end = time.time()
    print(f"GPU circuit forward (50 calls): {(end - start)*1000:.2f} ms")

except Exception as e:
    print("⚠️ lightning.gpu test failed:", e)
    dev_gpu = qml.device("default.qubit", wires=n_wires)

# --- 测试 CPU 版本 ---
dev_cpu = qml.device("default.qubit", wires=n_wires)

@qml.qnode(dev_cpu, interface="torch", diff_method="adjoint")
def circuit_cpu(x):
    for i in range(n_wires):
        qml.RX(x, wires=i)
        qml.RY(x, wires=i)
    qml.broadcast(qml.CNOT, wires=range(n_wires), pattern="ring")
    return qml.expval(qml.PauliZ(0))

x_cpu = torch.tensor(0.123, dtype=torch.float64)
for _ in range(3):
    _ = circuit_cpu(x_cpu)

start = time.time()
for _ in range(50):
    _ = circuit_cpu(x_cpu)
end = time.time()
print(f"CPU circuit forward (50 calls): {(end - start)*1000:.2f} ms")
print("========================================\n")
