import torch
import torch.nn as nn
from torch.autograd import grad
import pennylane as qml
import torch_optimizer as optim

# =======================
# ⚙️ 设备设置
# =======================
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"torch.cuda.is_available(): {use_cuda}")
if use_cuda:
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

torch.set_default_dtype(torch.float64)

# 尝试加载 lightning.gpu
n_wires = 5
try:
    dev = qml.device("lightning.gpu" if use_cuda else "default.qubit", wires=n_wires)
    print(f"PennyLane device: {dev.short_name}")
except Exception as e:
    print("⚠️ Failed to use lightning.gpu, fallback to default.qubit:", e)
    dev = qml.device("default.qubit", wires=n_wires)

# =======================
# 🎯 HJB 参数
# =======================
r = 0.05
T = 1.0
gamma = 2.0
mu = 0.1
sigma = 0.1
hat_alpha = (1/(1 - gamma)) * ((mu - r) / sigma**2)
k = -gamma * r + 0.5 * (gamma/(gamma-1)) * ((mu - r)/sigma)**2
x_layer1 = 3
x_layer2 = 4
t_layer1 = 3
t_layer2 = 4

# =======================
# 📊 采样函数
# =======================
def sample_collocation(n):
    t = torch.linspace(0.01, 0.99, n, device=device).unsqueeze(1)
    x = torch.linspace(0.01, 0.99, n, device=device).unsqueeze(1)
    return t, x

def sample_boundary2(n):
    t = torch.linspace(0.01, 0.99, n, device=device).unsqueeze(1)
    x = torch.full((n, 1), 0.99, device=device)
    return t, x

def sample_boundary(n):
    t = torch.full((n, 1), 0.99, device=device)
    x = torch.linspace(0.01, 0.99, n, device=device).unsqueeze(1)
    return t, x

def U(x):
    return x**gamma / gamma

# =======================
# ⚛️ 量子部分
# =======================
def S(z, wire):
    qml.RX(-2 * torch.acos(z), wires=wire)

def U_qsp(phases, z, wire):
    qml.RZ(phases[0], wires=wire)
    for phi in phases[1:]:
        S(z, wire)
        qml.RZ(phi, wires=wire)

def W_lcu(x, t, ph_x1, ph_x2, ph_t1, ph_t2):
    qml.PauliX(wires=1)
    qml.ctrl(U_qsp, control=1)(ph_x1, x, 2)
    qml.PauliX(wires=1)
    qml.ctrl(U_qsp, control=1)(ph_x2, x, 2)
    qml.PauliX(wires=3)
    qml.ctrl(U_qsp, control=3)(ph_t1, t, 4)
    qml.PauliX(wires=3)
    qml.ctrl(U_qsp, control=3)(ph_t2, t, 4)

# lightning.gpu 支持 adjoint 微分
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def f_Wp(t, x, ph_x1, ph_x2, ph_t1, ph_t2):
    for w in range(n_wires):
        qml.Hadamard(wires=w)
    qml.ctrl(W_lcu, control=0)(x, t, ph_x1, ph_x2, ph_t1, ph_t2)
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(wires=0))

# =======================
# 🤖 Classical PINN
# =======================
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        self.activation = nn.Tanh()

    def forward(self, t, x):
        inputs = torch.cat([t, x], dim=1)
        out = inputs
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        return self.layers[-1](out)

# =======================
# ⚛️ QPINN
# =======================
class QPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ph_x1 = nn.Parameter(2 * torch.pi * torch.rand(x_layer1, device=device))
        self.ph_x2 = nn.Parameter(2 * torch.pi * torch.rand(x_layer2, device=device))
        self.ph_t1 = nn.Parameter(2 * torch.pi * torch.rand(t_layer1, device=device))
        self.ph_t2 = nn.Parameter(2 * torch.pi * torch.rand(t_layer2, device=device))

    def forward(self, t, x):
        batch = t.shape[0]
        outs = []
        for i in range(batch):
            outs.append(f_Wp(t[i,0], x[i,0], self.ph_x1, self.ph_x2, self.ph_t1, self.ph_t2))
        return torch.stack(outs).unsqueeze(-1)

# =======================
# 🧮 HJB 残差
# =======================
def hjb_residual(model, t, x):
    t = t.clone().requires_grad_(True)
    x = x.clone().requires_grad_(True)
    v = model(t, x)
    v_t = grad(v, t, torch.ones_like(v), create_graph=True)[0]
    v_x = grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_xx = grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    drift = x * (hat_alpha * (mu - r) + r) * v_x
    diffusion = 0.5 * hat_alpha**2 * sigma**2 * x**2 * v_xx
    return v_t + drift + diffusion

# =======================
# 🚀 初始化与优化器
# =======================
pinn_model = PINN([2,50,50,50,50,50,50,50,1]).to(device)
qpinn_model = QPINN().to(device)

epochs = 1000
cos_epochs = 150
mse = nn.MSELoss()

opt_pinn  = optim.Lamb(pinn_model.parameters(),  lr=1e-2, weight_decay=0, betas=(0.0, 0.0))
opt_qpinn = optim.Lamb(qpinn_model.parameters(), lr=1e-2, weight_decay=0, betas=(0.0, 0.0))
scheduler_pinn  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pinn,  T_max=cos_epochs, eta_min=1e-3)
scheduler_qpinn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_qpinn, T_max=cos_epochs, eta_min=1e-3)

# ======================= #
# 📁 保存设置
# ======================= #
import os
import csv
from pathlib import Path

save_dir = Path("results")
save_dir.mkdir(exist_ok=True)

# CSV headers
pinn_csv = save_dir / "pinn_losses.csv"
qpinn_csv = save_dir / "qpinn_losses.csv"

# Write headers once
for csv_path, header in [
    (pinn_csv,  ["epoch", "total", "pde", "bc1", "bc2"]),
    (qpinn_csv, ["epoch", "total", "pde", "bc1", "bc2"])
]:
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

# How often to checkpoint the full model
save_every = 1

# Track best loss for final save
best_pinn_loss = float("inf")
best_qpinn_loss = float("inf")
best_pinn_epoch = -1
best_qpinn_epoch = -1

# ======================= #
# 🔁 训练循环 (修改版)
# ======================= #
losses_pinn, losses_qpinn = [], []

for epoch in range(epochs):
    tc, xc = sample_collocation(50)
    tb1, xb1 = sample_boundary(50)
    tb2, xb2 = sample_boundary2(50)

    # ------------------- #
    #   PINN
    # ------------------- #
    res_p = hjb_residual(pinn_model, tc, xc)
    lpde_p = mse(res_p, torch.zeros_like(res_p))
    lbc1_p = mse(pinn_model(tb1, xb1), U(xb1))
    lbc2_p = mse(pinn_model(tb2, xb2), torch.exp(-k * (T - tb2)) / gamma)
    loss_p = lpde_p + lbc1_p + 5 * lbc2_p

    opt_pinn.zero_grad()
    loss_p.backward()
    opt_pinn.step()
    if epoch <= cos_epochs:
        scheduler_pinn.step()

    # ------------------- #
    #   QPINN
    # ------------------- #
    res_q = hjb_residual(qpinn_model, tc, xc)
    lpde_q = mse(res_q, torch.zeros_like(res_q))
    lbc1_q = mse(qpinn_model(tb1, xb1), U(xb1))
    lbc2_q = mse(qpinn_model(tb2, xb2), torch.exp(-k * (T - tb2)) / gamma)
    loss_q = lpde_q + lbc1_q + 5 * lbc2_q

    opt_qpinn.zero_grad()
    loss_q.backward()
    opt_qpinn.step()
    if epoch <= cos_epochs:
        scheduler_qpinn.step()

    # ------------------- #
    #   记录 & 保存
    # ------------------- #
    losses_pinn.append(loss_p.item())
    losses_qpinn.append(loss_q.item())

    # CSV row
    with open(pinn_csv, "a", newline="") as f:
        csv.writer(f).writerow([epoch,
                                f"{loss_p.item():.6e}",
                                f"{lpde_p.item():.6e}",
                                f"{lbc1_p.item():.6e}",
                                f"{lbc2_p.item():.6e}"])
    with open(qpinn_csv, "a", newline="") as f:
        csv.writer(f).writerow([epoch,
                                f"{loss_q.item():.6e}",
                                f"{lpde_q.item():.6e}",
                                f"{lbc1_q.item():.6e}",
                                f"{lbc2_q.item():.6e}"])

    # Periodic checkpoint
    if (epoch + 1) % save_every == 0:
        torch.save(pinn_model.state_dict(),
                   save_dir / f"pinn_epoch{epoch+1:04d}.pt")
        torch.save(qpinn_model.state_dict(),
                   save_dir / f"qpinn_epoch{epoch+1:04d}.pt")

    # ------------------- #
    #   日志输出
    # ------------------- #
    if epoch % 5 == 0:
        print(f"[Epoch {epoch:04d}] "
              f"PINN: {loss_p.item():.3e} | QPINN: {loss_q.item():.3e}")
        print(f"  details → PDE={lpde_q.item():.3e}, "
              f"BC1={lbc1_q.item():.3e}, BC2={lbc2_q.item():.3e}")

    # ------------------- #
    #   学习率切换
    # ------------------- #
    if epoch == cos_epochs:
        opt_pinn = optim.Lamb(pinn_model.parameters(),
                              lr=1e-3, weight_decay=0, betas=(0.0, 0.0))
        opt_qpinn = optim.Lamb(qpinn_model.parameters(),
                               lr=1e-3, weight_decay=0, betas=(0.0, 0.0))
    if epoch == 300:
        opt_pinn = optim.Lamb(pinn_model.parameters(),
                              lr=1e-4, weight_decay=0, betas=(0.0, 0.0))
        opt_qpinn = optim.Lamb(qpinn_model.parameters(),
                               lr=1e-4, weight_decay=0, betas=(0.0, 0.0))

# ----------------------- #
# 训练结束后打印最佳模型信息
# ----------------------- #
print("\n=== Training finished ===")
print(f"Best PINN  – epoch {best_pinn_epoch:04d}, loss {best_pinn_loss:.3e} → saved as 'pinn_best.pt'")
print(f"Best QPINN – epoch {best_qpinn_epoch:04d}, loss {best_qpinn_loss:.3e} → saved as 'qpinn_best.pt'")