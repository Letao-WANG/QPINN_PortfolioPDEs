import torch
import torch.nn as nn
from torch.autograd import grad
import pennylane as qml
import torch_optimizer as optim
import uuid

# Set device to CUDA if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# HJB parameters
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

torch.set_default_dtype(torch.float64)
n_wires = 5
dev = qml.device("default.qubit", wires=n_wires)

# Sampling functions
def sample_collocation(n):
    t = torch.linspace(0.01, 0.99, n, device=device).unsqueeze(1)
    x = torch.linspace(0.01, 0.99, n, device=device).unsqueeze(1)
    return t, x

def sample_boundary2(n):
    t = torch.linspace(0.01, 0.99, n, device=device).unsqueeze(1)
    x = torch.linspace(0.99, 0.99, n, device=device).unsqueeze(1)
    return t, x

def sample_boundary(n):
    t = torch.linspace(0.99, 0.99, n, device=device).unsqueeze(1)
    x = torch.linspace(0.01, 0.99, n, device=device).unsqueeze(1)
    return t, x

def U(x):
    return x**gamma / gamma

def S(z, wire):
    qml.RX(-2 * torch.acos(z), wires=wire)

def U_qsp(phases, z, wire):
    qml.RZ(phases[0], wires=wire)
    for φ in phases[1:]:
        S(z, wire)
        qml.RZ(φ, wires=wire)

def W_lcu(x, t, ph_x1, ph_x2, ph_t1, ph_t2):
    qml.PauliX(wires=1)
    qml.ctrl(U_qsp, control=1)(ph_x1, x, 2)
    qml.PauliX(wires=1)
    qml.ctrl(U_qsp, control=1)(ph_x2, x, 2)
    qml.PauliX(wires=3)
    qml.ctrl(U_qsp, control=3)(ph_t1, t, 4)
    qml.PauliX(wires=3)
    qml.ctrl(U_qsp, control=3)(ph_t2, t, 4)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def f_Wp(t, x, ph_x1, ph_x2, ph_t1, ph_t2):
    for w in range(n_wires):
        qml.Hadamard(wires=w)
    qml.ctrl(W_lcu, control=0)(x, t, ph_x1, ph_x2, ph_t1, ph_t2)
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(wires=0))

# Classical PINN
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        self.activation = nn.Tanh()

    def forward(self, t, x):
        inputs = torch.cat([t, x], dim=1)
        out = inputs
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        return self.layers[-1](out)

# QPINN wrapper
class QPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ph_x1 = nn.Parameter(2 * torch.pi * torch.rand(x_layer1, device=device))
        self.ph_x2 = nn.Parameter(2 * torch.pi * torch.rand(x_layer2, device=device))
        self.ph_t1 = nn.Parameter(2 * torch.pi * torch.rand(t_layer1, device=device))
        self.ph_t2 = nn.Parameter(2 * torch.pi * torch.rand(t_layer2, device=device))

    def forward(self, t, x):
        batch = t.shape[0]
        out = []
        for i in range(batch):
            out.append(f_Wp(t[i,0], x[i,0],
                            self.ph_x1, self.ph_x2,
                            self.ph_t1, self.ph_t2))
        return torch.stack(out).unsqueeze(-1).to(device)

# HJB residual
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

# Instantiate models and move to device
pinn_model = PINN([2,50,50,50,50,50,50,50,1]).to(device)
qpinn_model = QPINN().to(device)
epochs = 1000
cos_epochs = 150

opt_pinn = optim.Lamb(pinn_model.parameters(), lr=1e-2, weight_decay=0, betas=(0.0, 0.0))
opt_qpinn = optim.Lamb(qpinn_model.parameters(), lr=1e-2, weight_decay=0, betas=(0.0, 0.0))
scheduler_pinn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pinn, T_max=cos_epochs, eta_min=1e-3)
scheduler_qpinn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_qpinn, T_max=cos_epochs, eta_min=1e-3)
mse = nn.MSELoss()

# Storage
losses_pinn, losses_qpinn = [], []

for epoch in range(epochs):
    # Sample points
    tc, xc = sample_collocation(50)
    tb1, xb1 = sample_boundary(50)
    tb2, xb2 = sample_boundary2(50)

    # PINN losses
    res_p = hjb_residual(pinn_model, tc, xc)
    lpde_p = mse(res_p, torch.zeros_like(res_p, device=device))
    lbc1_p = mse(pinn_model(tb1, xb1), U(xb1))
    lbc2_p = mse(pinn_model(tb2, xb2), torch.exp(-k * (T - tb2)) / gamma)
    loss_p = lpde_p + lbc1_p + 5 * lbc2_p
    opt_pinn.zero_grad()
    loss_p.backward()
    opt_pinn.step()
    if epoch <= cos_epochs:
        scheduler_pinn.step()

    # QPINN losses
    res_q = hjb_residual(qpinn_model, tc, xc)
    lpde_q = mse(res_q, torch.zeros_like(res_q, device=device))
    lbc1_q = mse(qpinn_model(tb1, xb1), U(xb1))
    lbc2_q = mse(qpinn_model(tb2, xb2), torch.exp(-k * (T - tb2)) / gamma)
    loss_q = lpde_q + lbc1_q + 5 * lbc2_q
    opt_qpinn.zero_grad()
    loss_q.backward()
    opt_qpinn.step()
    if epoch <= cos_epochs:
        scheduler_qpinn.step()

    losses_pinn.append(loss_p.item())
    losses_qpinn.append(loss_q.item())
    if epoch % 5 == 0:
        print(f"Epoch {epoch}, loss_PINN = {loss_p.item():.3e}, loss_QPINN = {loss_q.item():.3e}")
        print(f"Epoch {epoch}, loss = {loss_q.item():.3e}, loss_pde = {lpde_q.item():.3e}, lbc1_q = {lbc1_q.item():.3e}, lbc2_q = {lbc2_q.item():.3e}")

    if epoch == cos_epochs:
        opt_pinn = optim.Lamb(pinn_model.parameters(), lr=1e-3, weight_decay=0, betas=(0.0, 0.0))
        opt_qpinn = optim.Lamb(qpinn_model.parameters(), lr=1e-3, weight_decay=0, betas=(0.0, 0.0))

    if epoch == 300:
        opt_pinn = optim.Lamb(pinn_model.parameters(), lr=1e-4, weight_decay=0, betas=(0.0, 0.0))
        opt_qpinn = optim.Lamb(qpinn_model.parameters(), lr=1e-4, weight_decay=0, betas=(0.0, 0.0))