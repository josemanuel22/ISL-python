import numpy as np
from tqdm import tqdm
import scipy.stats as stats
from scipy.special import expit

def _sigmoid(y_hat, y):
    """Calculate the sigmoid function centered at y using PyTorch."""
    return torch.sigmoid((y - y_hat) * 10.0)

def psi_m(y, m):
    """Calculate the bump function centered at m using PyTorch."""
    stddev = 0.1
    return torch.exp(-0.5 * ((y - m) / stddev) ** 2)

def phi(y_k, y_n):
    """Calculate the sum of sigmoid functions using PyTorch."""
    return torch.sum(_sigmoid(y_k, y_n))

def gamma(y_k, y_n, m):
    """Calculate the contribution to the m-th bin of the histogram using PyTorch."""
    e_m = torch.tensor([1.0 if j == m else 0.0 for j in range(len(y_k) + 1)])
    return e_m * psi_m(phi(y_k, y_n), m)

def generate_a_k(y_hat, y):
    """Calculate the values of the real observation y in each of the components of the approximate histogram using PyTorch."""
    K = len(y_hat)
    return torch.sum(torch.stack([gamma(y_hat, y, k) for k in range(K)]), dim=0)

def scalar_diff(q):
    """Scalar difference between the vector representing our surrogate histogram and the uniform distribution vector using PyTorch."""
    K = len(q)
    return torch.sum((q - 1/(K+1)) ** 2)

def jensen_shannon_divergence(p, q):
    """Calculate the Jensen-Shannon divergence using PyTorch."""
    epsilon = 1e-3  # to avoid log(0)
    p_safe, q_safe = p + epsilon, q + epsilon
    m = 0.5 * (p_safe + q_safe)
    return 0.5 * (F.kl_div(m.log(), p_safe, reduction='batchmean') + F.kl_div(m.log(), q_safe, reduction='batchmean'))

def jensen_shannon_grad(q):
    """Jensen-Shannon difference between q tensor and uniform distribution tensor using PyTorch."""
    K = len(q)
    uniform_dist = torch.full((K,), fill_value=1/K)
    return jensen_shannon_divergence(q, uniform_dist)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Hyperparameters
ISLhparams = {
    'eta': 1e-3,  # learning rate
    'epochs': 100,
    'samples': 1000
    # add other hyperparameters here
}

# Custom loss function
def invariant_statistical_loss(model, data_loader, hparams):
    optimizer = optim.Adam(model.parameters(), lr=hparams['eta'])
    losses = []
    for _ in tqdm(range(hparams['epochs'])):
        for data in data_loader:
            optimizer.zero_grad()
            noise = torch.normal(0.0, 1.0, size=(hparams['samples'], 1))
            y_k = model(noise)
            a_k = generate_a_k(y_k, data)
            loss_value = scalar_diff(a_k / torch.sum(a_k))
            loss = torch.tensor(loss_value, requires_grad=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return losses
