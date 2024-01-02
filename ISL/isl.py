import numpy as np
import scipy.stats as stats
from scipy.special import expit

def _sigmoid(y_hat, y):
    """Calculate the sigmoid function centered at y."""
    return expit((y - y_hat) * 10.0)

def psi_m(y, m):
    """Calculate the bump function centered at m."""
    stddev = 0.1
    return np.exp(-0.5 * ((y - m) / stddev) ** 2)

def phi(y_k, y_n):
    """Calculate the sum of sigmoid functions."""
    return np.sum(_sigmoid(y_k, y_n))

def gamma(y_k, y_n, m):
    """Calculate the contribution to the m-th bin of the histogram."""
    e_m = np.array([1.0 if j == m else 0.0 for j in range(len(y_k) + 1)])
    return e_m * psi_m(phi(y_k, y_n), m)

def generate_a_k(y_hat, y):
    """Calculate the values of the real observation y in each of the components of the approximate histogram."""
    K = len(y_hat)
    return np.sum([gamma(y_hat, y, k) for k in range(K)], axis=0)

def scalar_diff(q):
    """Scalar difference between the vector representing our surrogate histogram and the uniform distribution vector."""
    K = len(q)
    return np.sum((q - 1/(K+1)) ** 2)

def jensen_shannon_divergence(p, q):
    """Calculate the Jensen-Shannon divergence."""
    epsilon = 1e-3  # to avoid log(0)
    p_safe, q_safe = p + epsilon, q + epsilon
    m = 0.5 * (p_safe + q_safe)
    return 0.5 * (stats.entropy(p_safe, m) + stats.entropy(q_safe, m))

def jensen_shannon_grad(q):
    """Jensen-Shannon difference between q vector and uniform distribution vector."""
    uniform_dist = np.full(len(q), fill_value=1/len(q))
    return jensen_shannon_divergence(q, uniform_dist)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Hyperparameters
ISLhparams = {
    'eta': 1e-3,  # learning rate
    'epochs': 100,
    # add other hyperparameters here
}

# Custom loss function
def invariant_statistical_loss(model, data_loader, hparams):
    optimizer = optim.Adam(model.parameters(), lr=hparams['eta'])
    losses = []
    for epoch in range(hparams['epochs']):
        for data in data_loader:
            optimizer.zero_grad()
            # Assuming data is a tuple of (input, target)
            input, target = data
            y_k = model(input)
            a_k = generate_a_k(y_k.detach().numpy(), target.numpy())
            loss_value = scalar_diff(a_k / np.sum(a_k))
            loss = torch.tensor(loss_value, requires_grad=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return losses
