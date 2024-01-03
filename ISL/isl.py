from tqdm import tqdm
import scipy.stats as stats
import torch.nn.functional as F
from scipy.stats import chisquare
import torch
import torch.nn as nn
import torch.optim as optim

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
    epsilon = 1e-10  # Small constant to avoid log(0)
    p_safe = p + epsilon
    q_safe = q + epsilon

    # Ensure the distributions are normalized
    p_safe /= p_safe.sum()
    q_safe /= q_safe.sum()

    m = 0.5 * (p_safe + q_safe)

    # Calculate the KL divergences and ensure to take the log of p_safe and q_safe
    kl_div_p = F.kl_div(m.log(), p_safe, reduction='batchmean')
    kl_div_q = F.kl_div(m.log(), q_safe, reduction='batchmean')

    # Jensen-Shannon divergence is the average of these KL divergences
    return 0.5 * (kl_div_p + kl_div_q)

def jensen_shannon_grad(q):
    """Jensen-Shannon difference between q tensor and uniform distribution tensor using PyTorch."""
    K = len(q)
    uniform_dist = torch.full((K,), fill_value=1/K)
    return jensen_shannon_divergence(q, uniform_dist)



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
    for data in tqdm(data_loader):
        a_k = torch.zeros(int(hparams['K']) + 1)
        for y in data:
            optimizer.zero_grad()
            x_k = torch.normal(0.0, 1.0, size=(hparams['K'], 1))
            y_k = model(x_k)
            a_k += generate_a_k(y_k, y)
        loss = scalar_diff(a_k / torch.sum(a_k))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses



def get_window_of_Ak(transform, model, data, K):
    """
    Generate a window of the rv's Ak for a given model and target function.
    """
    window = [torch.sum(model(transform(K)).T < d) for d in data]
    return [torch.sum(window == i).item() for i in range(K + 1)]

def convergence_to_uniform(ak):
    """
    Test the convergence of the distribution of the window of the rv's Ak to a uniform
    distribution. It is implemented using a Chi-Square test.
    """
    expected = torch.full((len(ak),), 1 / len(ak))
    return chisquare(ak, expected.numpy())[1] > 0.05

def get_better_K(nn_model, data, min_K, hparams):
    """
    Find a better K value based on the convergence to uniform distribution.
    """
    K = hparams['max_k']
    for k in range(min_K, hparams['max_k'] + 1):
        if not convergence_to_uniform(get_window_of_Ak(hparams['transform'], nn_model, data, k)):
            K = k
            break
    return K

def auto_invariant_statistical_loss(nn_model, data, hparams):
    assert len(data) == hparams['samples']

    K = 2
    print(f"K value set to {K}.")
    losses = []
    optimizer = optim.Adam(nn_model.parameters(), lr=hparams['eta'])

    for _ in tqdm(range(hparams['epochs'])):
        K_hat = get_better_K(nn_model, data, K, hparams)
        if K < K_hat:
            K = K_hat
            print(f"K value set to {K}.")

        def closure():
            optimizer.zero_grad()
            a_k = torch.zeros(K + 1)
            for i in range(hparams['samples']):
                x = torch.normal(0.0, 1.0, size=(K, 1))
                y_k = nn_model(x)
                a_k += generate_a_k(y_k, data[i])  # Assuming generate_a_k is defined
            loss = scalar_diff(a_k / a_k.sum())  # Assuming scalar_diff is defined
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        losses.append(loss.item())

    return losses
