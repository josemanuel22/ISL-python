from tqdm import tqdm
import scipy.stats as stats
import torch.nn.functional as F
from scipy.stats import chisquare
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from functools import partial
from functorch.experimental.control_flow import map


def _sigmoid(y_hat, y):
    """Calculate the sigmoid function centered at y using PyTorch."""
    # return torch.relu((y - y_hat))
    return torch.sigmoid((y - y_hat) * 10.0)
    # return torch.nn.functional.softplus((y - y_hat))


def psi_m(y, m):
    """Calculate the bump function centered at m using PyTorch."""
    stddev = 0.1
    return torch.exp(-0.5 * ((y - m) / stddev) ** 2)


def phi(y_k, y_n):
    """Calculate the sum of sigmoid functions using PyTorch."""
    return torch.sum(_sigmoid(y_k, y_n))


def gamma(y_k, y_n, m):
    """Calculate the contribution to the m-th bin of the histogram using PyTorch."""
    # e_m = torch.tensor([1.0 if j == m else 0.0 for j in range(len(y_k) + 1)])
    e_m = torch.zeros(len(y_k) + 1, device=y_k.device)
    e_m[m] = 1.0
    return e_m * psi_m(phi(y_k, y_n), m)


def gamma_batch(y_k, y_n, m):
    """Calculate the contribution to the m-th bin of the histogram using PyTorch."""
    # e_m = torch.tensor([1.0 if j == m else 0.0 for j in range(len(y_k) + 1)])
    e_m = torch.eye(len(y_k) + 1, device=y_k.device, requires_grad=True)
    psi_m_partial = partial(psi_m, phi(y_k, y_n))
    return e_m * map(psi_m_partial, m)


def generate_a_k(y_hat, y):
    """Calculate the values of the real observation y in each of the components of the approximate histogram using PyTorch."""
    K = len(y_hat)
    return torch.sum(torch.stack([gamma(y_hat, y, k) for k in range(K+1)]), dim=0)


def generate_a_k_vectorize(y_hat, y):
    """Calculate the values of the real observation y in each of the components of the approximate histogram using PyTorch."""
    K = len(y_hat)
    indices = torch.arange(K+1, device=y_hat.device)
    gamma_values = gamma_batch(y_hat, y, indices)
    return torch.sum(gamma_values, dim=0)


def generate_a_k_batch(y_hat, y):
    """Calculate the values of the real observation y in each of the components of the approximate histogram using PyTorch."""
    K = y_hat.size(1)
    batch_size = y_hat.size(0)

    # Reshape y_hat and y to ensure correct broadcasting
    y_hat_repeated = y_hat.unsqueeze(2)  # shape (batch_size, K, 1, output_dim)
    y_repeated = y.unsqueeze(1)  # shape (batch_size, 1, output_dim)

    # Compute gamma values for all k
    gamma_values = torch.stack(
        [gamma_batch(y_hat_repeated, y_repeated, k, K) for k in range(K)], dim=0)
    return torch.sum(gamma_values, dim=0)


def scalar_diff(q):
    """Scalar difference between the vector representing our surrogate histogram and the uniform distribution vector using PyTorch."""
    return torch.sum((q - 1/len(q)) ** 2)


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


def get_window_of_Ak(model, data, K):
    """
    Generate a window of the rv's Ak for a given model and target function.
    """
    x_k = torch.normal(0.0, 1.0, size=(K, 1))
    window = [torch.sum(model(x_k).T < d) for d in data]
    return [torch.sum(torch.tensor(window) == i).item() for i in range(K + 1)]


def convergence_to_uniform(ak):
    """
    Test the convergence of the distribution of the window of the rv's Ak to a uniform
    distribution. It is implemented using a Chi-Square test.
    """
    # Convert to numpy array and ensure non-negative values
    ak_np = torch.tensor(ak).numpy()
    ak_np = ak_np.clip(min=0)

    # Adjust expected frequencies to ensure the total sum matches
    total = ak_np.sum()
    expected_np = (total / len(ak_np)) * np.ones(len(ak_np))

    # Perform the chi-square test
    _, p_value = chisquare(ak_np, expected_np)
    return p_value > 0.05


def get_better_K(nn_model, data, min_K, hparams):
    """
    Find a better K value based on the convergence to uniform distribution.
    """
    K = hparams['max_k']
    for k in range(min_K, hparams['max_k'] + 1):
        if not convergence_to_uniform(get_window_of_Ak(nn_model, data, k)):
            K = k
            break
    return K


def auto_invariant_statistical_loss(nn_model, data_loader, hparams):
    assert len(data_loader) == hparams['epochs']

    K = 2
    print(f"K value set to {K}.")
    losses = []
    optimizer = optim.Adam(nn_model.parameters(), lr=hparams['eta'])

    for data in tqdm(data_loader):
        K_hat = get_better_K(nn_model, data, K, hparams)
        if K < K_hat:
            K = K_hat
            print(f"K value set to {K}.")

        def closure():
            optimizer.zero_grad()
            a_k = torch.zeros(K + 1)
            distribution = torch.distributions.Normal(0, 1)
            for y in tqdm(data):
                x = distribution.rsample((K, 1)).float()
                y_k = nn_model(x)
                a_k += generate_a_k(y_k, y)  # Assuming generate_a_k is defined
            # Assuming scalar_diff is defined
            loss = scalar_diff(a_k / a_k.sum())
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        losses.append(loss.item())

    return losses


def auto_invariant_statistical_loss_batch(nn_model, data_loader, hparams):
    assert len(data_loader) == hparams['epochs']

    K = 2
    print(f"K value set to {K}.")
    losses = []
    optimizer = optim.Adam(nn_model.parameters(), lr=hparams['eta'])

    for data in tqdm(data_loader):
        K_hat = get_better_K(nn_model, data, K, hparams)
        if K < K_hat:
            K = K_hat
            print(f"K value set to {K}.")

        def closure():
            optimizer.zero_grad()
            distribution = torch.distributions.Normal(0, 1)

            def process_single_y(y):
                x = distribution.rsample((K, 1)).float()
                y_k = nn_model(x)
                return generate_a_k(y_k, y)

            a_k = torch.sum(
                map(process_single_y, data), dim=0)
            a_k.requires_grad_(True)
            loss = scalar_diff(a_k / a_k.sum())
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        losses.append(loss.item())

    return losses


# def ts_invariant_statistical_loss(rec, gen, X_t, X_t1, hparams):
#     losses = []
#     optim_rec = optim.Adam(rec.parameters(), lr=hparams['eta'])
#     optim_gen = optim.Adam(gen.parameters(), lr=hparams['eta'])
#     for (batch_X_t, batch_X_t1) in zip(X_t, X_t1):
#         rec.zero_grad()

#         for j in range(0, len(batch_X_t)-hparams['window_size'], hparams['window_size']):

#             def closure():
#                 a_k = torch.zeros(hparams['K'] + 1)
#                 batch_X_t[j:j + hparams['window_size']].view(-1, hparams['window_size'])
#                 s = rec(batch_X_t[j:j + hparams['window_size']].T)
#                 for i in range(hparams['window_size']):
#                     x_k = torch.normal(0.0, 1.0, size=(hparams['K'], 1))
#                     y_k = torch.cat([gen(torch.cat((x, s[:, i]))) for x in x_k])
#                     a_k += generate_a_k(y_k, batch_X_t1[j + i])
#                 loss = scalar_diff(a_k / a_k.sum())
#                 loss.backward()
#                 return loss
#             optim_rec.step(closure)
#             optim_gen.step(closure)
#             losses.append(closure().item())
#     return losses


class CondTimeGenModel(nn.Module):
    def __init__(self, seq_model, gen_model):
        super(CondTimeGenModel, self).__init__()
        self.seq_model = seq_model
        self.gen_model = gen_model
        # Assuming `state` and `noise` attributes are to be dynamically handled during forward pass

    def forward(self, x):
        self.state = self.seq_model(x)
        batch_size = x.size(1)
        distribution = torch.distributions.Normal(0, 1)
        noise = distribution.rsample((K, batch_size, 1)).float()
        # noise = torch.randn((1, batch_size, 1)).float()
        gen_input = torch.cat((noise, self.state.unsqueeze(0)), dim=0)
        return self.gen_model(gen_input)

    def generate(self, s, x, K):
        batch_size = x.size(1)
        noise = torch.randn((K, batch_size, 1)).float()
        gen_inputs = [
            torch.cat((noise[i, :, :], s.unsqueeze(0)), dim=0) for i in range(K)]
        return [self.gen_model(gen_inputs[i]) for i in range(K)]

    def generate_batch(self, s, x, K):
        batch_size = x.size(1)
        noise = torch.randn((K, batch_size, 1)).float()
        s_repeated = s.repeat(1, K).unsqueeze(0)
        noise_reshaped = noise.view(-1, batch_size * K, 1)
        gen_input = torch.cat((noise_reshaped, s_repeated), dim=0)
        return self.gen_model(gen_input)

    def generated_fictitious(self, x, K):
        self.state = self.seq_model(x)
        batch_size = x.size(0)
        distribution = torch.distributions.Normal(0, 1)
        noise = distribution.rsample((K, batch_size, 1)).float()
        # noise = torch.randn((K, batch_size, 1), requires_grad=True).float()
        s_repeated = self.state.repeat(K, 1)

        # noise_reshaped = noise.view(-1, batch_size * K, 1)
        noise_reshaped = noise.flatten(start_dim=0, end_dim=1)
        gen_input = torch.cat((noise_reshaped, s_repeated), 1)
        return self.gen_model(gen_input)


def generated_fictitious(model, x, K):
    state = model.seq_model(x)
    batch_size = x.size(0)
    distribution = torch.distributions.Normal(0, 1)
    noise = distribution.rsample(
        (K, batch_size, 1)).float()
    noise.requires_grad_(True)
    # noise = torch.randn((K, batch_size, 1), requires_grad=True).float()

    # s_repeated = state.repeat(K, 1)
    s_repeated = state.repeat_interleave(K, dim=0)

    # noise_reshaped = noise.view(-1, batch_size * K, 1)
    noise_reshaped = noise.flatten(start_dim=0, end_dim=1)
    gen_input = torch.cat((noise_reshaped, s_repeated), 1)
    return model.gen_model(gen_input)


def ts_invariant_statistical_loss(model, X_t, X_t_plus_1, hparams):
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=hparams['eta'])
    model.train()  # Set the model to training mode

    for batch_X_t, batch_X_t_plus_1 in tqdm(zip(X_t, X_t_plus_1)):
        model.zero_grad()  # Reset gradients
        for j in range(0, len(batch_X_t) - hparams['window_size'], hparams['window_size']):
            model.zero_grad()

            def closure():
                optimizer.zero_grad()
                a_k = torch.zeros(hparams['K'] + 1)
                input_tensor = batch_X_t[j: j + hparams['window_size']
                                         ].unsqueeze(1).unsqueeze(-1).float()
                y_k = model.generated_fictitious(
                    input_tensor, hparams['K']
                )
                a_k_values = [
                    generate_a_k(y_k[:, i:i + hparams['K']],
                                 batch_X_t_plus_1[i])
                    for i in range(0, hparams['window_size'], hparams['K'])
                ]
                a_k = sum(a_k_values)
                loss = scalar_diff(a_k / a_k.sum())
                loss.backward()
                return loss

            loss = closure()
            for name, param in model.named_parameters():
                print(f"Gradient for {name}: {param.grad}")
            optimizer.step(closure)  # Update model parameters
            losses.append(loss.item())

    return losses


def ts_invariant_statistical_loss_2(model, X_t, X_t_plus_1, hparams):
    losses = []
    optimizer = optim.Adam(model.parameters(), lr=hparams['eta'])
    model.train()  # Set the model to training mode

    for batch_X_t, batch_X_t_plus_1 in tqdm(zip(X_t, X_t_plus_1)):
        optimizer.zero_grad()  # Reset gradients
        input_tensor = batch_X_t.unsqueeze(1).unsqueeze(-1).float()
        y_k = generated_fictitious(model, input_tensor, hparams['K'])

        a_k_value = [
            generate_a_k(y_k[i*hparams['K']:(i+1) *
                             hparams['K'], :], batch_X_t_plus_1[i])
            for i in range(0, hparams['window_size'])
        ]
        a_k = torch.stack(a_k_value, dim=1)

        a_k = torch.sum(a_k, dim=1)
        norm = torch.norm(a_k, p=2)
        loss = scalar_diff(a_k / norm)
        loss.backward()
        optimizer.step()  # Update model parameters
        losses.append(loss.item())

    return losses
