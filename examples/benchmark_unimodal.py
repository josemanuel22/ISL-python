import sys
sys.path.append('../ISL-python')

from ISL.isl import invariant_statistical_loss, auto_invariant_statistical_loss
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal, Pareto
from torch.utils.data import DataLoader, TensorDataset

# Define the neural network architectures
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 7),
            nn.ELU(),
            nn.Linear(7, 13),
            nn.ELU(),
            nn.Linear(13, 7),
            nn.ELU(),
            nn.Linear(7, 1)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 11),
            nn.ELU(),
            nn.Linear(11, 29),
            nn.ELU(),
            nn.Linear(29, 11),
            nn.ELU(),
            nn.Linear(11, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters and models setup
noise_model = Normal(0.0, 1.0)
target_model = [Normal(5.0, 2.0), Pareto(5.0, 1.0)]  # Custom mixture model required
gen = Generator()
dscr = Discriminator()
# Hyperparameters for Vanilla GAN (may need to adjust)
hparams_vanilla_gan = {
    'data_size': 100,
    'batch_size': 1,
    'epochs': 10000,
    'lr_dscr': 1e-4,
    'lr_gen': 1e-4,
    'dscr_steps': 4,
    'gen_steps': 0,
    'noise_model': noise_model,
    'target_model': target_model
}

# Function to train Vanilla GAN (implementation depends on your GAN setup)

# Hyperparameters for Auto Invariant Statistical Loss
hparams = {
    'max_k': 10,
    'samples': 1000,
    'epochs': 1000,
    'eta': 1e-2,
    'transform': noise_model,
    'K': 10
}

def sample_from_target_model(target_model, num_samples):
    # Sample from the target_model here
    # This is a placeholder for your actual implementation
    # For example, if target_model is a normal distribution:
    samples = torch.normal(5.0, 2.0, size=(num_samples, 1))
    return samples

# Generate the training dataset
num_total_samples = hparams['samples'] * hparams['epochs']
train_data = sample_from_target_model(target_model, num_total_samples)

# Assuming the train_data is a 1D tensor, reshape it to 2D (num_samples, num_features) if necessary
# If train_data is already 2D (num_samples, num_features), you can skip the reshaping
#train_data = train_data.view(-1, 1)

# Create a DataLoader
# Note: In PyTorch, the DataLoader expects a dataset object, so we wrap our data in a TensorDataset
#dataset = TensorDataset(train_data)  # Using train_data as both inputs and targets for simplicity
loader = DataLoader(train_data, batch_size=hparams['samples'], shuffle=True)

# Train the model
# Assuming 'gen' is your model and 'auto_invariant_statistical_loss' is defined as per your previous messages
losses = invariant_statistical_loss(gen, loader, hparams)

hparams = {
    'max_k': 10,
    'samples': 1000,
    'epochs': 1000,
    'eta': 1e-2,
    'transform': noise_model,
    'K': 10
}

losses = auto_invariant_statistical_loss(gen, loader, hparams)

import matplotlib.pyplot as plt
import torch

# Assuming 'model' is your trained model and 'noise' is the input for the model
# Generating data
noise = torch.normal(0.0, 1.0, size=(10000, 1))
gen.eval()  # Set the gen to evaluation mode
with torch.no_grad():  # Turn off gradients for evaluation
    generated_data = gen(noise).detach().cpu().numpy()  # Convert to NumPy array

# Plotting the histogram
import matplotlib.pyplot as plt

plt.hist(generated_data, bins=50, alpha=0.75)  # Adjust the number of bins as needed
plt.title('Histogram of Generated Data')
plt.xlabel('Data values')
plt.ylabel('Frequency')
plt.show()
