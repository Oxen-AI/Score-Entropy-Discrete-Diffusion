import numpy as np
import matplotlib.pyplot as plt
from sedd.models.noise import LogLinearNoise
import torch

# Generate data: x values and their corresponding sine values
eps = 1e-3
steps = 128
x = np.linspace(1, eps, steps)
y = np.zeros(steps)
for i in range(steps):
    noise = LogLinearNoise()
    sigma, _ = noise(torch.tensor(x[i]))
    # y[i] = 1 - (-sigma).exp()
    y[i] = sigma.item()

# Create a 2D plot
plt.figure()
plt.plot(x, y, label='Noise')
plt.title('Log Linear Noise')
plt.xlabel('t')
plt.ylabel('LogLinearNoise(t)')
plt.legend()

# Save the plot to a file
plt.savefig('log_linear_noise.png')
