import numpy as np
import matplotlib.pyplot as plt
from sedd.models.noise import LogLinearNoise
import torch

# Generate data: x values and their corresponding sine values
eps = 1e-3
steps = 128
x = torch.linspace(1, eps, steps)
y = torch.zeros(steps)
for i in range(steps):
    y[i] = 1 - (-x[i]).exp()

# Create a 2D plot
plt.figure()
plt.plot(x, y, label='Noise')
plt.title('1 - e^(-x)')
plt.xlabel('t')
plt.ylabel('1 - e^(-x)')
plt.legend()

# Save the plot to a file
plt.savefig('e_neg_x.png')
