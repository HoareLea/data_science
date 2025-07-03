# Central Limit Theorem

"""
    - The Central Limit Theorem (CLT) states that the distribution of sample means will tend to be normally distributed, regardless of the shape
      of the population distribution, as long as the sample size is sufficiently large.
    - This script will demonstrate the CLT by simulating the sampling process from non-normal distributions and showing that the
      distribution of the sample means approaches a normal distribution as the sample size increases

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Central Limit Theorem demonstration with different distributions

# Parameters
population_size = 10000  # size of the population
sample_size = 10000  # size of each sample
num_samples = 1000  # number of samples

# Generate populations with different distributions
distributions = {
    "Uniform": np.random.uniform(low=0, high=1, size=population_size),
    "Exponential": np.random.exponential(scale=1.0, size=population_size),
    "Poisson": np.random.poisson(lam=3.0, size=population_size),
    "Binomial": np.random.binomial(n=10, p=0.5, size=population_size),
}

# Plot settings
fig, axes = plt.subplots(nrows=len(distributions), ncols=2, figsize=(14, 4 * len(distributions)))
fig.tight_layout(pad=5.0)

# Loop over each distribution and calculate sample means
for i, (dist_name, population) in enumerate(distributions.items()):
    # Collect sample means
    sample_means = [np.mean(np.random.choice(population, size=sample_size, replace=True)) for _ in range(num_samples)]

    # Plot the population distribution
    axes[i, 0].hist(population, bins=50, color='blue', alpha=0.7)
    axes[i, 0].set_title(f'Population Distribution ({dist_name})')
    axes[i, 0].set_xlabel('Value')
    axes[i, 0].set_ylabel('Frequency')

    # Plot the sample means distribution
    axes[i, 1].hist(sample_means, bins=50, color='green', alpha=0.7)
    axes[i, 1].set_title(f'Sample Means Distribution ({dist_name})')
    axes[i, 1].set_xlabel('Mean Value')
    axes[i, 1].set_ylabel('Frequency')

plt.savefig(Path("Statistical_Concepts/images/central_limit_theorem.png"))
