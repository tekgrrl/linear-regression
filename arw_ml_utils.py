import os
import numpy as np

def regenerate_food_truck_data(n_samples, seed=0):
    """
    Generates a dataset for a food truck profit prediction based on city population.
    
    Parameters:
        n_samples (int): The number of sample data points to generate.
        seed (int): The seed for NumPy's random number generator for reproducibility.
        
    """

    # Create a random number generator with the given seed
    rng = np.random.default_rng(seed)
    
    # Generate city populations uniformly distributed between 5 and 25 (in 10,000s)
    populations = rng.uniform(5, 25, n_samples)
    
    # Generate profits with a linear relationship plus some noise
    # Assume a base profit model of profit = (4.5 * population) - 14 + noise
    noise = rng.normal(0, 1, n_samples)  # Normal distribution noise
    profits = (4.5 * populations) - 14 + noise
    
    # Combine populations and profits into a single dataset
    dataset = np.column_stack((populations, profits))

    # if data directory does not exist, create it
    if not os.path.exists('data'):
        os.makedirs('data')

    # if dataset exists as a file, delete it
    if os.path.exists('data/food-truck-data.txt'):
        os.remove('data/food-truck-data.txt')
    
    # Save the dataset to a file    
    with open('data/food-truck-data.txt', 'w') as f:
        for row in dataset:
            f.write(f'{row[0]:.6f},{row[1]:.6f}\n')

    print(f'Generated and saved {n_samples} data points to data/food-truck-data.txt')
