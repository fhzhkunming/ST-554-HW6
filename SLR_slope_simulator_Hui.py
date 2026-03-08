"""
SLR_slope_simulator.py

This module defines a class for simulating the sampling distribution of the
slope estimator in simple linear regression.
"""

# import modules needed
import numpy as np
from numpy.random import default_rng
from sklearn import linear_model
import matplotlib.pyplot as plt

class SLR_slope_simulator:
    """
    A class to simulate slope estimates from repeated 
    simple linear regression (SLR) fits.
    """
    def __init__(self, beta_0, beta_1, x, sigma, seed):
        # store model parameters
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.sigma = sigma
        
        # store x values and sample size
        self.x = x
        self.n = len(x)
        
        # set ramdom number generator
        self.rng = default_rng(seed)
        
        # create empty list to store slope estimates
        self.slopes = []
    
    def generate_data(self):
        """
        Generate one dataset (x, y) from the linear model:
        y = beta_0 + beta_1*x + noise
        """
        noise = self.rng.normal(0, self.sigma, self.n)
        y = self.beta_0 + self.beta_1 * self.x + noise
        return self.x, y
    
    # prepare for the LR fit
    def fit_slope(self, x, y):
        """
        Fit SLR model and return the estimated slope.
        """
        reg = linear_model.LinearRegression()
        reg.fit(x.reshape(-1, 1), y)
        return reg.coef_[0]
    
    # run simulations
    def run_simulations(self, n_sims):
        """
        Run n_sims simulations and store slopes in self.slope
        """
        self.slopes = [] # reset before running
        for _ in range(n_sims):
            x, y = self.generate_data()
            slope = self.fit_slope(x, y)
            self.slopes.append(slope)
           
    def plot_sampling_distribution(self):
        """
        Plot a histogram of the simulated slope estimates.
        """
        # check if simulations have been run
        if len(self.slopes) == 0:
            print("run_simulations() must be called first!")
        else:
            plt.hist(self.slopes)
            plt.xlabel("Slope estimates")
            plt.ylabel("Frequency")
            plt.title("Sampling Distribution of the Slope")
            plt.show()
    
    def find_prob(self, value, sided):
        """
        Compute probability based on slope distribution.
        
        sided options:
        - "above": P(slope > value)
        - "below": P(slope < value)
        - "two-sided": P(|slope| > |value|)
        """
        # check if simulations have been run
        if len(self.slopes) == 0:
            print("run_simulations() must be called first!")
            return None
        
        # convert slopes to an np.array
        slopes = np.array(self.slopes)
        
        if sided == "above":
            prob = np.mean(slopes > value)            
        elif sided == "below":
            prob = np.mean(slopes < value)
        elif sided == "two-sided":
            prob = np.mean(np.abs(slopes) > abs(value))        
        else:
            raise ValueError("sided must be 'above', 'below', or 'two-sided'")        
        return prob      