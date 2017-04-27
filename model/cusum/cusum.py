import pystan
import pandas as pd
import numpy as np
import time
import math
import pickle
import seaborn as sns
from matplotlib import pyplot as plt

def load_data():
    temp_expected = pd.read_csv('../../data/csv/expected.csv')
    E = np.array(temp_expected['x'])
    
    temp_sim = pd.read_csv('../../data/csv/simulated.csv')
    temp_times = temp_sim[['Time1', 'Time2', 'Time3', 'Time4', 'Time5', 'Time6', 'Time7', 'Time8', 'Time9', 'Time10', 'Time11', 'Time12', 'Time13', 'Time14', 'Time15']]
    observed_values = np.matrix(temp_times, dtype=np.int)
    
    numRegions = observed_values.shape[0] #number of regions
    nt = observed_values.shape[1] #number of time points
    
    return numRegions, nt, E, observed_values

numRegions, nt, E, observed_values = load_data()

class cusum:

    def __init__(self, observed, expected):
        self.observed = observed
        self.expected = expected

    def norm_by_expectation(self):
        return np.array([obs/self.expected[i] for i, obs in enumerate(self.observed)])

    def general_trend(self):
        return self.norm_by_expectation().mean(axis=0)

    def remove_trend(self, trend):
        self.observed_norm = []
        for time_series in self.observed:
            normed_ts = np.array([np.round(time_series[i]/trend[i]) for i in range(len(trend))])
            self.observed_norm.append(normed_ts)
        self.observed_norm = np.array(self.observed_norm)

    def log_likelihood(self, x, lmbda):
        return -lmbda + x*np.log(lmbda)

    def likelihood_ratio(self, x, in_control, out_control):
        numerator = self.log_likelihood(x, out_control)
        denominator = self.log_likelihood(x, in_control)
        return numerator - denominator

    def control_chart(self, time_series, expectation, alpha):
        S = np.zeros(len(time_series))
        S[0] = self.likelihood_ratio(time_series[0], expectation, expectation*alpha)
        for i in range(1, len(time_series)):
            S[i] = max(0, S[i-1] + self.likelihood_ratio(time_series[i], expectation, expectation*alpha))
        return S

    def flag(self, chart, h):
        return bool((chart > h).sum())

    def simulate_trend(self, expectation, size, alpha, h):
        random_ts = np.random.poisson(expectation, size=size)
        chart = self.control_chart(random_ts, expectation, alpha)
        return self.flag(chart, h)

    def find_h(self, expectation, size, alpha, h_vals):
        #h_vals = np.arainge(0, 10, size=100)
        false_flag = []
        for h in h_vals:
            simulations = np.array([self.simulate_trend(expectation, size, alpha, h) for i in range(1000)])
            false_flag.append(simulations.mean())
        return next(h for h, prop in zip(h_vals, false_flag) if prop < 0.05)

    def generate_h(self, alpha, h_vals):
        temp = []
        our_size = self.observed.shape[1]
        for value in self.expected:
            new_value = self.find_h(value, our_size, alpha, h_vals)
            temp.append(new_value)
        self.h_array = np.array(temp)
