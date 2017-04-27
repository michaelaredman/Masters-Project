import pystan
import pandas as pd
import numpy as np
import time
import math
import pickle
from cusum import cusum as cs
import seaborn as sns
from matplotlib import pyplot as plt

def load_data():
    temp_expected = pd.read_csv('../../data/csv/expected.csv')
    E = np.array(temp_expected['x'])
    
    temp_sim = pd.read_csv('../../data/csv/simulated.csv')
    temp_times = temp_sim[['Time1', 'Time2', 'Time3', 'Time4', 'Time5', 'Time6', 'Time7', 'Time8', 'Time9', 'Time10', 'Time11', 'Time12', 'Time13', 'Time14', 'Time15']]
    observed_values = np.array(temp_times, dtype=np.int)
    
    numRegions = observed_values.shape[0] #number of regions
    nt = observed_values.shape[1] #number of time points
    
    return numRegions, nt, E, observed_values

numRegions, nt, E, observed_values = load_data()

class cusum:

    def __init__(self, observed, expected):
        self.observed = np.array(observed)
        self.expected = np.array(expected)

        self.numRegions = self.observed.shape[0]
        self.nt = self.observed.shape[1]

        self._calc_expected_trends()
        
    def _norm_by_expectation(self):
        """
        Return the observed values divided by their expected values
        """
        return np.array([obs/self.expected[i] for i, obs in enumerate(self.observed)])

    def _general_trend(self):
        """
        Return the mean temporal trend for a process of unit expectation
        """
        return self._norm_by_expectation().mean(axis=0)

    def _calc_expected_trends(self):
        """
        Calculate the temporally adjusted expectations for each region
        """
        general_trend = self._general_trend()
        self.expected_trend = np.zeros(shape=(self.numRegions, self.nt))
        for i in range(self.numRegions):
            self.expected_trend[i, :] = self.expected * general_trend 

    def control_chart(self, time_series, expectation, alpha):
        """
        Calls Fortran subroutine to return the CUSUM control chart for the a given time series
        which is assumed to be a Poisson distributed with given rate parameters.

        Inputs
        ------
        time_series : array (len_series)
            Time series which generates the control chart
        
        expectation : array (len_series)
            In-control rate parameter at each time step 

        alpha : real > 1
            Ratio of in-control rate to out-of-control rate

        Output
        ------
        control_chart : array (len_series)
            The desired control chart.
        """
        control_chart = cs.control_chart(time_series, expectation, alpha)
        return control_chart

    def simulate_trend(self, expectation, size, alpha, h):
        trends = np.zeros(dim=(size, len(expectation)))
        for i in range(expectation):
            trends[:, i] = np.random.poisson(expectation[i], size=size)
        return trends


    def generate_h(self, alpha, h_vals):
        temp = []
        our_size = self.observed.shape[1]
        for value in self.expected:
            new_value = self.find_h(value, our_size, alpha, h_vals)
            temp.append(new_value)
        self.h_array = np.array(temp)
