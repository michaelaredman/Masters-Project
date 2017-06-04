import pandas as pd
import numpy as np
import math
from cusum import cusum as cs
import logging
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

logger = logging.getLogger("cusum")
logging.basicConfig(level=logging.DEBUG)

class cusum:

    def __init__(self, observed, expected, seed=None):
        self.observed = np.array(observed)
        self.expected = np.array(expected)

        np.random.seed(seed=seed)

        self.numRegions = self.observed.shape[0]
        self.nt = self.observed.shape[1]

        self._calc_expected_trends()

        #default alpha (ratio of in-control to out-of-control rates)
        self.alpha = 1.5
        #default size (number of times to simulated trend to find h value)
        self.size = 4000
        
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
            self.expected_trend[i, :] = self.expected[i] * general_trend 

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

    def simulate_trend(self, expectation, size):
        """
        Simulate a Poisson r.v., for a given temporal expectation, (size) times.
        """
        trends = np.zeros(shape=(size, len(expectation)))
        for i in range(len(expectation)):
            trends[:, i] = np.random.poisson(expectation[i], size=size)
        return trends

    def find_h(self, expectation, size, alpha, h_values, p_value):
        simulated_series = self.simulate_trend(expectation, size)
        false_positive_rates = cs.false_positive(simulated_series, expectation, alpha, h_values)
        try:
            my_h = next(h for h, rate in zip(h_values, false_positive_rates) if rate < p_value)
        except StopIteration:
            logger.warning("Unable to find suitable h value. False positive rates were as follows:")
            logger.warning(np.array2string(false_positive_rates))
            my_h = 0
            
        return my_h
        
    def generate_h(self, h_values, p_value):
        temp = []
        for regional_expectation in self.expected_trend:
            new_value = self.find_h(regional_expectation, self.size, self.alpha, h_values, p_value=p_value)
            temp.append(new_value)
        self.h_array = np.array(temp)

    def test(self):
        flags = np.zeros(shape=self.numRegions)
        for i in range(self.numRegions):
            c_chart = cs.control_chart(self.observed[i, :], self.expected_trend[i, :], alpha=self.alpha)
            flags[i] = cs.flag(c_chart, self.h_array[i])
        return flags

    def test_regions(self):
        flags = self.test()
        return np.where(flags == 1)[0]
        


csum = cusum(observed_values, E)
csum.generate_h(np.linspace(0, 10, 250), p_value=0.01)
unusual_predicted = csum.test_regions() + 1
