import pystan
import pandas as pd
import numpy as np
import time
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
        pass

    def norm_by_expectation(self):
        pass
