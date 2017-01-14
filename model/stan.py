import pystan
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt

def load_data():
    temp_expected = pd.read_csv('../data/csv/expected.csv')
    E = np.array(temp_expected['x'])
    
    temp_sim = pd.read_csv('../data/csv/simulated.csv')
    temp_times = temp_sim[['Time1', 'Time2', 'Time3', 'Time4', 'Time5', 'Time6', 'Time7', 'Time8', 'Time9', 'Time10', 'Time11', 'Time12', 'Time13', 'Time14', 'Time15']]
    observed_values = np.matrix(temp_times, dtype=np.int)
    
    adj = pd.read_csv('../data/csv/adjacency.csv', index_col=0)
    W = np.matrix(adj)
    
    numRegions = observed_values.shape[0] #number of regions
    nt = observed_values.shape[1] #number of time points
    
    #making the inverse covariance matricies for the CAR models (ignoring their variances)
    alpha = 0.75 #this was 1 in the model but that makes the covariance matrix singular

    return numRegions, nt, E, W, alpha, observed_values 

numRegions, nt, E, W, alpha, observed_values = load_data()

model_data = {'numRegions': numRegions,
              'nt': nt,
              'observed': observed_values,
              'log_expected': np.log(E),
              'W_n': int(W.sum()/2.0),
              'W': W,
              'alpha': 0.75}

print('Starting fit at: ', time.ctime())

fit = pystan.stan(file='bym.stan', data=model_data, iter=1000, chains=4)

fit.plot()
plt.show()
