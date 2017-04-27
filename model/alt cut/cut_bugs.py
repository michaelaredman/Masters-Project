import pystan
import pandas as pd
import numpy as np
import time
import seaborn as sns
import pickle
import theano
import theano.tensor as T
from matplotlib import pyplot as plt

def load_data():
    temp_expected = pd.read_csv('../../data/csv/expected.csv')
    E = np.array(temp_expected['x'])
    
    temp_sim = pd.read_csv('../../data/csv/simulated_spatial_corr.csv')
    temp_times = temp_sim[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15']]
    observed_values = np.matrix(temp_times, dtype=np.int)
    
    adj = pd.read_csv('../../data/csv/adjacency.csv', index_col=0)
    W = np.matrix(adj)
    
    numRegions = observed_values.shape[0] #number of regions
    nt = observed_values.shape[1] #number of time points
    
    alpha = 0.75 #this was 1 in the model but that makes the covariance matrix singular

    return numRegions, nt, E, W, alpha, observed_values 

numRegions, nt, E, W, alpha, observed_values = load_data()

model1_data = {'numRegions': numRegions,
               'nt': nt,
               'observed': observed_values,
               'log_expected': np.log(E),
               'W_n': int(W.sum()/2.0),
               'W': W}

iter_num = 1000

fit_model1 = pystan.stan(file='model1.stan', data=model1_data, iter=iter_num, chains=4)

model1_trace = fit_model1.extract()

model2_data = {'numRegions' : numRegions,
               'nt' : nt,
               'observed' : observed_values,
               'log_expected' : np.log(E)}

fit_model2 = pystan.stan(file='model2.stan', data=model2_data, iter=iter_num, chains=4)

model2_trace = fit_model2.extract()

mu_general = fit_model1['mu_general']
mu_specific = fit_model2['mu_specific']

general = pd.DataFrame(mu_general.mean(axis = 0))
specific = pd.DataFrame(mu_specific.mean(axis = 0))

general.to_csv('general.csv')
specific.to_csv('specific.csv')

num_samples = 20
selection = np.random.choice(np.arange(1000), num_samples, replace=False)

general_sample = mu_general[selection, :, :]
specific_sample = mu_specific[selection, :, :]

np.savetxt('general_sample.csv', general_sample.flatten())
np.savetxt('specific_sample.csv', specific_sample.flatten())
