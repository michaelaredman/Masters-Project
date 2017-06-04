import numpy as np
import pandas as pd
from scipy.misc import factorial

observations_A_df = pd.read_csv('../../data/csv/simulated_A.csv')
observations_A = np.array(observations_A_df[observations_A_df.columns[1:]])
observations_B_df = pd.read_csv('../../data/csv/simulated_B.csv')
observations_B = np.array(observations_B_df[observations_B_df.columns[1:]])

numSamples = 50

log_mu_general_A = np.loadtxt('general_sample_A.csv').reshape(numSamples, 210, 15)
log_mu_specific_A = np.loadtxt('specific_sample_A.csv').reshape(numSamples, 210, 15)
log_mu_general_B = np.loadtxt('general_sample_B.csv').reshape(numSamples, 210, 15)
log_mu_specific_B= np.loadtxt('specific_sample_B.csv').reshape(numSamples, 210, 15)

#Raumanujan's approximation for log(x!) was needed. 
ramanujan = lambda n: n*np.log(n) - n + np.log(n*(1 + 4*n*(1 + 2*n)))/6 + np.log(np.pi)/2

def poisson_lpdf(x, lmbda):
    return -lmbda + x*np.log(lmbda) - ramanujan(x)

def poisson_likelihood(x, lmbda):
    log_likelihoods = poisson_lpdf(x, lmbda)
    return np.exp(np.sum(log_likelihoods))




