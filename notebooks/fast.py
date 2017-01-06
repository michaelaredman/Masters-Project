import pymc3 as pm
import scipy
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import time
from matplotlib import pyplot as plt

theano.config.compute_test_value = 'raise'
theano.config.gcc.cxxflags = "-fbracket-depth=16000 -O0" # default is 256 | compilation optimizations are turned off for faster compilation
#theano.config.openmp = True # this isn't working with gcc for some reason
theano.config.optimizer = 'fast_compile'

def load_data():
    temp_expected = pd.read_csv('~/4th Year/project/data/csv/expected.csv')
    E = np.array(temp_expected['x'])
    
    temp_sim = pd.read_csv('~/4th Year/project/data/csv/simulated.csv')
    temp_times = temp_sim[['Time1', 'Time2', 'Time3', 'Time4', 'Time5', 'Time6', 'Time7', 'Time8', 'Time9', 'Time10', 'Time11', 'Time12', 'Time13', 'Time14', 'Time15']]
    observed_values = np.matrix(temp_times)
    
    adj = pd.read_csv('/Users/Mike/4th Year/project/data/csv/adjacency.csv', index_col=0)
    W = np.matrix(adj)
    
    numRegions = observed_values.shape[0] #number of regions
    nt = observed_values.shape[1] #number of time points
    Q = np.diag(np.ones(nt-1), k=1) #'adjacency matrix' in time
    
    #making the inverse covariance matricies for the CAR models (ignoring their variances)
    alpha = 0.75 #this was 1 in the model but that makes the covariance matrix singular
    D = np.diag(np.array(W.sum(0))[0]) #diag(d_1,..,d_numRegions) with d_i the number of neighbours of region i
    Tau_v_unscaled = np.array(D - alpha*W)
    Tau_gamma_unscaled = np.identity(n=nt) - Q

    return numRegions, nt, E, Tau_v_unscaled, Tau_gamma_unscaled, observed_values

prob_z = 0.95 #probability of a region following the area specific model

numRegions, nt, E, Tau_v_unscaled, Tau_gamma_unscaled, observed_values = load_data()

model = pm.Model()


print('Data loaded')

print('Starting time: ', time.ctime())

nt = 10
numRegions = 30

with model:
    
    sigma_reg = pm.HalfNormal('sigma_reg', sd=1)
    sigma_time = pm.HalfNormal('sigma_time', sd=1)
    
    mu = np.empty(shape=(numRegions, nt), dtype=object)
    
    time_dev = pm.MvNormal('time_dev', mu=np.zeros(nt), cov=np.identity(nt)*sigma_time, shape=nt)
    spatial_dev = pm.MvNormal('spatial_dev', mu=np.zeros(numRegions), cov=np.identity(numRegions)*sigma_reg, shape=numRegions)
    
    for i in range(numRegions):
        for t in range(nt):
            mu[i, t] = E[i]*T.exp(spatial_dev[i]+time_dev[t])
        
    observed = np.empty(shape=(numRegions, nt), dtype=object)
    for i in range(numRegions):
        for t in range(nt):
            observed[i, t] = pm.Poisson('observed_{}_{}'.format(i,t), mu = mu[i,t], observed=observed_values[i,t])


print('Model defined')

with model:
    step = pm.Metropolis(model.vars)
    start = pm.find_MAP()
    print('Time MAP found: ', time.ctime())
    print(start)
    trace = pm.sample(draws=10000, step=step, start=start)

print('End time: ', time.ctime())

trace_burn = trace[:][3000:]
pm.traceplot(trace_burn)
plt.savefig('trace.png')



