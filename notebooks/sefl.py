import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import theano.tensor as T

theano.config.compute_test_value = 'raise'

temp_observed = pd.read_csv('observed.csv', delimiter=',')
temp_expected = pd.read_csv('expected.csv', delimiter=',')

observed_values = np.matrix(temp_observed, dtype=np.float64)
E = np.array(temp_expected, dtype=np.float64)


with model:
    
    sigma_reg = pm.HalfNormal('sigma_reg', sd=1) 
    sigma_time = pm.HalfNormal('sigma_time', sd=1)
    
    time_dev = pm.MvNormal('time_dev', mu=np.zeros(nt), cov=np.identity(nt)*sigma_time, shape=nt) #temporal component
    spatial_dev = pm.MvNormal('spatial_dev', mu=np.zeros(numRegions), cov=np.identity(numRegions)*sigma_reg, shape=numRegions) #spatial component 

    mu = []
    for i in range(numRegions):
        for t in range(nt):
            mu.append(E[i]*T.exp(spatial_dev[i] + time_dev[t]))
    mu = T.stack(mu)

    observed = []
    for i in range(numRegions):
        for t in range(nt):
            observed.append(pm.Poisson('observed_{}_{}'.format(i,t), mu = mu[i+numRegions*t], observed=observed_values[i,t]))

with model:
    start = pm.find_MAP()
    step = pm.Metropolis(model.vars)
    trace = pm.sample(draws=10000, step=step, start=start)
