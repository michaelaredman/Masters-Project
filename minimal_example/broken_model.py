import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import theano.tensor as T

theano.config.compute_test_value = 'raise'
theano.config.gcc.cxxflags = "-fbracket-depth=16000"

temp_observed = pd.read_csv('observed.txt', delimiter=',')
temp_expected = pd.read_csv('exp.txt')

observed_values = np.matrix(temp_observed, dtype=np.float64)
E = np.array(temp_expected['x'], dtype=np.float64)

model = pm.Model()

numRegions = observed_values.shape[0]
nt = observed_values.shape[1]

with model:
    
    sigma_reg = pm.HalfNormal('sigma_reg', sd=1) 
    sigma_time = pm.HalfNormal('sigma_time', sd=1)
    
    time_dev = pm.MvNormal('time_dev', mu=np.zeros(nt), cov=np.identity(nt)*sigma_time, shape=nt) #temporal component
    spatial_dev = pm.MvNormal('spatial_dev', mu=np.zeros(numRegions), cov=np.identity(numRegions)*sigma_reg, shape=numRegions) #spatial component 

    mu = []
    for t in range(nt):
        for i in range(numRegions):
            mu.append(E[i]*T.exp(spatial_dev[i] + time_dev[t]))
    mu = T.stack(mu)

    print('mu defined')

    observed = []
    for i in range(numRegions):
        for t in range(nt):
            observed.append(pm.Poisson('observed_{}_{}'.format(i,t), mu = mu[i+numRegions*t], observed=observed_values[i,t]))

with model:
    start = pm.find_MAP()
    step = pm.Metropolis(model.vars)
    trace = pm.sample(draws=10000, step=step, start=start)
