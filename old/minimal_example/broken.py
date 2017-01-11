import numpy as np
import pandas as pd
import pymc3 as pm
import theano
import theano.tensor as T

theano.config.compute_test_value = 'raise'
theano.config.gcc.cxxflags = "-fbracket-depth=16000"
theano.config.optimizer = 'fast_compile'

temp_observed = pd.read_csv('observed.txt', delimiter=',')
temp_expected = pd.read_csv('exp.txt')

observed_values = np.matrix(temp_observed, dtype=np.float64)
E = np.array(temp_expected['x'], dtype=np.float64)

model = pm.Model()

numRegions = observed_values.shape[0]
nt = observed_values.shape[1]

numRegions = 10
nt = 2

with model:
    
    sigma_reg = pm.HalfNormal('sigma_reg', sd=1) 
    sigma_time = pm.HalfNormal('sigma_time', sd=1)
    
    time_dev = pm.MvNormal('time_dev', mu=np.zeros(nt), cov=np.identity(nt)*sigma_time, shape=nt) #temporal component
    spatial_dev = pm.MvNormal('spatial_dev', mu=np.zeros(numRegions), cov=np.identity(numRegions)*sigma_reg, shape=numRegions) #spatial component 

    print('starting mu def')
    
    mu = []
    for i in range(numRegions):
        mu.append(T.stack([E[i]*T.exp(spatial_dev[i] + time_dev[t]) for t in range(nt)]))

    print('mu defined')

    observed = []
    for i in range(numRegions):
        observed.append(pm.Poisson('observed_{}'.format(i), mu = mu[i], observed=observed_values[i, :nt], shape=nt))
    
    print('observed defined')

with model:
    #db = pm.backends.Text('test')
    #print('starting MAP find')
    #start = pm.find_MAP()
    #print(start)
    #step = pm.Metropolis(model.vars)
    trace = pm.sample(draws=10000)
    #advi_sample = pm.advi(10000)
