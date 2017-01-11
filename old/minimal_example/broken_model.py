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

with model:
    
    sigma_reg = pm.HalfNormal('sigma_reg', sd=1) 
    sigma_time = pm.HalfNormal('sigma_time', sd=1)
    
    time_dev = pm.MvNormal('time_dev', mu=np.zeros(nt), cov=np.identity(nt)*sigma_time, shape=nt) #temporal component
    spatial_dev = pm.MvNormal('spatial_dev', mu=np.zeros(numRegions), cov=np.identity(numRegions)*sigma_reg, shape=numRegions) #spatial component 

    print('starting mu def')
    
    #mu = []
    #for t in range(nt):
    #    for i in range(numRegions):
    #        mu.append(pm.Deterministic('mu_{}_{}'.format(i, t), E[i]*T.exp(spatial_dev[i] + time_dev[t])))
    #print('starting stack')
    #mu_temp_stack = []
    #for i in range(15):
    #    mu_temp_stack.append(T.stack(mu[i*209:209*(i+1)]))
    #    print('the {}th mu is stacked'.format(i))
    #mu = T.stack(mu_temp_stack)
    #mu = mu.flatten()

    mu_alt = np.empty(nt*numRegions, dtype=object)
    for t in range(nt):
        for i in range(numRegions):
            mu_alt[i + numRegions*t] = pm.Deterministic('mu_{}_{}'.format(i, t), E[i]*T.exp(spatial_dev[i] + time_dev[t]))
    mu = T.stack(mu_alt)
    
    print('mu defined')

    observed = []
    for i in range(numRegions):
        for t in range(nt):
            observed.append(pm.Poisson('observed_{}_{}'.format(i,t), mu = mu[i+numRegions*t], observed=observed_values[i,t]))

    print('observed defined')

with model:
    start = pm.find_MAP()
    step = pm.Metropolis(model.vars)
    trace = pm.sample(draws=10000, step=step, start=start)
