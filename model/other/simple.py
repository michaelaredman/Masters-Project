import matplotlib
matplotlib.use('Agg')
import pymc3 as pm
import scipy
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import time
from matplotlib import pyplot as plt
from pymc3.distributions.timeseries import GaussianRandomWalk
from pymc3.distributions import Continuous
from spatial_models import CAR

theano.config.compute_test_value = 'raise'
#theano.config.gcc.cxxflags = "-fbracket-depth=16000 -O0" # default is 256 | compilation optimizations are turned off for faster compilation
theano.config.exception_verbosity= 'high'
#theano.config.openmp = True # this isn't working with gcc on mac for some reason
#theano.config.profile_memory = True

def load_data():
    temp_expected = pd.read_csv('../data/csv/expected.csv')
    E = theano.shared(np.array(temp_expected['x']))
    
    temp_sim = pd.read_csv('../data/csv/simulated.csv')
    temp_times = temp_sim[['Time1', 'Time2', 'Time3', 'Time4', 'Time5', 'Time6', 'Time7', 'Time8', 'Time9', 'Time10', 'Time11', 'Time12', 'Time13', 'Time14', 'Time15']]
    observed_values = np.matrix(temp_times)
    
    adj = pd.read_csv('../data/csv/adjacency.csv', index_col=0)
    W = np.matrix(adj)
    
    numRegions = observed_values.shape[0] #number of regions
    nt = observed_values.shape[1] #number of time points
    
    #making the inverse covariance matricies for the CAR models (ignoring their variances)
    alpha = 0.75 #this was 1 in the model but that makes the covariance matrix singular
    D = np.diag(np.array(W.sum(0))[0]) #diag(d_1,..,d_numRegions) with d_i the number of neighbours of region i
    # Tau_v_unscaled = theano.shared(np.array(D - alpha*W)) # no longer needed with new code

    return numRegions, nt, E, W, D, alpha, observed_values 

prob_z = 0.95 #probability of a region following the area specific model

numRegions, nt, E, W, D, alpha, observed_values = load_data()

model = pm.Model()

observed_values = theano.shared(observed_values)

print('Data loaded')

print('Starting time: ', time.ctime())

nt = 5

def sum_to_zero(variable):
    return variable - variable.mean()
    
with model:
    """
    BYM prior on the spatial component
    
        lambda ~ Normal(v_i, sigma_lambda)
    
        v ~ CAR(W, sigma_v)
    
    where W is the adjacency matrix.
    
    The CAR model is equivalent to,
    
        v ~ N(0, T^-1)
    
    with,  T = (D - alpha*W)*sigma_v
    
    D = diag(d_1, ..., d_n)
    d_i = 'degree' of region i
    alpha = level of spatial dependence 
    
    We place vague priors on the variances.
    
    """
    sigma_v = pm.HalfNormal('sigma_v', sd=1) # change this to something more vague
    #sigma_lmbda = pm.HalfNormal('sigma_lmbda', sd=1) # change this to something more vague
    
    v = CAR('v', adj=W, deg=D, tau=sigma_v, alpha=alpha, shape=numRegions, testval=np.ones(numRegions))
    #v_stz = pm.Deterministic('v_stz', sum_to_zero(v))
    #lmbda = pm.MvNormal('lmbda', mu = v, cov = sigma_lmbda*np.identity(numRegions), shape=numRegions)
    
    
    sigma_temporal = pm.HalfNormal('sigma_temporal', sd=1)
    temporal  = GaussianRandomWalk('temporal', sd=sigma_temporal, shape=nt)
    
    # mu_it = Expected*exp(lambda_i + temporal_t)
    
    mu_temp = [] #rate parameters over the time points
    for i in range(numRegions):
        mu_temp.append(T.stack([E[i]*T.exp(v[i] + temporal[t]) for t in range(nt)]))
    mu = T.stack(mu_temp)
    
    observed = pm.Poisson('observed', mu = mu, observed=observed_values[:, :nt], shape=(numRegions, nt))
    
print('Model defined at ', time.ctime())

with model:
    #step = pm.Metropolis()
    #comp_step = pm.CompoundStep([
        
    #print('Metropolis initialized at', time.ctime())
    #db = pm.backends.Text('metropolis/nuts_trace')
    #trace = pm.sample(draws=10000, step = step, trace=db)

    step = pm.NUTS(scaling=np.ones(217))

    print('first step defined')
    
    trace = pm.sample(5000, step)
    cov = pm.trace_cov(trace[1000:])

    
    step2 = pm.NUTS(scaling=cov, is_cov=True)
    print('2nd step defined')
    trace2 = pm.sample(5000, step2)
    

print('End time: ', time.ctime())

#trace_burn = trace[:][3000:]
#pm.traceplot(trace)
#plt.savefig('nuts.png')

#trace = pm.backends.text.load('test')
