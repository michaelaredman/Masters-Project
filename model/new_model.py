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

theano.config.compute_test_value = 'raise'
#theano.config.gcc.cxxflags = "-fbracket-depth=16000 -O0" # default is 256 | compilation optimizations are turned off for faster compilation
theano.config.exception_verbosity= 'high'
theano.config.openmp = True # this isn't working with gcc for some reason
theano.config.optimizer = 'fast_compile'

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

nt = 7

class CAR(Continuous):
    
    def __init__(self, adj=None, deg=None, tau=None, alpha=None, *args, **kwargs):
        super(CAR, self).__init__(*args, **kwargs)
        self.D = deg
        self.adj = adj # adjacency matrix
        self.tau = tau 
        self.alpha = alpha
        self.calculations()
        print('init CAR')

    def calculations(self):
        D = self.D
        adj = self.adj
        alpha = self.alpha
        det_D = T.nlinalg.trace(D) # the determinant of D
        D_inv = np.linalg.inv(D)
        D_inv_sqrt = np.power(D_inv, 0.5)
        temp_matrix = np.matmul(D_inv_sqrt, adj)
        lambda_matrix = np.matmul(temp_matrix, D_inv_sqrt)
        eigenvals = np.linalg.eigvals(lambda_matrix)
        self.term_evals = 0.5*np.log(np.ones(self.shape) - alpha*eigenvals).sum()
        self.term_detD = 0.5*np.log(det_D)
        self.term_const = -self.shape * 0.5 * np.log(2.0*np.pi)
        self.prec = theano.shared(D - alpha*W) # precision matrix sans tau
        print('Calculations calculated')
        

    def logp(self, value):
        tau = self.tau
        prec = self.prec
        term_evals = self.term_evals
        term_detD = self.term_detD
        term_const = self.term_const
        
        term_tau = self.shape * 0.5 * T.log(tau) 
        
        # this should be rewritten to use sparse matricies
        term_phi_partial = T.dot(value, prec)
        term_phi = - 0.5 * tau * T.dot(term_phi_partial, T.transpose(value))
        
        result = term_const + term_tau + term_detD + term_evals + term_phi
        return result

print('CAR definition worked!')

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
    #sigma_lambda = pm.HalfNormal('sigma_lambda', sd=1)
    
    v = CAR('v', adj=W, deg=D, tau=sigma_v, alpha=alpha, shape=numRegions, testval=np.ones(numRegions))
    #lmbda = pm.MvNormal('lambda', mu=v, cov=np.identity(numRegions)*sigma_lambda, shape=numRegions)
    
    
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
    step = pm.Metropolis()
    print('Metropolis initialized at', time.ctime())
    db = pm.backends.Text('blah_trace')
    trace = pm.sample(draws=8000, trace=db, step = step)

print('End time: ', time.ctime())

#trace_burn = trace[:][3000:]
pm.traceplot(trace)
plt.savefig('tr.png')

#trace = pm.backends.text.load('test')
