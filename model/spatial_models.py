import matplotlib
matplotlib.use('Agg')
import pymc3 as pm
import numpy as np
import theano
import theano.tensor as T
from pymc3.distributions import Continuous

class CAR(Continuous):
    
    def __init__(self, adj=None, deg=None, tau=None, alpha=None, *args, **kwargs):
        super(CAR, self).__init__(*args, **kwargs)
        self.D = deg # degree of each region - rewrite to calculate this in on initialization from adj
        self.adj = adj # adjacency matrix
        self.tau = tau # variance parameter
        self.alpha = alpha # spatial dependence parameter
        self.calculations()

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
        self.prec = theano.shared(D - alpha*adj) # precision matrix sans tau

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
