from rltools.MountainCar import MountainCar
from rltools.theanotools import Theano_RBF_Projector, sym_RBF, sym_NRBF
from rltools.valueandpolicy import NeuroValPol, LinearPolicy
from rltools.representation import NRBFCoding

import sys

import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import theano

# theano.config.compute_test_value = 'raise'

class rbf_input_layer(object):
    def __init__(self, centers, widths, bias_term):
        self.centers = centers
        self.widths = widths
        self.bias_term = bias_term
        
    def __call__(self, s):
        x, _, size = sym_RBF(s, s, s, self.centers, self.widths, bias_term=self.bias_term)
        return x, size

def get_action(linpol, state, param, sigma):
    out = linpol(state, param)
    rv = scipy.stats.norm(out, sigma)
    return rv.rvs(size = 1)

def evaluate_valuefn(valuefn, params, points):
    values = np.empty(points.shape[0])
    for i, p in enumerate(points):
        values[i] = valuefn(p, params)
    return values

def evaluate_pol(linpol, params, points,):
    values = np.empty(points.shape[0])
    for i, p in enumerate(points):
        values[i] = linpol(p, params)
    return values

domain = MountainCar(random_start=True, max_episode=1000)
s_range = domain.state_range

xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 10),
                         np.linspace(s_range[0][1], s_range[1][1], 10))
       
centers = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))).astype(theano.config.floatX)
   
widths = (s_range[1]-s_range[0]).astype(theano.config.floatX)*0.085

phi = Theano_RBF_Projector(centers, widths, bias_term=False, normalized=True)
# NRBFCoding(widths, centers) #
alpha = 0.001
alpha_pi = 0.01
gamma = 0.99
beta_1 = 0.0
beta_2 = 0.0
layers= [1000, 300,100]

input_layer = rbf_input_layer(centers, widths, False)

valpol = NeuroValPol(2,
                     phi.size+1, 
                     layers, 
                     np.random.RandomState(np.random.randint(1)),
                     alpha, 
                     gamma, 
                     beta_1,
                     beta_2)

linpol = LinearPolicy(phi.size, 1, phi)
bparam = np.hstack((-np.ones(phi.size/4)*0.001,
                    -np.ones(phi.size/4)*0.001,
                    np.ones(phi.size/4)*0.001,
                    np.ones(phi.size/4)*0.001, 
                    0))  #np.zeros((1, phi.size+1))

# 
params = np.vstack((bparam, 
                    np.ones((1, phi.size+1))*0.001, 
                    -np.ones((1, phi.size+1))*0.001))
#                     (np.random.rand(5,phi.size+1) - 0.5)*0.002))

params = bparam.reshape((1,-1))

sigma = (domain.action_range[1]- domain.action_range[0])*0.1
print sigma

nsamples = 100        
xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], nsamples),
                     np.linspace(s_range[0][1], s_range[1][1], nsamples))
points = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))
grid = phi(points)


theta = np.zeros(phi.size)

num_episodes = 10000
max_length = 10000
for i in xrange(num_episodes):
    r, s_t = domain.reset()
    rewards = []
    for j in xrange(max_length):
        a = get_action(linpol, s_t, params[0,:], sigma)
#         print s_t, a
        r, s_tp1 = domain.step(a)
        rewards.append(r)
        
        rho = linpol.get_gaussian_rho(s_t, a, params, sigma)
        
#         x_t = phi(s_t)
#         x_tp1 = phi(s_tp1) if s_tp1 is not None else np.zeros_like(x_t)
#         delta =r + gamma*theta.dot(x_tp1) - theta.dot(x_t)
#         theta += alpha*delta*x_t
#         
        
        valpol.update(s_t, r, s_tp1, params, rho)
        
        v, grad = valpol.evluate_with_gradient(s_t, params[0,:])
        params[0,:] += grad[0,:]*0.0
        
        if s_tp1 is None:
            break
        
        s_t = s_tp1
    print np.sum(rewards)
    if (i%50) == 0:
        plt.figure()
        plt.subplot(1,2,1)
        plt.pcolormesh(xx, yy, evaluate_valuefn(valpol, params[0,:], points).reshape((nsamples, -1)))
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.pcolormesh(xx, yy, grid.dot(params[0,:-1]).reshape((nsamples, -1)) + params[0,-1])
#         plt.pcolormesh(xx, yy, grid.dot(theta).reshape((nsamples, -1)))
        plt.colorbar()
        plt.show()
    
        
    
