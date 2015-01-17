from rltools.theanotools import NeuroSFTD, sym_RBF
from rltools.SwingPendulum import SwingPendulum, Swing_stabilize

from itertools import izip

import theano
import theano.tensor as T

import numpy as np

import matplotlib.pyplot as plt


theano.config.compute_test_value = 'ignore'
# theano.config.exception_verbosity = 'high'

def generate_traj(domain, policy, max_length):
    domain = domain.copy()
    r, s = domain.reset()
    states = []
    rewards = []
    next_states = []
    for i in xrange(max_length):
        states.append(s)
        r, s = domain.step(policy(s))
        rewards.append(r)
        next_states.append(s)
        if s is None:
            break
    return states, rewards, next_states

def generate_data(domain, policy, num, max_length):
    for i in xrange(num):
        yield generate_traj(domain, policy, max_length)
        

domain = SwingPendulum(random_start=True)
policy = Swing_stabilize(domain)
s_range = domain.state_range

ds = T.TensorType(dtype = theano.config.floatX, broadcastable=(False,False))('ds')
s = T.TensorType(dtype = theano.config.floatX, broadcastable=(False,False))('s')

s.tag.test_value = np.random.rand(5, 2).astype('float32')
ds.tag.test_value = np.random.rand(5, 2).astype('float32')


xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 10),
                     np.linspace(s_range[0][0], s_range[1][0], 10))

centers = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))).astype(theano.config.floatX)

widths = (s_range[1]-s_range[0]).astype(theano.config.floatX)*0.2

x,_,size = sym_RBF(s, s, ds, centers, widths, bias_term=True)
neurosftd = NeuroSFTD(x,
                      s,
                      ds, 
                      size, 
                      [], 
                      np.random.RandomState(1), 
                      alpha=0.05, 
                      alpha_mu=0.01, 
                      eta=0.0, 
                      beta_1=0.01, 
                      beta_2=0.05,
                      activations = [ None])

print 'learning...'
num_episode = 100
max_length = 10000
for states, rewards, next_states in generate_data(domain, policy, num_episode, max_length):
    for s_t, r_t, s_tp1 in izip(states, rewards, next_states):
        neurosftd.update(s_t, r_t, s_tp1)
        
print 'plotting...'
nsamples = 30        
xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 30),
                     np.linspace(s_range[0][0], s_range[1][0], 30))
points = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))).astype(theano.config.floatX)

values = neurosftd(points)

plt.figure()
plt.pcolormesh(xx, yy, values.reshape((nsamples,-1)))
plt.colorbar()
plt.show()


    