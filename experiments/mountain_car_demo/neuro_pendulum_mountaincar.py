from rltools.theanotools import QNeuroSFTD, sym_RBF
from rltools.MountainCar import MountainCar
from rltools.policy import Egreedy
from rltools.agent import TabularActionSarsa

import numpy as np
import theano

theano.config.compute_test_value = 'warn'

class rbf_input_layer(object):
    def __init__(self, centers, widths, bias_term):
        self.centers = centers
        self.widths = widths
        self.bias_term = bias_term
        
    def __call__(self, s, ds):
        return sym_RBF(s, s, ds, self.centers, self.widths, bias_term=self.bias_term)


alpha = 0.001
alpha_mu = 0.005
eta = 0.3
layers = []

domain = MountainCar(random_start=True, max_episode=4000)
s_range = domain.state_range
n_actions = len(domain.discrete_actions)
 
xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 10),
                     np.linspace(s_range[0][0], s_range[1][0], 10))
 
centers = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))).astype(theano.config.floatX)
 
widths = (s_range[1]-s_range[0]).astype(theano.config.floatX)*0.085
input_layer = rbf_input_layer(centers, widths, False)
neurosftd = QNeuroSFTD(2,
                       n_actions, 
                      layers, 
                      np.random.RandomState(1),
                      input_layer = input_layer, 
                      alpha=alpha, 
                      alpha_mu=alpha_mu, 
                      eta=eta, 
                      beta_1=0.0, 
                      beta_2=0.0,
                      activations = None)

valuefn = neurosftd

policy = Egreedy(np.arange(n_actions), valuefn)
agent = TabularActionSarsa(domain.discrete_actions, policy, valuefn)
 
k = 1
num_episodes = 100
for i in xrange(num_episodes):
    r_t, s_t = domain.reset()
    agent.reset()
    count = 0
    while s_t != None:
        r_t, s_t = domain.step(agent.step(r_t, s_t))
        count += 1
    agent.step(r_t, s_t)         
    print count
 
