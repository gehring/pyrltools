from rltools.SwingPendulum import SwingPendulum, Swing_stabilize
from rltools.theanotools import Theano_Tiling, Theano_RBF_Projector
from rltools.agent import TabularActionSarsa
from rltools.policy import Egreedy
from rltools.valuefn import TDSR, LinearTD
from rltools.representation import TabularActionProjector
from rltools.MountainCar import MountainCar

from scipy.sparse import csc_matrix

import theano

import matplotlib.pyplot as plt

import numpy as np
import scipy.sparse as sp

class sparse_tilecoding(object):
    def __init__(self, tilecoding, num_actions):
        self.tilecoding = tilecoding
        self.num_actions = num_actions
        self.size = num_actions * tilecoding.size
    
    def __call__(self, state, action = None):
        index = self.tilecoding(state)
        if action is None:
            phi = csc_matrix((np.ones(index.size), np.vstack((index, np.zeros_like(index)))),
                             shape = (self.tilecoding.size, 1))
            phi_sa = sp.kron(phi, sp.eye(self.num_actions), format = 'csc')
        else:
            phi_sa = csc_matrix((np.ones(index.size), 
                                np.vstack(((index+self.tilecoding.size*action),
                                 np.zeros_like(index)))),
                                shape = (self.size, 1))
        return phi_sa
    

def evaluate_valuefn(valuefn, points):
    values = np.empty(points.shape[0])
    for i, p in enumerate(points):
        values[i] = valuefn(p).max()
    return values
    
    
# domain = SwingPendulum(random_start=True)
# domain.control_rate = 0.01
domain = MountainCar(random_start=True, max_episode=1000)
s_range = domain.state_range

tiling = Theano_Tiling(input_indicies = [np.arange(2)], 
                       ntiles = [10], 
                       ntilings = [10], 
                       hashing = None, 
                       state_range = s_range, 
                       bias_term = True)
 
phi_sa = sparse_tilecoding(tiling, 3)

xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 10),
                         np.linspace(s_range[0][1], s_range[1][1], 10))
       
centers = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))).astype(theano.config.floatX)
   
widths = (s_range[1]-s_range[0]).astype(theano.config.floatX)*0.085


phi_sa = TabularActionProjector(domain.discrete_actions, 
                                Theano_RBF_Projector(centers=centers, 
                                                     widths=widths, 
                                                     bias_term=True, 
                                                     normalized=False))




phi = Theano_RBF_Projector(centers=centers, 
                                                     widths=widths, 
                                                     bias_term=True, 
                                                     normalized=False)

alpha = 0.1
alpha_R = 0.01
lamb = 0.6
gamma = 0.99
rank = 10


valuefn = TDSR(phi_sa, 
               alpha = alpha, 
               alpha_R = alpha_R, 
               lamb = lamb, 
               gamma = gamma, 
               rank = rank,
               replacing_trace = False,
               use_U_only=False)

valuefn = LinearTD(3, phi, alpha, lamb, gamma, replacing_trace=False)

policy = Egreedy(np.arange(3), valuefn, epsilon = 0.05)
# policy = Swing_stabilize(domain)

agent = TabularActionSarsa(actions=domain.discrete_actions, 
                           policy=policy, 
                           valuefn=valuefn)
# 
# agent = TabularActionSarsa(actions=[0], 
#                            policy=lambda s: 0, 
#                            valuefn=valuefn)

nsamples = 100        
xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], nsamples),
                     np.linspace(s_range[0][1], s_range[1][1], nsamples))
points = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1))))
values = []



num_ep = 20
max_length = 100000
print 'starting learning...'
count = 0
for i in xrange(num_ep):
    r, s = domain.reset()
    rewards = []
#     n1 = np.linalg.norm(valuefn.R)
    for j in xrange(max_length):
        r, s = domain.step(agent.step(r, s))
#         r, s = domain.step(policy)
#         agent.step(r, s)
        rewards.append(r)
#         if (count%1000) == 0:
#             valuefn.rank = min(50, valuefn.rank + 1)
        count +=1 
       
        if s is None:
            a = agent.step(r, s)
#             if (count%1000) == 0:
#                 valuefn.rank = min(50, valuefn.rank + 1)
            count += 1
            break
#     valuefn.correct_orthogonality()
#     print 'n', n1, np.linalg.norm(valuefn.R)
    if i % (num_ep/20) == 0:
        print 'episode ' + str(i) + ' is done ', np.sum(rewards)
        values.append(evaluate_valuefn(valuefn, points))
        

plt.figure()
for i, val in enumerate(values[:18:2]):
    plt.subplot(3,3,i+1)
    plt.pcolormesh(xx, yy, val.reshape((nsamples,-1)))
    plt.colorbar()
    plt.xlabel('$theta$')
    plt.ylabel('$\dot{theta}$')
plt.show()