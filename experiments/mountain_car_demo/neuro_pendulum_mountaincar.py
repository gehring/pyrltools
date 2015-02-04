from rltools.theanotools import QNeuroSFTD, sym_RBF, Theano_RBF_Projector
from rltools.MountainCar import MountainCar, InifiniteMountainCar
from rltools.policy import Egreedy
from rltools.agent import TabularActionSarsa
from rltools.valuefn import LinearTD
from rltools.experiment import getAllRuns

import numpy as np
import theano
import matplotlib.pyplot as plt
import pickle
import os

from IPython.parallel import Client

import sys

# theano.config.compute_test_value = 'warn'

class rbf_input_layer(object):
    def __init__(self, centers, widths, bias_term):
        self.centers = centers
        self.widths = widths
        self.bias_term = bias_term
        
    def __call__(self, s, ds):
        return sym_RBF(s, s, ds, self.centers, self.widths, bias_term=self.bias_term)

def evaluate(valuefn, points):
    val = valuefn(points)
    return np.max(val,0)

def evaluate_all(valuefn, points):
    val = valuefn(points)
    return [v for v in val]


def run_exp(p):
    alpha, alpha_mu, eta, num_episodes, max_length, layers = p[['alpha', 'alpha_mu', 'eta', 'num_episodes', 'max_length', 'layers']]
    
    
    
    domain = InifiniteMountainCar(random_start=True, max_episode=1000)
    s_range = domain.state_range
    n_actions = len(domain.discrete_actions)
     
    xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 10),
                         np.linspace(s_range[0][1], s_range[1][1], 10))
     
    centers = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))).astype(theano.config.floatX)
     
    widths = (s_range[1]-s_range[0]).astype(theano.config.floatX)*0.085
    input_layer = None
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
    
    # phi = Theano_RBF_Projector(centers, widths, True, False)
    
    
    # valuefn = LinearTD(n_actions, phi, alpha, 0.0, 0.99, False)
    
#     policy = lambda x: np.random.randint(0, 3) if np.random.rand(1)<0.2 else (0 if x[1]<0 else 2)
    policy = Egreedy(np.arange(n_actions), valuefn, epsilon = 0.01)
    
    agent = TabularActionSarsa(domain.discrete_actions, policy, valuefn)
    
#     samples = 40
#     xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], samples),
#                          np.linspace(s_range[0][1], s_range[1][1], samples))
#      
#     points = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))).astype(theano.config.floatX)
     
    k = 1
    score = []
    for i in xrange(num_episodes):
        r_t, s_t = domain.reset()
        agent.reset()
        count = 0
        rew = 0.0
        for i in xrange(max_length):
            a =agent.step(r_t, s_t)
            r_t, s_t = domain.step(a)
            count += 1
            rew += r_t
        score.append(rew/count)
    
    return score, p


alphas = [0.01, 0.001, 0.0001]
alpha_mus = [0.01], #[0.01, 0.05, 0.001]
etas = [0.0]# [0.0, 0.3, 0.6]
num_episodes = [15]
max_length = [5000]
layers = [[200]]
num_runs = 5

params= {'alpha':alphas,
         'alpha_mu': alpha_mus,
         'eta': etas,
         'num_episodes': num_episodes,
         'max_length' : max_length,
         'layers' : layers,
         'num_runs': num_runs}

filename = '/media/cgehri/data/experiment_data/pendulum/test-100-integ-002--'

params = (alphas, alpha_mus, etas)

client = Client()
client[:].execute('from rltools.MountainCar import InifiniteMountainCar', block=True)
client[:].execute('from rltools.theanotools import QNeuroSFTD, sym_RBF, sym_NRBF', block=True)
client[:].execute('from itertools import izip', block=True)
client[:].execute('from rltools.policy import Egreedy', block=True)
client[:].execute('from rltools.agent import TabularActionSarsa', block=True)
client[:].execute('import theano', block=True)
client[:].execute('import numpy as np', block=True)

client[:]['rbf_input_layer'] = rbf_input_layer

lbview =  client.load_balanced_view()
lbview.block = False
lbview.retries = True

print 'Starting...'
results = lbview.map( run_exp, getAllRuns(params),
                     ordered = False,
                     block = False)

completed = []
print 'Just completed:'
for s in results:
    completed.append(s)
    print s[1]
    with open(filename + 'partial-data.data', 'wb') as f:
        pickle.dump((completed, params), f)
    
try:
    os.rename(filename + 'partial-data.data', filename + 'complete-data.data')
except Exception as e:
    print e
print 'Done!'

# alpha = 0.01
# alpha_mu = alpha
# eta = 0.0
# layers = [200]
# 
# domain = InifiniteMountainCar(random_start=True, max_episode=1000)
# s_range = domain.state_range
# n_actions = len(domain.discrete_actions)
#  
# xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 10),
#                      np.linspace(s_range[0][1], s_range[1][1], 10))
#  
# centers = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))).astype(theano.config.floatX)
#  
# widths = (s_range[1]-s_range[0]).astype(theano.config.floatX)*0.085
# input_layer = None
# input_layer = rbf_input_layer(centers, widths, False)
# neurosftd = QNeuroSFTD(2,
#                        n_actions, 
#                       layers, 
#                       np.random.RandomState(1),
#                       input_layer = input_layer, 
#                       alpha=alpha, 
#                       alpha_mu=alpha_mu, 
#                       eta=eta, 
#                       beta_1=0.0, 
#                       beta_2=0.0,
#                       activations = None)
# 
# valuefn = neurosftd
# 
# # phi = Theano_RBF_Projector(centers, widths, True, False)
# 
# 
# # valuefn = LinearTD(n_actions, phi, alpha, 0.0, 0.99, False)
# 
# policy = lambda x: np.random.randint(0, 3) if np.random.rand(1)<0.2 else (0 if x[1]<0 else 2)
# policy = Egreedy(np.arange(n_actions), valuefn, epsilon = 0.01)
# 
# agent = TabularActionSarsa(domain.discrete_actions, policy, valuefn)
# 
# samples = 40
# xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], samples),
#                      np.linspace(s_range[0][1], s_range[1][1], samples))
#  
# points = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))).astype(theano.config.floatX)
#  
# k = 1
# num_episodes = 10000
# for i in xrange(num_episodes):
#     r_t, s_t = domain.reset()
#     agent.reset()
#     count = 0
#     rew = 0.0
#     for i in xrange(5000):
# #         valuefn.mu = 0.0
#         a =agent.step(r_t, s_t)
#         r_t, s_t = domain.step(a)
#         count += 1
#         rew += r_t
# #     agent.step(0, s_t)         
#     print rew/count, valuefn.mu
#     k += 1
#     if (k%10)==0:
#         plt.figure()
#         vals = evaluate_all(valuefn, points)
#         for i,v in enumerate(vals):
#             plt.subplot(2,2,i+1)
#             plt.pcolormesh(xx, yy, v.reshape((samples,-1)))
#             plt.colorbar()
#         plt.subplot(2,2,4)
#         plt.pcolormesh(xx, yy, np.max(vals,0).reshape((samples,-1)))
#         plt.colorbar()
#         plt.show()
 
