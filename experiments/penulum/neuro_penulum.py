from rltools.theanotools import NeuroSFTD, sym_RBF, sym_NRBF
from rltools.SwingPendulum import SwingPendulum, Swing_stabilize

from itertools import izip, product, chain, repeat

import theano
import theano.tensor as T

import numpy as np

import matplotlib.pyplot as plt
from IPython.parallel import Client
import pickle
import os
import sys

# theano.config.compute_test_value = 'warn'
# theano.config.exception_verbosity = 'high'

def sample_true_value(p):
    domain, policy, max_length, start_points = p
    rew = []
    for start in start_points:
        domain.reset()
        domain.state[:] = start
        r = 0
        s = start
        while s is not None:
            r_t, s = domain.step(policy(s), policy)
            r += r_t
        rew.append(r)
    return domain.control_rate, np.array(rew)

def approx_avg_rew(p):
    domain, policy, num_episodes, max_length = p
    rewards = []
    for _, rew, _  in generate_data(domain, policy, num_episodes, max_length):
        rewards.append( rew)
    return domain.control_rate, np.mean(np.hstack(rewards))

def generate_traj(domain, policy, max_length):
    r, s = domain.reset()
    states = []
    rewards = []
    next_states = []
    for i in xrange(max_length):
        states.append(s)
        r, s = domain.step(policy(s), policy)
        rewards.append(r)
        next_states.append(s)
        if s is None:
            break
    next_states[-1] = None
    return states, rewards, next_states

def generate_data(domain, policy, num, max_length):
    for i in xrange(num):
        yield generate_traj(domain, policy, max_length)
        

class rbf_input_layer(object):
    def __init__(self, centers, widths, bias_term):
        self.centers = centers
        self.widths = widths
        self.bias_term = bias_term
        
    def __call__(self, s, ds):
        return sym_RBF(s, s, ds, self.centers, self.widths, bias_term=self.bias_term)
def run_exp(p):
    control_rate, alpha, alpha_mu, eta, num_episode, max_length, layers, points = p
    domain = SwingPendulum(random_start=True)
    domain.integ_rate = 0.002
    domain.control_rate = control_rate
    policy = Swing_stabilize(domain)
    s_range = domain.state_range
     
     
    xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 10),
                         np.linspace(s_range[0][1], s_range[1][1], 10))
     
    centers = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))).astype(theano.config.floatX)
     
    widths = (s_range[1]-s_range[0]).astype(theano.config.floatX)*0.085
    input_layer = rbf_input_layer(centers, widths, False)
    neurosftd = NeuroSFTD(2, 
                          layers, 
                          np.random.RandomState(1),
                          input_layer = input_layer, 
                          alpha=alpha, 
                          alpha_mu=alpha_mu, 
                          eta=eta, 
                          beta_1=0.0, 
                          beta_2=0.0,
                          activations = None)
     
#     print 'learning...'
    for states, rewards, next_states in generate_data(domain, policy, num_episode, int(max_length/control_rate)):
        for s_t, r_t, s_tp1 in izip(states, rewards, next_states):
            neurosftd.update(s_t, r_t, s_tp1)
             
    
     
    values = neurosftd(points)
    return values, {'rate':control_rate, 
                    'alpha':alpha, 
                    'alpha_mu':alpha_mu, 
                    'eta':eta, 
                    'num_episode':num_episode, 
                    'max_length':max_length, 
                    'layers':layers}
        
control_rate = [0.04, 0.02, 0.01, 0.006, 0.002]
alphas = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
alpha_mus = [0.01, 0.05, 0.001, 0.0005]
etas = [0.0, 0.3, 0.6, 0.9]
num_episodes = 100
max_length = 100
layers = [100]

filename = '/media/cgehri/data/experiment_data/pendulum/test-100-integ-002--'

params = (control_rate, alphas, alpha_mus, etas)

domain = SwingPendulum(random_start=True)
s_range = domain.state_range
nsamples = 40        
xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], nsamples),
                     np.linspace(s_range[0][1], s_range[1][1], nsamples))
points = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))).astype(theano.config.floatX)


# avg_rew={}
# true_val = {}
# for cr in control_rate:
#     domain = SwingPendulum(random_start=True)
#     domain.control_rate = cr
#     policy = Swing_stabilize(domain)
#     avg_rew[cr] = approx_avg_rew(domain, policy, 100, int(max_length/cr))
#     true_val[cr] = sample_true_value(domain, policy,  int(max_length/cr), points)
# 
# sys.exit()




client = Client()
client[:].execute('from rltools.SwingPendulum import SwingPendulum, Swing_stabilize', block=True)
client[:].execute('from rltools.theanotools import NeuroSFTD, sym_RBF, sym_NRBF', block=True)
client[:].execute('from itertools import izip', block=True)
client[:].execute('import theano', block=True)
client[:].execute('import theano.tensor as T', block=True)
client[:].execute('import numpy as np', block=True)

client[:]['generate_traj'] = generate_traj
client[:]['generate_data'] = generate_data
client[:]['sample_true_value'] = sample_true_value
client[:]['approx_avg_rew'] = approx_avg_rew
client[:]['rbf_input_layer'] = rbf_input_layer

lbview =  client.load_balanced_view()
lbview.block = False
lbview.retries = True

compare_domains = []
max_lengths = []
for cr in control_rate:
    domain = SwingPendulum(random_start=True)
    domain.control_rate = cr
    policy = Swing_stabilize(domain)
    compare_domains.append(domain)
    max_lengths.append(int(max_length/cr))
    
num_of_samples = 1000
print 'Computing average rewards...'
avg_rew = lbview.map(approx_avg_rew, izip(compare_domains,
                                          repeat(policy),
                                          repeat(num_of_samples),
                                          max_lengths),
                     ordered = True,
                     block = True)

print 'Computing True Value functions...'
true_val = lbview.map(sample_true_value, izip(compare_domains,
                                          repeat(policy),
                                          max_lengths,
                                          repeat(points)),
                     ordered = True,
                     block = True)
avg_rew = dict(avg_rew)
true_val = dict(true_val)
print 'Empirical Truth computed!'

results = lbview.map( run_exp, product(control_rate,
                                        alphas,
                                        alpha_mus,
                                        etas,
                                        [num_episodes],
                                        [max_length],
                                        [layers],
                                        [points]),
                     ordered = False,
                     block = False)

completed = []

print 'Starting...'
print 'Just completed:'
for s in results:
    completed.append(s)
    print s[1]
    with open(filename + 'partial-data.data', 'wb') as f:
        pickle.dump((completed, avg_rew, true_val, params), f)
    
try:
    os.rename(filename + 'partial-data.data', filename + 'complete-data.data')
except Exception as e:
    print e
print 'Done!'
        
    
    