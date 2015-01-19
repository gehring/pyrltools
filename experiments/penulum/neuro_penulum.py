from rltools.theanotools import NeuroSFTD, sym_RBF, sym_NRBF
from rltools.SwingPendulum import SwingPendulum, Swing_stabilize

from itertools import izip, product

import theano
import theano.tensor as T

import numpy as np

import matplotlib.pyplot as plt
from IPython.parallel import Client
import pickle

# theano.config.compute_test_value = 'warn'
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
    control_rate, alpha, alpha_mu, eta, num_episode, max_length, layers = p
    domain = SwingPendulum(random_start=True)
    domain.control_rate = control_rate
    policy = Swing_stabilize(domain)
    s_range = domain.state_range
     
     
    xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], 10),
                         np.linspace(s_range[0][0], s_range[1][0], 10))
     
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
             
    nsamples = 40        
    xx, yy = np.meshgrid(np.linspace(s_range[0][0], s_range[1][0], nsamples),
                         np.linspace(s_range[0][0], s_range[1][0], nsamples))
    points = np.hstack((xx.reshape((-1,1)), yy.reshape((-1,1)))).astype(theano.config.floatX)
     
    values = neurosftd(points)
    return values, (control_rate, alpha, alpha_mu, eta, num_episodes, max_length, layers)
        
control_rate = [0.1, 0.05]#[0.2, 0.1, 0.05, 0.01]
alphas = [0.01]#[0.05, 0.01, 0.005]
alpha_mus = [0.01]#[0.01, 0.001]
etas = [0.3]#[0.0, 0.3, 0.6, 0.9]
num_episodes = 100
max_length = 100
layers = []


client = Client()
client[:].execute('from rltools.SwingPendulum import SwingPendulum, Swing_stabilize', block=True)
client[:].execute('from rltools.theanotools import NeuroSFTD, sym_RBF, sym_NRBF', block=True)
client[:].execute('from itertools import izip', block=True)
client[:].execute('import theano', block=True)
client[:].execute('import theano.tensor as T', block=True)
client[:].execute('import numpy as np', block=True)

client[:]['generate_traj'] = generate_traj
client[:]['generate_data'] = generate_data
client[:]['rbf_input_layer'] = rbf_input_layer

lbview =  client.load_balanced_view()
lbview.block = False
lbview.retries = True

map(run_exp, product(control_rate,
                                        alphas,
                                        alpha_mus,
                                        etas,
                                        [num_episodes],
                                        [max_length],
                                        [layers]))

results = lbview.map( run_exp, product(control_rate,
                                        alphas,
                                        alpha_mus,
                                        etas,
                                        [num_episodes],
                                        [max_length],
                                        [layers]),
                     ordered = False,
                     block = False)

completed = []

print 'Starting!'
print 'Just completed:'
for s in results:
    completed.append(s)
    print s[1]
    with open('partial-data.data', 'wb') as f:
        pickle.dump(completed, f)
    
with open('complete-data.data', 'wb') as f:
        pickle.dump(completed)

        
    
    