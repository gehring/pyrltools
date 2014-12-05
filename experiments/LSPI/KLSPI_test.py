from rltools.MountainCar import MountainCar, PumpingPolicy
from rltools.representation import StateActionProjection, Multiplied_kernel, Rbf_kernel, Poly_kernel, Partial_Kernel
from rltools.agent import LSPI, TransitionData, SFLSPI
from rltools.policy import Egreedy, SoftMax
from rltools.agent import maxValue
from itertools import product

import numpy as np

import matplotlib.pyplot as plt
from rltools.valuefn import KSFLSQ


def plot_valuefn(maxval, valuefn):
    min_pos = -1.2
    max_pos = 0.6

    max_speed = 0.07

    s_range = [ np.array([min_pos, -max_speed]),
           np.array([max_pos, max_speed])]
    x = np.linspace(s_range[0][0], s_range[1][0], 20)
    y = np.linspace(s_range[0][1], s_range[1][1], 20)
    xx, yy = np.meshgrid(x, y)
    samples = np.hstack((xx.reshape((-1, 1)), yy.reshape((-1,1))))
    v = np.empty(samples.shape[0], dtype='O')
    v[:] = [samples[i,:] for i in xrange(samples.shape[0])]
    v = maxval(v, valuefn)
    plt.contourf(x, y, v.reshape((x.shape[0],-1)))
    plt.colorbar()
    plt.show()

def plot_policy_valuefn(agent, policy, gamma, phi, regressor, maxval, **args):
    X_t, r_t, X_tp1 = agent.samples.constructMatricesfromPolicy(policy)
    valuefn = regressor(X_t, r_t, X_tp1, **args)
    plot_valuefn(maxval, valuefn)
    

gamma = 0.99


domain = MountainCar(True, 10000)
s_range = domain.state_range
actions = domain.discrete_actions
pump_policy = PumpingPolicy()
phi_sa = StateActionProjection(actions, size = 3)

def blank_valuefn(s, a= None):
    if a is None:
        return np.zeros(len(actions))
    else:
        return 0

def generate_samples(state_range, actions, domain, num_per_dim, phi_sa):
    states = [np.linspace(mi, ma, num_per_dim, True) for mi, ma in zip(*state_range)]
    states = [(np.array(sa[1:]), sa[0]) for sa in product(actions, *states)]
    
    
    sa_t = np.empty((len(states), phi_sa.size), dtype='float')
    s_tp1 = np.empty(len(states), dtype='O')
    r_t = np.empty(len(states), dtype='float')
    for i,(s, a) in enumerate(states):
        domain.state[:] = s
        r, next_s = domain.step(a)
        sa_t[i,:] = phi_sa(s,a)
        r_t[i] = r
        s_tp1[i] = next_s
    rnd_index = np.random.choice(len(states), len(states), replace = False)
    return (sa_t[rnd_index], r_t[rnd_index], s_tp1[rnd_index])

policy = Egreedy(actions, blank_valuefn, epsilon=0.05)

start_samples = None
print 'Building initial samples'
start_samples = generate_samples(s_range, actions, domain, 10, phi_sa)
print str(start_samples[0].shape[0]) +' initial samples obtained'

spsamples = TransitionData(start_samples, 
                                       phi_sa, 
                                       max_samples=1000,
                                       replace_data=False)
kernel = Multiplied_kernel(Partial_Kernel(np.arange(2, dtype = 'int'), Rbf_kernel(np.array([0.15, 0.01])*5)),
                           Partial_Kernel(np.ones(1, dtype = 'int')*2, Poly_kernel(2, 1)))
# kernel = Rbf_kernel(np.array([0.1, 0.01, 0.005])*4)
agent = SFLSPI(np.array(actions), 
             pump_policy, 
             gamma, 
             phi_sa, 
             valuefn = blank_valuefn , 
             samples = spsamples, 
             batch_size = 100,
             iteration_per_batch = 2, 
             improve_behaviour = False,
             method = KSFLSQ,
             kernel = kernel,
             lamb =0.01)



num_episodes = 2000
k = 1
valuefn = agent.valuefn
agent.maxval = np.vectorize(maxValue,
                                   otypes =[np.float],
                                   excluded = 'valuefn')
# agent.improve_policy(80)
for i in xrange(num_episodes):

    r_t, s_t = domain.reset()
    agent.reset()
    count = 0
    cumulative_reward = 0

    while s_t != None:
        # apply an action from the agent
        # the domain will return a 'None' state when terminating
        a_t=agent.step(r_t, s_t)
        r_t, s_t = domain.step(a_t)
        count += 1
        cumulative_reward += r_t
        
#         plot_policy_valuefn(agent, 
#                             pump_policy, 
#                             gamma, 
#                             phi_sa, 
#                             KSFLSQ, 
#                             agent.maxval, 
#                             kernel = kernel)
        
        if agent.valuefn is not valuefn:
            valuefn = agent.valuefn
            plot_valuefn(agent.argmaxval, agent.valuefn)
#
        

    # final update step for the agent
    agent.step(r_t, s_t)
    
#     if i % 2 == 0:
#         if render_value_fn:
#             plt.gca().clear()
#             plt.contourf(*getValueFn(valuefn))
#             plt.title('episode ' + str(i))
#             plt.savefig(file_path + str(k) + '.png')
#             k +=1

    # print cumulative reward it took to reach the goal
    # this should converge to around -120 for non-random start mountain car
    print cumulative_reward
    
