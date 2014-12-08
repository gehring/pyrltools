from rltools.MountainCar import MountainCar, PumpingPolicy
from rltools.representation import TabularActionProjector
from rltools.representation import TileCoding
from rltools.agent import LSPI, BinarySparseTransitionData
from rltools.policy import Egreedy, SoftMax
from rltools.agent import maxValue
from itertools import product

from scipy.interpolate import griddata
import numpy as np

import matplotlib.pyplot as plt
from rltools.valuefn import LSQ,SFLSQ
from sklearn.cluster.tests.test_k_means import n_samples

def run_episode_from(s_t, domain, policy, gamma):
    rewards, _ = domain.reset()
    domain.state[:] = s_t
    i = 1
    while s_t is not None:
        r_t, s_t = domain.step(policy(s_t))
        rewards += r_t*gamma**i
        i+= 1
    return rewards

def estimate_value_fn_Monte_Carlo(domain, policy, gamma, n_samples=1, resolution=10):
    xx, yy = np.meshgrid(*[np.linspace(min, max, resolution, True)
                                for min, max in zip(*domain.state_range)])
    points = np.hstack((xx.reshape((-1, 1)), yy.reshape((-1, 1))))
    values = np.array([ np.mean([run_episode_from(s, domain, policy, gamma) for i in xrange(n_samples)])
                               for s in points])

    return lambda s: griddata(points, values, [s])

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

def plot_quiver(s_t, s_tp1):
    X,Y,U,V = np.array(zip(*[np.hstack((s,sp)) for s, sp in zip(s_t, s_tp1) if sp is not None]))
    plt.quiver(X, Y, U-X, V-Y)
    plt.show()

class NoAction(object):
    def __init__(self, phi):
        self.phi = phi
        self.size = phi.size
    def __call__(self, state, action=None):
        if action is None:
            return np.array([self.phi(state)])
        else:
            return self.phi(state)
gamma = 0.99



domain = MountainCar(True, 10000)
s_range = domain.state_range
actions = domain.discrete_actions

phi = TileCoding([np.arange(2)],
                 [10],
                 [10],
                 hashing=None,
                 state_range = s_range,
                 bias_term=True)

# phi_sa = TabularActionProjector(actions, phi)
phi_sa = NoAction(phi)

print phi_sa.size

def blank_valuefn(s, a= None):
    if a is None:
        return np.zeros(len(actions))
    else:
        return 0

def generate_samples(state_range, actions, domain, num_per_dim, phi_sa):
    states = [np.linspace(mi, ma, num_per_dim, True) for mi, ma in zip(*state_range)]
    states = [(np.array(sa[1:]), sa[0]) for sa in product(actions, *states)]

    sa_t = np.empty(len(states), dtype='O')
    s_tp1 = np.empty(len(states), dtype='O')
    r_t = np.empty(len(states), dtype='float')
    for i,(s, a) in enumerate(states):
        domain.state[:] = s
        r, next_s = domain.step(a)
        sa_t[i] = phi_sa(s,a)
        r_t[i] = r
        s_tp1[i] = next_s
    rnd_index = np.random.choice(len(states), len(states), replace = False)
    return (sa_t[rnd_index], r_t[rnd_index], s_tp1[rnd_index]), (states, s_tp1)

def dummy_max(v, val):
    return np.array([ val(s) for s in v])

policy =  PumpingPolicy() #Egreedy(actions, blank_valuefn, epsilon=0.05)


# valfn = estimate_value_fn_Monte_Carlo(domain, policy, gamma, 1, 50)
# plot_valuefn(dummy_max, valfn)


start_samples = None
# print 'Building initial samples'
# start_samples, (states, s_tp1) = generate_samples(s_range, actions, domain, 30, phi_sa)
# print str(start_samples[0].size) +' initial samples obtained'

spsamples = BinarySparseTransitionData(start_samples,
                                       phi_sa,
                                       max_samples=200000)
agent = LSPI(np.array(actions),
             policy,
             gamma,
             phi_sa,
             valuefn = blank_valuefn ,
             samples = spsamples,
             batch_size = 200000,
             iteration_per_batch = 1,
             improve_behaviour = False,
             method = SFLSQ)

num_episodes = 200000
k = 1
valuefn = agent.valuefn
agent.maxval = np.vectorize(maxValue,
                                   otypes =[np.float],
                                   excluded = 'valuefn')

# plot_quiver(zip(*states)[0], s_tp1)

# agent.improve_policy(1)
# plot_valuefn(agent.maxval, agent.valuefn)
# agent.improve_policy(1)
# plot_valuefn(agent.maxval, agent.valuefn)
# agent.improve_policy(10)
# plot_valuefn(agent.maxval, agent.valuefn)
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
        if agent.valuefn is not valuefn:
#             plot_valuefn(agent.argmaxval, agent.valuefn)
            plot_valuefn(agent.maxval, agent.valuefn)
            valuefn = agent.valuefn

    # final update step for the agent
    agent.step(r_t, s_t)


#
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