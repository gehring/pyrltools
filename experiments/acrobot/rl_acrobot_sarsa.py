from rltools.representation import TileCoding, UNH
from rltools.valuefn import LinearTD
from rltools.agent import TabularActionSarsa
from rltools.policy import Egreedy
from rltools.acrobot import Acrobot
import pickle
import numpy as np
from itertools import product
input_indicies = [np.arange(4),
                np.array([0,1,2]),
                np.array([0,1,3]),
                np.array([3,1,2]),
                np.array([0,3,2]),
                np.array([0,1]),
                np.array([0,2]),
                np.array([0,3]),
                np.array([1,2]),
                np.array([1,3]),
                np.array([2,3]),
                np.array([0]),
                np.array([1]),
                np.array([2]),
                np.array([3])]

dt = 0.05
domain = Acrobot(random_start= False,
                 max_episode = 10/dt-1,
                  m1 = 1, m2 = 1, l1 = 1, l2=2, b1=0.1, b2=0.1)
domain.dt[-1] = dt

s_range = domain.state_range
a_range = domain.action_range

sa_range = [np.hstack((s_range[0], a_range[0])),
            np.hstack((s_range[1], a_range[1]))]

tiles_all = np.array([6,6,7,7])
ntiles = [ tiles_all[i] for i in input_indicies]
 
ntilings = [24,
            3,
            3,
            3,
            3,
            2,
            2,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3]
memories = [ np.prod(nt)/3 + 100 for nt in ntiles]
hashing = [UNH(m) for m in memories]

class IdenStateAction(object):
    def __init__(self, proj, actions):
        self.proj = proj
        self.actions = actions
        self.size = proj.size
        
    def __call__(self, s, a = None):
        if s is None:
            return None
        if a is None:
            return np.vstack([ self.proj(np.hstack((s,act))) for act in self.actions])
        else:
            return self.proj(np.hstack((s,a)))
        
 
phi = TileCoding(input_indicies,
                 ntiles,
                 ntilings,
                 hashing,
                 sa_range,
                 bias_term = True)



act = [ np.array(a) for a in np.linspace(-10, 10, 5, True)]

alpha = 0.01/(sum(ntilings)+1)
gamma = 0.99
lamb = 0.99
print phi.size

valuefn = LinearTD(len(act), phi, alpha, lamb, gamma, True)
policy = Egreedy(np.arange(len(act)), valuefn, epsilon = 0.0)

agent = TabularActionSarsa(act, policy, valuefn)



num_episodes = 50000
# namein= 'test4'
# with open('agent-'+namein+'.data', 'rb') as f:
#     (phi, valuefn, policy,agent) = pickle.load(f)
nameout='sarsa05'

thres = np.pi/4
domain.goal_range = [np.array([np.pi - thres, -thres, -thres, -2*thres]),
                  np.array([np.pi + thres, thres, thres, 2*thres])]

w = np.array([8,8,4,4])
p = np.array([np.pi, np.pi])

for i in xrange(num_episodes):
    domain.start_state[:] = [np.pi-(0.05)*(np.random.rand(1)-0.5),
                             (np.random.rand(1)-0.5)*0.01,
                             (np.random.rand(1)-0.5)*0.01,
                             (np.random.rand(1)-0.5)*0.01]
    r_t, s_t = domain.reset()
    agent.reset()
    count = 0
    cumulative_reward = 0

    while s_t != None:
        # apply an action from the agent
        # the domain will return a 'None' state when terminating
        a_t = agent.step(r_t, s_t)
        r_t, s_t = domain.step(a_t)
        count += 1
        cumulative_reward += r_t

    # final update step for the agent
    agent.step(r_t, s_t)

#     if np.any(np.isnan(valuefn.theta)):
#         print 'NaN found, abort!'
#         break

    if i%10 == 0:
        with open('agent-'+nameout+'.data', 'wb') as f:
            pickle.dump((phi, policy, agent), f)

    # print cumulative reward it took to reach the goal
    # this should converge to around -120 for non-random start mountain car
    print cumulative_reward

