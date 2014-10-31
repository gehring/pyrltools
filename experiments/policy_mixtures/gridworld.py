from rltools.policy import policy_evaluation, SoftMax, Egreedy
from rltools.valuefn import LinearTDPolicyMixture, LinearTD, linearValueFn
from rltools.agent import LinearTabularPolicySarsa, TabularActionSarsa
from itertools import product
from rltools.GridWorld import GridWorld, boundary_condition
import numpy as np
import matplotlib.pyplot as plt

gamma = 0.9
alpha = 0.1
lamb = 0.5

dx = np.array([ [1,0],
                [-1,0],
                [0,1],
                [0,-1]], dtype='int32')

def getVectorField(valuefn):
    res=12
    U = np.empty(res**2)
    V = np.empty(res**2)
    x = np.linspace(0, 11, res)
    y = np.linspace(0, 11, res)
    for i,s in enumerate(product(x, y)):
        s = np.array(s)
        a = np.argmax([valuefn(s,0), valuefn(s,1), valuefn(s,2), valuefn(s,3)])
        U[i] = dx[a][0]
        V[i] = dx[a][1]
    X, Y = np.meshgrid(x, y)
    return X, Y, U.reshape((res,res)).T, V.reshape((res,res)).T

def viewValuefn(valuefn):
    res=12
    V = np.empty(res**2)
    x = np.linspace(0, 11, res)
    y = np.linspace(0, 11, res)
    for i,s in enumerate(product(x, y)):
        s = np.array(s)
        V[i] = valuefn(s)# max([valuefn(s,0), valuefn(s,1), valuefn(s,2), valuefn(s,3)])
    X, Y = np.meshgrid(x, y)
    return X, Y, V.reshape((res,res)).T

class TrivialPolicy(object):
    def __init__(self, size, action):
        self.action = action
        self.prob = np.zeros(size)
        self.prob[action] = 1.0
    def __call__(self, state):
        return self.action
    def getprob(self, state):
        return self.prob
    

pis = [TrivialPolicy(4, i) for i in range(4)]
middle = set()
for x in product(range(4,7), range(4,7)):
    s = np.array(x, dtype='int32')
    s.flags.writeable = False
    middle.add(s.data)

def reward(state, action=None):
    s = state.copy()
    s.flags.writeable = False
    if s.data in middle:
        return 1
    else:
        return 0

start_range = [np.zeros(2, dtype = 'int32'), np.ones(2, dtype = 'int32')*9]
boundary = boundary_condition(start_range)

islegal = lambda s: True

def terminal(state):
    return not boundary(state) #np.any(state < 0) or  np.any(state >9)

class phi(object):
    def __init__(self):
        self.size = 12*12+1
    def __call__(self, state):
        phi_t = np.zeros(self.size, dtype = 'int32')
        if state != None:
            state = np.array(state, dtype = 'int32')
            phi_t[np.ravel_multi_index(state, (12,12))] = 1
            phi_t[-1] = 1
        return phi_t




gridworld = GridWorld(reward,
                      islegal,
                      terminal,
                      start_range,
                      random_start = True,
                      max_episode = 20)
valuefns = [policy_evaluation([reward],
                              gamma,
                              pi,
                              gridworld,
                              projector = phi(),
                              method = 'LSTDlambda',
                              number_episodes = 1000,
                              max_episode_length = 20)[-1]
            for pi in pis]
valuefn = LinearTDPolicyMixture(4, phi(), alpha, lamb, gamma)
for i,v in enumerate(valuefns):
    valuefn.theta[i,:] = v.theta
# valuefn = LinearTD(4, phi(), alpha, lamb, gamma, True)
# policy = Egreedy(np.arange(4), valuefn, epsilon=0.05)
# agent = TabularActionSarsa(range(4), policy, valuefn)
mix_policy = Egreedy(np.arange(4), valuefn, epsilon = 0.05)
agent = LinearTabularPolicySarsa(range(4), mix_policy, pis, valuefn)

num_episodes = 20000
domain = gridworld
for i in xrange(num_episodes):
    r_t, s_t = domain.reset()
    agent.reset()
    count = 0
    cumulative_reward = 0

    while s_t != None:
        # apply an action from the agent
        # the domain will return a 'None' state when terminating
        r_t, s_t = domain.step(agent.step(r_t, s_t))
        count += 1
        cumulative_reward += r_t

    # final update step for the agent
    agent.step(r_t, s_t)

    # print cumulative reward it took to reach the goal
    # this should converge to around -120 for non-random start mountain car
#     print cumulative_reward
val =    policy_evaluation([reward],
                              gamma,
                              lambda x: agent.proposeAction(x),
                              gridworld,
                              projector = phi(),
                              method = 'LSTDlambda',
                              number_episodes = 1000,
                              max_episode_length = 20)[-1]
                              

plt.contourf(*viewValuefn(linearValueFn(valuefn.theta[0,:], phi())))
plt.quiver(*getVectorField(valuefn))
plt.show()
    

