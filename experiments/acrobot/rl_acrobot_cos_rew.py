from rltools.representation import TileCoding
from rltools.agent import TabularActionSarsa
from rltools.valuefn import LinearTD
from rltools.policy import Egreedy
from rltools.acrobot import Acrobot
import pickle
import numpy as np
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

domain = Acrobot(random_start= False,
                 max_episode = 10000,
                  m1 = 1, m2 = 1, l1 = 1, l2=2, b1=0.1, b2=0.1)
domain.dt[-1] = 0.05

tiles_all = np.array([6,6,7,7])
ntiles = [ tiles_all[i] for i in input_indicies]
 
ntilings = [12,
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
 
phi = TileCoding(input_indicies,
                 ntiles,
                 ntilings,
                 None,
                 domain.state_range,
                 bias_term = True)
valuefn = LinearTD(len(domain.discrete_actions), phi, 0.2/48, 0.8, 0.9999)
policy = Egreedy(np.arange(3), valuefn, epsilon = 0.0)
agent = TabularActionSarsa(domain.discrete_actions, policy, valuefn)


num_episodes = 400
# namein= 'test5'
# with open('agent-'+namein+'.data', 'rb') as f:
#     (phi, valuefn, policy,agent) = pickle.load(f)
nameout='cosr2'
valuefn.alpha = 0.2/48

thres = 0.01
domain.goal_range = [np.array([np.pi - thres, -thres, -thres, -thres]),
                  np.array([np.pi + thres, thres, thres, thres]),]

for i in xrange(num_episodes):

    r_t, s_t = domain.reset()
    agent.reset()
    count = 0
    cumulative_reward = 0

    while s_t != None:
        # apply an action from the agent
        # the domain will return a 'None' state when terminating
        r_t = -np.cos(s_t[0]) + min(np.cos(s_t[1]), 0.4)
        cumulative_reward += r_t
        r_t, s_t = domain.step(agent.step(r_t, s_t))
        
        count += 1
        

    # final update step for the agent
    agent.step(0, s_t)

    if i%5 == 0:
        with open('agent-'+nameout+'.data', 'wb') as f:
            pickle.dump((phi, valuefn, policy, agent), f)

    # print cumulative reward it took to reach the goal
    # this should converge to around -120 for non-random start mountain car
    print cumulative_reward
