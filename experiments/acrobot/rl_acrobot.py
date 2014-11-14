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
dt = 0.005
domain = Acrobot(random_start= False,
                 max_episode = 20/dt-1,
                  m1 = 1, m2 = 1, l1 = 1, l2=2, b1=0.1, b2=0.1)
domain.dt[-1] = dt

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
 
phi = TileCoding(input_indicies,
                 ntiles,
                 ntilings,
                 None,
                 domain.state_range,
                 bias_term = True)

act = [ np.array(a) for a in np.linspace(-10, 10, 13, True)]
valuefn = LinearTD(len(act), phi, 0.2/48, 0.99, 0.999)
policy = Egreedy(np.arange(len(act)), valuefn, epsilon = 0.0)
agent = TabularActionSarsa(act, policy, valuefn)



num_episodes = 50000
namein= 'test4'
with open('agent-'+namein+'.data', 'rb') as f:
    (phi, valuefn, policy,agent) = pickle.load(f)
nameout='test5'
valuefn.alpha = 0.01/80
valuefn.lamb = 0.9

thres = np.pi/6
domain.goal_range = [np.array([np.pi - thres, -thres, -thres, -thres]),
                  np.array([np.pi + thres, thres, thres, thres])]

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
        r_t, s_t = domain.step(agent.step(r_t/2, s_t))
        count += 1
        cumulative_reward += r_t

    # final update step for the agent
    agent.step(r_t/2, s_t)

    if np.any(np.isnan(valuefn.theta)):
        print 'NaN found, abort!'
        break

    if i%10 == 0:
        with open('agent-'+nameout+'.data', 'wb') as f:
            pickle.dump((phi, valuefn, policy, agent), f)

    # print cumulative reward it took to reach the goal
    # this should converge to around -120 for non-random start mountain car
    print cumulative_reward

