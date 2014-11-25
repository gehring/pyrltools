from rltools.MountainCar import MountainCar
from rltools.representation import TabularActionProjector
from rltools.representation import TileCoding
from rltools.agent import LSPI, BinarySparseTransitionData
from rltools.policy import Egreedy
from rltools.agent import maxValue

import numpy as np

import matplotlib.pyplot as plt



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

phi_sa = TabularActionProjector(actions, phi)

def blank_valuefn(s, a= None):
    if a is None:
        return np.zeros(len(actions))
    else:
        return 0

policy = Egreedy(actions, blank_valuefn, epsilon=0.05)
spsamples = BinarySparseTransitionData(None, 
                                       phi_sa, 
                                       max_samples=100000)
agent = LSPI(np.array(actions), 
             policy, 
             gamma, 
             phi_sa, 
             valuefn = blank_valuefn , 
             samples = spsamples, 
             batch_size = 10000, 
             improve_behaviour = True)

num_episodes = 2000
k = 1
valuefn = agent.valuefn
agent.maxval = np.vectorize(maxValue,
                                   otypes =[np.float],
                                   excluded = 'valuefn')
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
        

    # final update step for the agent
    agent.step(r_t, s_t)
    if agent.valuefn is not valuefn:
            valuefn = agent.valuefn
    # debug stuff, to delete
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
            v = agent.maxval(v, agent.valuefn)
            plt.contourf(x, y, v.reshape((x.shape[0],-1)))
            plt.colorbar()
            plt.show()
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