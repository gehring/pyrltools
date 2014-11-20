from rltools.agent import FittedQIteration
from rltools.policy import Egreedy
from rltools.representation import TileCodingDense
from rltools.theano import MLPfit
from rltools.valuefn import MLPValueFn
from rltools.MountainCar import MountainCar

import numpy as np

rng = np.random.RandomState(312)

gamma = 0.95

domain = MountainCar(True, 1000)
a_range = domain.action_range
s_range = domain.state_range
sa_range = [np.hstack((s_range[0], a_range[0])),
            np.hstack((s_range[1], a_range[1]))]
actions = np.arange(a_range[0], a_range[1], 0.2)

def getMLPValueFn(x, y):
    mlp = MLPfit(0.01, 
                 100, 
                 rng, 
                 0.0001, 
                 0.0, 
                 x, 
                 y, 
                 100, 
                 0.2, 
                 5000,
                 4)
    return MLPValueFn(mlp, actions)


def blank_valuefn(s, a= None):
    if a is None:
        return np.zeros(len(actions))
    else:
        return 0

phi = TileCodingDense(input_indicies = [0,1,2],
                 ntiles=[10],
                 ntilings=[10],
                 hashing=None,
                 state_range= sa_range,
                 bias_term = True)


policy = Egreedy(actions, blank_valuefn)
agent = FittedQIteration(actions, 
                         policy, 
                         phi, 
                         getMLPValueFn, 
                         gamma, 
                         valuefn= None, 
                         samples= None, 
                         batch_size= 500, 
                         max_samples= 10000, 
                         dtype= theano.config.floatX, 
                         improve_behaviour= True)

num_episodes = 500
k = 1
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

    