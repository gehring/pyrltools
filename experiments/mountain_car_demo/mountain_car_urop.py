from itertools import product
from mountaincar import MountainCar
from tilecoding import TileCoding
from swingpendulum import SwingPendulum

import mountaincar
import matplotlib.pyplot as plt
import numpy as np

class Egreedy(object):
    def __init__(self, num_actions, valuefn, epsilon):
        self.value_fn = valuefn
        self.epsilon = epsilon
        self.num_actions = num_actions

    def __call__(self, state):
        # TODO: implement an e-greedy policy
        # this method should return a sampled action with respect to the policy
        action_index = None
        return action_index

class LinearTD(object):
    def __init__(self,
                 num_actions,
                 projector,
                 alpha,
                 gamma,
                 **argk):
        # discount factor
        self.gamma = gamma

        # learning rate
        self.alpha = alpha

        # function that returns feature vectors given a state
        self.phi = projector

        # parameters of the TD, each action have their own parameters
        self.theta = np.zeros((num_actions, projector.size))

    def __call__(self, state, action):
        if state == None:
            return 0
        else:
            return self.phi(state).dot(self.theta[action,:])

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if s_t == None:
            # beginning of episode: do nothing
            return
        else:
            # TODO: fill in update rules for TD

            # you can use self(s,a) to get the current approximation of the
            # value function at state s and action a

            # delta \leftarrow r + \gamma * Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)
            # theta_a \leftarrow theta_a + \alpha * delta * grad_{\theta_a}(Q(s_t,a_t))
            return

class TabularActionSarsa(object):
    def __init__(self, actions, policy, valuefn, **argk):
        self.policy = policy
        self.valuefn = valuefn
        self.s_t = None
        self.a_t = None
        self.actions = actions

    def step(self, r, s_tp1):
        if s_tp1 != None:
            a_tp1 = self.policy(s_tp1)
        else:
            a_tp1 = None

        self.valuefn.update(self.s_t, self.a_t, r, s_tp1, a_tp1)

        self.s_t = s_tp1
        self.a_t = a_tp1

        return self.actions[a_tp1] if a_tp1 != None else None

    def reset(self):
        self.s_t = None
        self.a_t = None

    def proposeAction(self, state):
        return self.actions[self.policy(state)]

domain = MountainCar(random_start= False, max_episode=10000)
# domain = SwingPendulum(random_start= False, max_episode_length=10000)

phi = TileCoding(input_indicies = [np.array([0,1])],
                 ntiles = [10],
                 ntilings=[10],
                 state_range = domain.state_range,
                 bias_term = True)

valuefn = LinearTD(len(domain.discrete_actions),
                   phi,
                   alpha = 0.01,
                   gamma= 0.995)

# this is a sub-optimal policy for mountain car that naively always
# increases the energy of the system. You can swap the e-greedy policy
# with this one to debug the your implementation of TD and e-greedy
pumping_policy = mountaincar.PumpingPolicy_index()

policy = Egreedy(len(domain.discrete_actions), valuefn, epsilon = 0.05)

# this agent picks the actions and receives sample transitions
agent = TabularActionSarsa(domain.discrete_actions, policy, valuefn)



def getValueFn(valuefn):
    val = np.empty(10000)
    x = np.linspace(domain.state_range[0][0], domain.state_range[1][0], 100)
    y = np.linspace(domain.state_range[0][1], domain.state_range[1][1], 100)
    for i,s in enumerate(product(x, y)):
        s = np.array(s)
        val[i] = max([valuefn(s,0), valuefn(s,1), valuefn(s,2)])
    X, Y = np.meshgrid(x, y)
    return X, Y, val.reshape((100,100)).T

render_value_fn = True

file_path = './'

if render_value_fn:
    plt.contourf(*getValueFn(valuefn))
    plt.title('initial')
    plt.savefig(file_path + '0.png')

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

    if i % 2 == 0:
        if render_value_fn:
            plt.gca().clear()
            plt.contourf(*getValueFn(valuefn))
            plt.title('episode ' + str(i))
            plt.savefig(file_path + str(k) + '.png')
            k +=1

    # print cumulative reward it took to reach the goal
    # this should converge to around -120 for non-random start mountain car
    print cumulative_reward

