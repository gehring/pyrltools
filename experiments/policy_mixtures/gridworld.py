from rltools.policy import policy_evaluation, SoftMax_mixture
from itertools import product
from rltools.GridWorld import GridWorld, boundary_condition
import numpy as np
import matplotlib.pyplot as plt

gamma = 0.9

pis = [lambda s: i for i in range(4)]

middle = set()
for x in product(range(4,7), range(4,7)):
    s = np.array(x, dtype='int32')
    s.flags.writeable = False
    middle |= s

def reward(state, action=None):
    s = state.copy()
    s.flags.writeable = False
    if state in middle:
        return 1
    else:
        return 0

start_range = [np.zeros(2, dtype = 'int32'), np.ones(2, dtype = 'int32')*9]
boundary = boundary_condition(start_range)

islegal = lambda s: boundary(s)

def terminal(state):
    s = state.copy()
    s.flags.writeable = False
    if state in middle:
        return True
    else:
        return False


def phi(state):
    phi_t = np.zeros(101, dtype = 'int32')
    phi_t[np.ravel_multi_index(state, (10,10))] = 1
    phi_t[-1] = 1
    return phi_t

gridworld = GridWorld(reward,
                      islegal,
                      terminal,
                      start_range,
                      random_start = True,
                      max_episode = 10)
valuefns = [policy_evaluation([reward],
                              gamma,
                              pi,
                              gridworld,
                              projector = phi,
                              method = 'LSTDlambda',
                              number_episodes = 1)[-1]
            for pi in pis]
pi_mix = SoftMax_mixture(valuefns, pis)
value_mix = [policy_evaluation([reward],
                              gamma,
                              pi_mix,
                              gridworld,
                              projector = phi,
                              method = 'LSTDlambda',
                              number_episodes = 1)[-1]]

val =[np.zeros((10,10)) for i in range(5)]
for x,y in product(range(10), range(10)):
    state = np.array([x,y], dtype='int32')
    for v, fn in zip(val, valuefns+value_mix):
        v[x,y] = fn(state)

fig, axes = plt.subplot(1, 5)
for ax, v, i in zip(axes, val, range(5)):
    ax.imshow(v, interpolation = 'nearest')
    ax.set_title(str(i) if i != 4 else 'mixture')
plt.show()
