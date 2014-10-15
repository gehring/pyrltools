from rltools.policy import policy_evaluation, SoftMax_mixture
from itertools import product
import numpy as np

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

islegal = lambda s: True

def terminal(state):
    s = state.copy()
    s.flags.writeable = False
    if state in middle:
        return True
    else:
        return False

gridworld = None
valuefns = [policy_evaluation(reward,
                              gamma,
                              pi,
                              gridworld,
                              'LSTDlambda')
            for pi in pis]