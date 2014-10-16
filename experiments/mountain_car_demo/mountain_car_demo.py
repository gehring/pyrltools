from rltools.MountainCar import MountainCar
from rltools.representation import TileCoding
from rltools.valuefn import LinearTD
from rltools.policy import Egreedy
from rltools.agent import TabularActionSarsa
from rltools.valuefn import LSTDlambda
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

domain = MountainCar(random_start= False, max_episode=1000)
proj = TileCoding(input_indicies = [[0,1]], 
                 ntiles = [10], 
                 ntilings=[10], 
                 state_range = domain.state_range, 
                 bias_term = True)
class PHI(object):
    def __init__(self):
        self.size = proj.size
    def __call__(self, s):
        return proj(s) if s != None else np.zeros(proj.size)
    
phi = PHI()

valuefn = LinearTD(len(domain.discrete_actions), 
                   phi,
                   alpha = 0.1,
                   lamb = 0.0,
                   gamma= 0.9)
policy = Egreedy(np.arange(len(domain.discrete_actions)), valuefn, epsilon = 0.1)
agent = TabularActionSarsa(domain.discrete_actions, policy, valuefn)


num_episodes = 100

def eval(valuefn):
    val = np.empty(100)
    for i,s in enumerate(product(np.linspace(domain.state_range[0][0], domain.state_range[1][0], 10),
                     np.linspace(domain.state_range[0][1], domain.state_range[1][1], 10))):
        s = np.array(s)
        val[i] = max([valuefn(s,0), valuefn(s,1), valuefn(s,2)])
    return val.reshape((10,10))

act = domain.discrete_actions
pump_pi= lambda s: act[0] if s[1]<0 else act[2]
val = LSTDlambda(pump_pi, domain, 0.9, None, phi, 30, 1000, 0.3)
# plt.ion()
# plt.imshow(eval(valuefn), interpolation= 'none')
# plt.pause(0.0001)
# 
# for i in xrange(num_episodes):
#     r_t, s_t = domain.reset()
#     agent.reset()
#     count = 0
#     while s_t != None:
#         r_t, s_t = domain.step(agent.step(r_t, s_t))
#         count += 1
#     agent.step(r_t, s_t)
#     plt.imshow(eval(valuefn), interpolation= 'none')
#     plt.pause(0.0001)
#     print count

