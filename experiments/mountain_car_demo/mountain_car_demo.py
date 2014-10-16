from rltools.MountainCar import MountainCar
from rltools.representation import TileCoding
from rltools.valuefn import LinearTD
from rltools.policy import Egreedy
from rltools.agent import TabularActionSarsa
from rltools.valuefn import LSTDlambda
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

domain = MountainCar(random_start= True, max_episode=1000)
proj = TileCoding(input_indicies = [[0,1]],
                 ntiles = [10],
                 ntilings=[8],
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
                   alpha = 0.01,
                   lamb = 0.0,
                   gamma= 0.9)
policy = Egreedy(np.arange(len(domain.discrete_actions)), valuefn, epsilon = 0.1)
agent = TabularActionSarsa(domain.discrete_actions, policy, valuefn)


num_episodes = 1000
def eval(valuefn):
    val = np.empty(10000)
    for i,s in enumerate(product(np.linspace(domain.state_range[0][0], domain.state_range[1][0], 100),
                     np.linspace(domain.state_range[0][1], domain.state_range[1][1], 100))):
        s = np.array(s)
        val[i] =valuefn(s)#max([valuefn(s,0), valuefn(s,1), valuefn(s,2)])
    return val.reshape((100,100))

act = domain.discrete_actions
pump_pi= lambda s: act[0] if s[1]<0 else act[2]
val = LSTDlambda(pump_pi, domain, 0.9, None, phi, 100, 1000, 0.6)
plt.imshow(eval(val), interpolation= 'none')
plt.show()


# plt.ion()
# plt.imshow(eval(valuefn), interpolation= 'none')
# plt.pause(0.0001)

# for i in xrange(num_episodes):
#     r_t, s_t = domain.reset()
#     agent.reset()
#     count = 0
#     while s_t != None:
#         r_t, s_t = domain.step(agent.step(r_t, s_t))
#         count += 1
#     agent.step(r_t, s_t)
# #
#     if i % 100 == 0:
#         plt.gca().clear()
#         plt.imshow(eval(valuefn), interpolation= 'none')
#         plt.pause(0.0001)
#     print count

