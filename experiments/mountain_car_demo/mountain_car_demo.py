from rltools.MountainCar import MountainCar
from rltools.representation import TileCoding
from rltools.valuefn import LinearTD
from rltools.policy import Egreedy
from rltools.agent import TabularActionSarsa
import numpy as np

domain = MountainCar(True, 1000)
phi = TileCoding(input_indicies = [[0,1]], 
                 ntiles = [10], 
                 ntilings=[10], 
                 state_range = domain.state_range, 
                 bias_term = True)

valuefn = LinearTD(len(domain.discrete_actions), 
                   phi,
                   alpha = 0.01,
                   lamb = 0.6,
                   gamma= 0.9)
policy = Egreedy(np.arange(len(domain.discrete_actions)), valuefn)
agent = TabularActionSarsa(domain.discrete_actions, policy, valuefn)


num_episodes = 100

for i in xrange(num_episodes):
    r_t, s_t = domain.reset()
    agent.reset()
    count = 0
    while s_t != None:
        r_t, s_t = domain.step(agent.step(r_t, s_t))
        count += 1
    agent.step(r_t, s_t)
    print count
        