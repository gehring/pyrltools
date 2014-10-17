from rltools.MountainCar import MountainCar
from rltools.representation import TileCoding
from rltools.valuefn import LinearTD
from rltools.policy import Egreedy
from rltools.agent import TabularActionSarsa
from rltools.valuefn import LSTDlambda
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from itertools import product

domain = MountainCar(random_start= True, max_episode=10000)

act = domain.discrete_actions
pump_pi= lambda s: act[0] if s[1]<0 else act[2]
pump_pi_index= lambda s: 0 if s[1]<0 else 2


proj = TileCoding(input_indicies = [np.array([0,1])],
                 ntiles = [12],
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
                   alpha = 0.01,
                   lamb = 0.5,
                   gamma= 0.99)
policy = Egreedy(np.arange(len(domain.discrete_actions)), valuefn, epsilon = 0.05)
agent = TabularActionSarsa(domain.discrete_actions, policy, valuefn)


num_episodes = 5000
def eval(valuefn):
    val = np.empty(10000)
    x = np.linspace(domain.state_range[0][0], domain.state_range[1][0], 100)
    y = np.linspace(domain.state_range[0][1], domain.state_range[1][1], 100)
    for i,s in enumerate(product(x, y)):
        s = np.array(s)
#         val[i] = valuefn(s, agent.policy(s))
        val[i] = max([valuefn(s,0), valuefn(s,1), valuefn(s,2)])
    X, Y = np.meshgrid(x, y)
    return X, Y, val.reshape((100,100)).T


def getVectorField(valuefn):
    res=20
    U = np.empty(res**2)
    V = np.empty(res**2)
    C = np.empty(res**2)
    x = np.linspace(domain.state_range[0][0], domain.state_range[1][0], res)
    y = np.linspace(domain.state_range[0][1], domain.state_range[1][1], res)
    for i,s in enumerate(product(x, y)):
        s = np.array(s)
        s_next = s.copy()
        a = np.argmax([valuefn(s,0), valuefn(s,1), valuefn(s,2)])
        s_next[1] += (act[a][0]*10 + np.cos(3*s[0])*-0.0025)
        s_next[:] = np.clip(s_next, *domain.state_range)
        s_next[0] += s_next[1]

        s_next[:] = np.clip(s_next, *domain.state_range)
        if s_next[0] <= domain.state_range[0][1] and s_next[1] < 0:
            s_next[1] = 0
        s_next -= s
        U[i] = s_next[0]
        V[i] = s_next[1]
        C[i] = a/2.0
    X, Y = np.meshgrid(x, y)
    return X, Y, U.reshape((res,res)).T, V.reshape((res,res)).T, C.reshape((res,res)).T


# val = LSTDlambda(pump_pi, domain, 0.9, None, phi, 1000, 1000, 0.6)
# plt.imshow(eval(val), interpolation= 'none')
# plt.show()


# plt.ion()
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.view_init(55,55)
# ax.plot_surface(*eval(valuefn), cmap = cm.coolwarm)
plt.contourf(*eval(valuefn))
plt.quiver(*getVectorField(valuefn))
plt.title('initial')
plt.savefig('D:\\mountain_car_demo\\policy\\0.png')
# plt.pause(0.0005)

k = 1
for i in xrange(num_episodes):
    r_t, s_t = domain.reset()
    agent.reset()
    count = 0
    while s_t != None:
        r_t, s_t = domain.step(agent.step(r_t, s_t))
        count += 1
    agent.step(r_t, s_t)
#
    if i % 25 == 0:
        plt.gca().clear()
#         plt.gcf().clear()
#         plt.
        plt.contourf(*eval(valuefn))
        plt.quiver(*getVectorField(valuefn))
#         plt.colorbar()
#         fig.clear()
#         ax = fig.gca(projection='3d')
#         ax.view_init(55,55)
#         ax.plot_surface(*eval(valuefn), cmap = cm.coolwarm)
        plt.title('episode ' + str(i))
        plt.savefig('D:\\mountain_car_demo\\policy\\' + str(k) + '.png')
        k +=1
#         plt.pause(0.0005)
#     print count

