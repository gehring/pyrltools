from rltools.agent import FittedQIteration
from rltools.policy import Egreedy
from rltools.representation import TileCodingDense
from rltools.theanotools import MLPfit
from rltools.valuefn import MLPValueFn, SklearnValueFn
from rltools.MountainCar import MountainCar

from sklearn.svm import SVR
from sklearn import preprocessing

import numpy as np
import theano

import matplotlib.pyplot as plt

rng = np.random.RandomState(312)

gamma = 0.95

domain = MountainCar(True, 1000)
a_range = domain.action_range
s_range = domain.state_range
sa_range = [np.hstack((s_range[0], a_range[0])),
            np.hstack((s_range[1], a_range[1]))]
actions = [ np.array([a]) for a in np.arange(a_range[0][0], 
                                             a_range[1][0], 
                                             0.1*(a_range[1][0]-a_range[0][0]))]



def blank_valuefn(s, a= None):
    if a is None:
        return np.zeros(len(actions))
    else:
        return 0
    
class IdenStateAction(object):
    def __init__(self, proj):
        self.proj = proj
        self.size = proj.size
        
    def __call__(self, s, a):
        if s is None:
            return None
        return self.proj(np.hstack((s,a))).astype(theano.config.floatX)
    
class IdenStateActionNoProj(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, s, a):
        return np.hstack((s,a)).astype(theano.config.floatX)

proj = TileCodingDense(input_indicies = [np.arange(3)],
                 ntiles=[4,4, 3],
                 ntilings=[30],
                 hashing=None,
                 state_range= sa_range,
                 bias_term = True)
phi = IdenStateAction(proj)
phi = IdenStateActionNoProj(3)


def getMLPValueFn(x, y):
    mlp = MLPfit(0.01, 
                 [100,10], 
                 rng, 
                 0.0001, 
                 0.0, 
                 x, 
                 y, 
                 1000, 
                 0.2, 
                 10000,
                 1,
                 patience = 100,
                 patience_increase= 2)
    return MLPValueFn(mlp, actions, phi)

def getSVRValueFn(x,y):
    scaler = preprocessing.StandardScaler().fit(x)
    svr = SVR(kernel = 'rbf', C=1e3, gamma=1)
    clf = svr.fit(scaler.transform(x), y)
    sclf = lambda s: clf.predict(scaler.transform(s))
    return MLPValueFn(sclf, actions, phi)

policy = Egreedy(actions, blank_valuefn, epsilon= 0.05)
agent = FittedQIteration(actions, 
                         policy, 
                         phi, 
                         getSVRValueFn, 
                         gamma, 
                         num_iterations=20,
                         valuefn= None, 
                         samples= None, 
                         batch_size= 5000, 
                         max_samples= 10000, 
                         dtype= theano.config.floatX, 
                         improve_behaviour= True)

num_episodes = 500
k = 1
valuefn = agent.valuefn
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
            samples = np.hstack((xx.reshape((-1, 1)), yy.reshape((-1,1)))).astype(agent.dtype)
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

    