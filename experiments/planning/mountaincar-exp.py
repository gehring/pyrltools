from rltools.MountainCar import MountainCar
from rltools.npplanning import sample_gaussian
from rltools.planner import SingleEmbeddedAgent

import numpy as np


def fourier_features(X, w):
    if X.ndim == 1:
        X = X.reshape((1,-1))
    features = np.hstack((X, np.zeros((X.shape[0], 1)))).dot(w)        
    features = np.hstack((np.sin(features), np.cos(features)))
    return features.squeeze()/ np.sqrt(w.shape[1])

def run_episode(domain, agent):
	r_t, x_tp1 = domain.reset()
	x_t, a_t = None, None

	traj = []

    agent.reset()
    while x_t is not None:
        agent.step(x_t, a_t, r_t, x_tp1)
        a_t = agent.get_action()
        x_t = x_tp1
        r_t, x_tp1 = domain.step(a_t)
        traj.append((x_t, a_t, r_t, x_tp1))
    return traj


#################################################

# initialize the domain
domain = MountainCar(random_start = True,
					max_episode = np.Infinity)

state_range = domain.state_range
actions = domain.discrete_actions

# define kernel over actions
def single_action_kernel(i):
    #return lambda b: np.exp(-np.sum(((actions[i][None,:]-b)/6.0)**2, axis=1)/(0.5**2))
    return lambda b: (actions[i][None,:] == b).astype('float').squeeze()

action_kernels = [ single_action_kernel(i) for i in xrange(len(actions))]

# initialize the representation
num_gauss = 2000
w = sample_gaussian(state_range[0].shape[0], num_gauss, scale)   
phi = lambda X: fourier_features(X, w)

agent = SingleEmbeddedAgent(plan_horizon = 100,
                 dim = num_gauss*2, 
                 X_t = X_t,
                 A_t = A_t, 
                 X_tp1 = X_tp1, 
                 R_t = R_t,
                 X_term = X_term,
                 A_term = A_term,
                 R_term = R_term,
                 max_rank = 300,
                 blend_coeff = 0.0,
                 phi = phi,
                 action_kernels = action_kernels,
                 actions = actions,
                 learning_rate = 0.01,
                 discount = 0.99,
                 update_models = True,
                 use_valuefn = True)

##################################################