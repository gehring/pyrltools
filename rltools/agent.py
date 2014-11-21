import numpy as np
from policy import weighted_values
from rltools.policy import Egreedy
import matplotlib.pyplot as plt
from sklearn import preprocessing

class Agent(object):
    def __init__(self):
        pass

    def step(self, r, state):
        pass

    def reset(self):
        pass

    def proposeAction(self, state):
        pass


class POSarsa(Agent):
    def __init__(self, policy, valuefn, tracker, **argk):
        self.policy = policy
        self.valuefn = valuefn
        self.s_t = None
        self.a_t = None
        self.tracker = tracker

    def step(self, r, s_tp1):
        s_tp1 = self.tracker.update(s_tp1)
        if s_tp1 != None:
            a_tp1 = self.policy(s_tp1)
        else:
            a_tp1 = None
        self.valuefn.update(self.s_t, self.a_t, r, s_tp1, a_tp1)

        self.s_t = s_tp1
        self.a_t = a_tp1

        return a_tp1

    def reset(self):
        self.s_t = None
        self.a_t = None
        self.tracker.reset()

    def getActor(self):
        # actor generated will change its behaviour if the agent's value fn is
        # changed
        return POActor(self.tracker.copy(), self.policy)

class POActor(object):
    def __init__(self, tracker, policy):
        self.tracker = tracker
        self.policy = policy

    def reset(self):
        self.tracker.reset()

    def proposeAction(self, state):
        s = self.tracker.update(state)
        return self.policy(s)



class Sarsa(Agent):
    def __init__(self, policy, valuefn, **argk):
        self.policy = policy
        self.valuefn = valuefn
        self.s_t = None
        self.a_t = None

    def step(self, r, s_tp1):
        if s_tp1 != None:
            a_tp1 = self.policy(s_tp1)
        else:
            a_tp1 = None
        self.valuefn.update(self.s_t, self.a_t, r, s_tp1, a_tp1)

        self.s_t = s_tp1
        self.a_t = a_tp1

        return a_tp1

    def reset(self):
        self.s_t = None
        self.a_t = None

    def proposeAction(self, state):
        return self.policy(state)

class TabularActionSarsa(Agent):
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
    
class TabularPolicySarsa(Agent):
    def __init__(self, actions, mix_policy, policies, valuefn, **argk):
        self.policy = mix_policy
        self.policies = policies
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

        return self.actions[self.policies[a_tp1](s_tp1)] if a_tp1 != None else None

    def reset(self):
        self.s_t = None
        self.a_t = None

    def proposeAction(self, state):
        return self.actions[self.policies[self.policy(state)](state)]
    
class PolicySarsa(Agent):
    def __init__(self, mix_policy, policies, valuefn, **argk):
        self.policy = mix_policy
        self.policies = policies
        self.valuefn = valuefn
        self.s_t = None
        self.a_t = None

    def step(self, r, s_tp1):
        if s_tp1 != None:
            a_tp1 = self.policy(s_tp1)
        else:
            a_tp1 = None
        self.valuefn.update(self.s_t, self.a_t, r, s_tp1, a_tp1)

        self.s_t = s_tp1
        self.a_t = a_tp1

        return self.policies[a_tp1](s_tp1) if a_tp1 != None else None

    def reset(self):
        self.s_t = None
        self.a_t = None

    def proposeAction(self, state):
        return self.policies[self.policy(state)](state)

class LinearTabularPolicySarsa(Agent):
    def __init__(self, actions, mix_policy, policies, valuefn, **argk):
        self.policy = mix_policy
        self.policies = policies
        self.valuefn = valuefn
        self.s_t = None
        self.rho = None
        self.actions = actions

    def step(self, r, s_tp1):
        if s_tp1 != None:
            p_pi = self.policy.getprob(s_tp1)
            pi_tp1 = weighted_values(p_pi)
            p_a = np.vstack([p.getprob(s_tp1) for p in self.policies])
            a_tp1 = weighted_values(p_a[pi_tp1,:])[0]
            rho_tp1 = p_a[:,a_tp1]/ p_a[:,a_tp1].dot(p_pi)
        else:
            a_tp1 = None
            rho_tp1 = None
            a_tp1 = None
        self.valuefn.update(self.s_t, r, s_tp1, self.rho)

        self.s_t = s_tp1
        self.rho = rho_tp1

        return self.actions[a_tp1] if a_tp1 != None else None

    def reset(self):
        self.s_t = None

    def proposeAction(self, state):
        return self.actions[self.policies[self.policy(state)](state)]

class FittedQIteration(Agent):
    def __init__(self, actions,
                 policy, 
                 stateactions_projector, 
                 valuefn_regressor,
                 gamma,
                 num_iterations = None,
                 valuefn = None, 
                 samples = None,
                 batch_size = 2000,
                 max_samples = 10000,
                 dtype = np.float,
                 improve_behaviour = True,
                 **argk):
        self.actions = actions
        
        # policy is expected to return a continuous action
        self.policy = policy
        
        self.phi = stateactions_projector
        self.regressor = valuefn_regressor
        self.gamma = gamma
        
        self.max_samples = max_samples
        
        self.samples = samples
        if self.samples == None:
            self.samples = (np.empty((0,0), dtype=dtype), 
                            np.empty(0,dtype=dtype), 
                            np.array(0, dtype='O') )
        self.sample_i = self.samples[0].shape[0]
        
        # this flag forces the oldest samples to be forgotten
        self.full = self.sample_i >= self.max_samples
        
        self.batch_size = batch_size
        
        self.valuefn = valuefn
        self.dtype = dtype
        self.maxval = np.vectorize(maxValue, 
                                           otypes = [self.dtype],
                                           excluded = 'valuefn')
        # state tracking
        self.s_t = None
        self.a_t = None
        self.count = 0
        self.dtype = dtype
        
        # update the behviour's valuefn
        self.improve_behaviour = improve_behaviour
        
        self.num_iter = num_iterations
        if self.num_iter is None:
            self.num_iter = int(1/(1-gamma))
        
    def step(self, r, s_tp1):
        if s_tp1 != None:
            a_tp1 = self.policy(s_tp1)
        else:
            a_tp1 = None
            
        if self.s_t is not None:
            self.update_samples(self.s_t, self.a_t, r, s_tp1)

        self.s_t = s_tp1
        self.a_t = a_tp1
        
        self.count += 1
        
        # if enough time has elapsed, recompute the max Q-value function
        if self.count>= self.batch_size:
            self.valuefn = None
            print 'generating new value fn'
            for i in xrange(self.num_iter):
                print 'iteration  #'+str(i)
                self.valuefn = self.regressor(*self.generateRegressionData())
            
            if self.improve_behaviour:
                self.policy.valuefn = self.valuefn
            self.count = 0
        

        return a_tp1

    def reset(self):
        self.s_t = None
        self.a_t = None

    def proposeAction(self, state):
        return self.policy(state)
    
    def getGreedyPolicy(self):
        return Egreedy(self.actions, self.valuefn, epsilon = 0.0)
        
    def generateRegressionData(self):
        sa_t, r_t, s_tp1 = self.samples
        if self.full:
            if self.valuefn == None:
                y = r_t
            else:
                y = r_t + self.maxval(s_tp1, valuefn = self.valuefn)*self.gamma
            x = sa_t
        else:
            if self.valuefn == None:
                y = r_t[:self.sample_i]
            else:
                y = r_t[:self.sample_i] + self.maxval(s_tp1[:self.sample_i],
                                                       valuefn = self.valuefn)*self.gamma
            x = sa_t[:self.sample_i]
        i = np.random.permutation(x.shape[0])
        return x[i,:], y[i]
    
    def update_samples(self, new_s_t, new_a_t, new_r_t, new_s_tp1):
        sa_t, r_t, s_tp1 = self.samples
        
        # check if current matrices can contain sample
        if not self.full and sa_t.shape[0] <= self.sample_i:
            # resize matrices
            sa_t = np.resize(sa_t, (min(self.max_samples, max(100, sa_t.shape[0]*2)), self.phi.size))
            r_t = np.resize(r_t, min(self.max_samples,max(100, sa_t.shape[0]*2)))
            s_tp1 = np.resize(s_tp1,min(self.max_samples,max(100, sa_t.shape[0]*2)))
            self.samples = (sa_t, r_t, s_tp1)
        
        # add sample
        sa_t[self.sample_i, : ] = self.phi(new_s_t, new_a_t)
        r_t[self.sample_i] = new_r_t
        s_tp1[self.sample_i] = new_s_tp1
        
        # if the max number of samples is reached, trigger the flag
        if self.sample_i+1 >= self.max_samples:
            self.full = True
        
        # wrap around index if the max samples is reached
        self.sample_i = (self.sample_i+1) % self.max_samples
        
def maxValue(s, valuefn):
    if s is not None:
        vals = valuefn(s)
        return np.max(vals)
    else:
        return 0

class Sarsa_Factory(object):
    def __init__(self, **argk):
        self.params = argk

    def __call__(self, **argk):
        params = dict(self.params)
        params.update([x for x in argk.items()])
        valuefn = params.get('valuefn')
        policy = params.get('policy')
        return Sarsa(**params)