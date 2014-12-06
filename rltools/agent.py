import numpy as np
from policy import weighted_values
from rltools.policy import Egreedy
from itertools import izip, chain
from rltools.valuefn import LSQ, SFLSQ, QAKLSQ
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time

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
        
        self.samples = TransitionData(samples, 
                                      stateactions_projector, 
                                      max_samples, 
                                      dtype)
        
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
            self.samples.update_samples(self.s_t, self.a_t, r, s_tp1)

        self.s_t = s_tp1
        self.a_t = a_tp1
        
        self.count += 1
        
        # if enough time has elapsed, recompute the max Q-value function
        if self.count>= self.batch_size:
            self.valuefn = None
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
        sa_t, r_t, s_tp1 = self.samples.getSamples()
        if self.valuefn == None:
            y = r_t
        else:
            y = r_t + self.maxval(s_tp1, valuefn = self.valuefn)*self.gamma
        x = sa_t
        i = np.random.permutation(x.shape[0])
        return x[i,:], y[i]

class TransitionData(object):
    def __init__(self,
                 samples,
                 stateaction_projector,
                 max_samples,
                 replace_data = True,
                 dtype = np.float):
        self.max_samples = max_samples
        self.phi = stateaction_projector
        
        self.samples = samples
        if self.samples == None:
            self.samples = (np.empty((0,0), dtype=dtype), 
                            np.empty(0,dtype=dtype), 
                            np.empty(0, dtype='O') )
        self.sample_i = self.samples[0].shape[0]
        
        # this flag forces the oldest samples to be forgotten
        self.full = self.sample_i >= self.max_samples
        self.replace_data = replace_data
        
    def update_samples(self, new_s_t, new_a_t, new_r_t, new_s_tp1):
        if not self.full or self.replace_data:
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
        
    def getSamples(self):
        sa_t, r_t, s_tp1 = self.samples
        if not self.full:
            r_t = r_t[:self.sample_i]
            s_tp1 = s_tp1[:self.sample_i]
            sa_t = sa_t[:self.sample_i]
        return sa_t, r_t, s_tp1
    
    def constructMatrices(self, a_tp1, ind = None):
        sa_t, r_t, s_tp1 = self.getSamples()
        if ind is not None:
            sa_t  = sa_t[ind, :]
            r_t = r_t[ind]
            s_tp1 = s_tp1[ind]
        
        X_t = sa_t
        
        # build X_tp1 sparse matrix
        X_tp1 = np.vstack([ self.phi(s,a) if s is not None else np.zeros(self.phi.size)
                             for s,a in izip(s_tp1, a_tp1)])
        
        return X_t, r_t, X_tp1
    
    def constructMatricesfromPolicy(self, policy):
        sa_t, r_t, s_tp1 = self.getSamples()
        X_t = sa_t
        
        # build X_tp1 sparse matrix
        X_tp1 = np.vstack([ self.phi(s).dot(policy.getprob(s)) if s is not None else np.zeros(self.phi.size)
                             for s in s_tp1])
        
        return X_t, r_t, X_tp1
    
    def constructRegressionMatrices(self, gamma, getmaxvalue, ind = None):
        sa_t, r_t, s_tp1 = self.getSamples()
        if ind is not None:
            sa_t  = sa_t[ind, :]
            r_t = r_t[ind]
            s_tp1 = s_tp1[ind]
        
        X = sa_t
        y = r_t + gamma*getmaxvalue(s_tp1)
        
        return X, y
            
class BinarySparseTransitionData(object):
    def __init__(self,
                 samples,
                 stateaction_projector,
                 max_samples,
                 replace_data = True,
                 dtype = np.float):
        self.max_samples = max_samples
        self.phi = stateaction_projector
        
        self.samples = samples
        if self.samples == None:
            self.samples = (np.empty(0, dtype='O'), 
                            np.empty(0,dtype=dtype), 
                            np.empty(0, dtype='O') )
        self.sample_i = self.samples[0].shape[0]
        
        # this flag forces the oldest samples to be forgotten
        self.full = self.sample_i >= self.max_samples
        if self.full:
            self.sample_i = 0
        self.replace_data = replace_data
        
    def update_samples(self, new_s_t, new_a_t, new_r_t, new_s_tp1):
        if not self.full or self.replace_data:
            sa_t, r_t, s_tp1 = self.samples
            
            # check if current matrices can contain sample
            if not self.full and sa_t.shape[0] <= self.sample_i:
                # resize matrices
                sa_t = np.resize(sa_t, min(self.max_samples, max(100, sa_t.shape[0]*2)))
                r_t = np.resize(r_t, min(self.max_samples,max(100, sa_t.shape[0]*2)))
                s_tp1 = np.resize(s_tp1,min(self.max_samples,max(100, sa_t.shape[0]*2)))
                self.samples = (sa_t, r_t, s_tp1)
            
            # add sample
            sa_t[self.sample_i] = self.phi(new_s_t, new_a_t)
            r_t[self.sample_i] = new_r_t
            s_tp1[self.sample_i] = new_s_tp1
            
            # if the max number of samples is reached, trigger the flag
            if self.sample_i+1 >= self.max_samples:
                self.full = True
            
            # wrap around index if the max samples is reached
            self.sample_i = (self.sample_i+1) % self.max_samples
        
    def getSamples(self):
        sa_t, r_t, s_tp1 = self.samples
        if not self.full:
            r_t = r_t[:self.sample_i]
            s_tp1 = s_tp1[:self.sample_i]
            sa_t = sa_t[:self.sample_i]
        return sa_t, r_t, s_tp1
    
    def constructMatrices(self, a_tp1, ind = None):
        sa_t, r_t, s_tp1 = self.getSamples()
        if ind is not None:
            sa_t = sa_t[ind]
            r_t = r_t[ind]
            s_tp1 = s_tp1[ind]
        shape = (sa_t.shape[0], self.phi.size)
        
        # build X_t sparse matrix
        indices = [ np.vstack((np.ones(sa.size, dtype= 'uint')*i, sa))  
                                for i,sa in enumerate(sa_t) ]
        indices = np.hstack(indices)
        X_t = sp.csc_matrix((np.ones(indices.shape[1]), indices),
                            shape = shape,
                            dtype= 'int')
        
        # build X_tp1 sparse matrix
        sa_tp1 = [ self.phi(s,a) if s is not None else np.empty(0)
                             for s,a in izip(s_tp1, a_tp1)]
        
        indices = [ np.vstack((np.ones(sa.size, dtype= 'uint')*i, sa) ) 
                                for i,sa in enumerate(sa_tp1) ]
        indices = np.hstack(indices)
        X_tp1 = sp.csc_matrix((np.ones(indices.shape[1]), indices), 
                              shape = shape, 
                              dtype= 'int')
        
        return X_t, r_t, X_tp1
    
    def constructMatricesfromPolicy(self, policy):
        sa_t, r_t, s_tp1 = self.getSamples()
        shape = (sa_t.shape[0], self.phi.size)
        
        # build X_t sparse matrix
        indices = [ np.vstack((np.ones(sa.size, dtype= 'uint')*i, sa))  
                                for i,sa in enumerate(sa_t) ]
        indices = np.hstack(indices)
        X_t = sp.csc_matrix((np.ones(indices.shape[1]), indices),
                            shape = shape,
                            dtype= 'int')
        
        # build X_tp1 sparse matrix
        sa_tp1 = [ self.phi(s).T.dot(policy.getprob(s)).T if s is not None else np.empty(0)
                             for s in s_tp1]
        
        indices = [ np.vstack((np.ones(sa.size, dtype= 'uint')*i, sa) ) 
                                for i,sa in enumerate(sa_tp1) ]
        indices = np.hstack(indices)
        X_tp1 = sp.csr_matrix((np.ones(indices.shape[1]), indices), 
                              shape = shape, 
                              dtype= 'int')
        
        return X_t, r_t, X_tp1
    
    def constructRegressionMatrices(self, gamma, getmaxvalue, ind = None):
        sa_t, r_t, s_tp1 = self.getSamples()
        if ind is not None:
            sa_t  = sa_t[ind, :]
            r_t = r_t[ind]
            s_tp1 = s_tp1[ind]
        shape = (sa_t.shape[0], self.phi.size)
            
        # build X_t sparse matrix
        indices = [ np.vstack((np.ones(sa.size, dtype= 'uint')*i, sa))  
                                for i,sa in enumerate(sa_t) ]
        indices = np.hstack(indices)
        X = sp.csc_matrix((np.ones(indices.shape[1]), indices),
                            shape = shape,
                            dtype= 'int')
        y = r_t + gamma*getmaxvalue(s_tp1)
        
        return X, y
            
        
        
def maxValue(s, valuefn):
    if s is not None:
        vals = valuefn(s)
        return np.max(vals)
    else:
        return 0
    
def argmaxValue(s, valuefn):
    if s is not None:
        values = valuefn(s)
        m = np.random.choice(np.argwhere(values == np.amax(values)).flatten(),1)[0]
        return m
    else:
        return 0
    
# class LSPI(FittedQIteration):
#     def __init__(self,
#                  actions,
#                  policy,
#                  gamma,
#                  stateactions_projector,
#                  valuefn = None,
#                  samples = None,
#                  batch_size = 2000,
#                  max_samples = 10000,
#                  improve_behaviour = True):
#         
#         self.samples = samples
#         if samples is None or isinstance(samples, tuple):
#             self.samples = TransitionData(samples, 
#                                           stateactions_projector, 
#                                           max_samples)
#         self.batch_size = batch_size
#         self.phi = stateactions_projector
#         self.gamma = gamma
#         self.policy = policy
#         self.actions = np.array(actions)
#         
#         self.argmaxval = np.vectorize(argmaxValue,
#                                    otypes =[np.int],
#                                    excluded = 'valuefn')
#         
#         self.valuefn = valuefn
#         
#         # state tracking
#         self.s_t = None
#         self.a_t = None
#         self.count = 0
#         
#         self.improve_behaviour = improve_behaviour
#         
#     def step(self, r, s_tp1):
#         if self.s_t is not None:
#             self.samples.update_samples(self.s_t, self.a_t, r, s_tp1)
#             self.count += 1
# 
#         
#         
#         
#         # if enough time has elapsed, recompute the max Q-value function
#         if self.count>= self.batch_size:
#             print 'Generating new policy'
#             start_time = time.clock()
#             _, _, sample_tp1 = self.samples.getSamples()
#             # get max actions for each sampled s_tp1
#             a_tp1 = self.actions[self.argmaxval(sample_tp1, self.valuefn)]
#             X_t, r_t, X_tp1 = self.samples.constructMatrices(a_tp1)
#             print 'time taken to process '+str(X_t.shape[0]) +' samples: '\
#                          + str(time.clock() - start_time) + ' seconds'
#             self.valuefn = LSQ(X_t, r_t, X_tp1, self.gamma, self.phi)
#             
#             if self.improve_behaviour:
#                 self.policy.valuefn = self.valuefn
#             self.count = 0
#             print 'total time taken for improvement step: ' + str(time.clock() - start_time) + ' seconds\n'
#         
#         
#         if s_tp1 != None:
#             a_tp1 = self.policy(s_tp1)
#         else:
#             a_tp1 = None
# 
#         self.s_t = s_tp1
#         self.a_t = a_tp1
#         
# 
#         return a_tp1
# 
#     def reset(self):
#         self.s_t = None
#         self.a_t = None
# 
#     def proposeAction(self, state):
#         return self.policy(state)
#     
#     def getGreedyPolicy(self):
#         return Egreedy(self.actions, self.valuefn, epsilon = 0.0)
    
class LSPI(FittedQIteration):
    def __init__(self,
                 actions,
                 policy,
                 gamma,
                 stateactions_projector,
                 valuefn = None,
                 samples = None,
                 batch_size = 2000,
                 iteration_per_batch = 1,
                 max_samples = 10000,
                 improve_behaviour = True,
                 method = None,
                 **args):
        
        self.samples = samples
        if samples is None or isinstance(samples, tuple):
            self.samples = TransitionData(samples, 
                                          stateactions_projector, 
                                          max_samples)
        self.batch_size = batch_size
        self.iteration_per_batch =iteration_per_batch
        self.phi = stateactions_projector
        self.gamma = gamma
        self.policy = policy
        self.actions = np.array(actions)
        
        self.regressor = method if method is not None else LSQ
        self.args = args
        
        self.argmaxval = np.vectorize(argmaxValue,
                                   otypes =[np.int],
                                   excluded = 'valuefn')
        
        self.valuefn = valuefn
        
        # state tracking
        self.s_t = None
        self.a_t = None
        self.count = 0
        
        self.improve_behaviour = improve_behaviour
        
    def step(self, r, s_tp1):
        if self.s_t is not None:
            self.samples.update_samples(self.s_t, self.a_t, r, s_tp1)
            self.count += 1

        
        
        
        # if enough time has elapsed, recompute the max Q-value function
        if self.count>= self.batch_size:
            self.improve_policy(iter= self.iteration_per_batch)
        
        
        if s_tp1 != None:
            a_tp1 = self.policy(s_tp1)
        else:
            a_tp1 = None

        self.s_t = s_tp1
        self.a_t = a_tp1
        

        return a_tp1

    def reset(self):
        self.s_t = None
        self.a_t = None

    def proposeAction(self, state):
        return self.policy(state)
    
    def getGreedyPolicy(self):
        return Egreedy(self.actions, self.valuefn, epsilon = 0.0)
    
    def improve_policy(self, iter= 1):
        print 'Generating new policy'
        start_time = time.clock()
        for i in xrange(iter):
            print 'running iteration #' + str(i+1)
            _, _, sample_tp1 = self.samples.getSamples()
            # get max actions for each sampled s_tp1
            a_tp1 = self.actions[self.argmaxval(sample_tp1, self.valuefn)]
#                 print 'time taken to find max actions : '\
#                             + str(time.clock() - start_time) + ' seconds'
            X_t, r_t, X_tp1 = self.samples.constructMatrices(a_tp1)
#                 print 'time taken to process '+str(X_t.shape[0]) +' samples: '\
#                              + str(time.clock() - start_time) + ' seconds'
            self.valuefn = self.regressor(X_t, r_t, X_tp1, self.gamma, self.phi, **self.args)
            
        if self.improve_behaviour:
            self.policy.valuefn = self.valuefn
        self.count = 0
        print 'total time taken for improvement step: ' + str(time.clock() - start_time) + ' seconds\n'
        
        
def get_max_action(state, valuefn):
    return valuefn.getmaxaction(state)        
        
        
class QuadKernelLSPI(LSPI):
    def __init__(self,
                 actions,
                 policy,
                 gamma,
                 stateactions_projector,
                 valuefn = None,
                 samples = None,
                 batch_size = 2000,
                 iteration_per_batch = 1,
                 max_samples = 10000,
                 improve_behaviour = True,
                 action_range = None,
                 max_number_of_samples = None,
                 **args):
        
        super(QuadKernelLSPI, self).__init__(
                 actions = actions,
                 policy = policy,
                 gamma = gamma,
                 stateactions_projector = stateactions_projector,
                 valuefn = valuefn,
                 samples = samples,
                 batch_size = batch_size,
                 iteration_per_batch = iteration_per_batch,
                 max_samples = max_samples,
                 improve_behaviour = improve_behaviour,
                 method = QAKLSQ,
                 **args)
        
        self.action_range = action_range
        self.maxactions = np.vectorize(get_max_action,
                                   otypes =[np.float],
                                   excluded = 'valuefn')
        self.max_number_of_samples = max_number_of_samples
        
    def improve_policy(self, iter= 1):
        print 'Generating new policy'
        start_time = time.clock()
        for i in xrange(iter):
            print 'running iteration #' + str(i+1)
            _, _, sample_tp1 = self.samples.getSamples()
            if (self.max_number_of_samples is not None 
                    and sample_tp1.size > self.max_number_of_samples):
                ind = np.random.choice(sample_tp1.size, self.max_number_of_samples, replace=False).astype('int')
                # get max actions for each sampled s_tp1
                a_tp1 = self.maxactions(sample_tp1[ind], valuefn = self.valuefn)
                if self.action_range is not None:
                    a_tp1 = np.clip(a_tp1, *self.action_range)
                
                X_t, r_t, X_tp1 = self.samples.constructMatrices(a_tp1, ind)
            else:  
                # get max actions for each sampled s_tp1
                a_tp1 = self.maxactions(sample_tp1, valuefn = self.valuefn)
                if self.action_range is not None:
                    a_tp1 = np.clip(a_tp1, *self.action_range)
                
                X_t, r_t, X_tp1 = self.samples.constructMatrices(a_tp1)
            self.valuefn = self.regressor(X_t, r_t, X_tp1, self.gamma, self.phi, **self.args)
            
        if self.improve_behaviour:
            self.policy.valuefn = self.valuefn
        self.count = 0
        print 'total time taken for improvement step: ' + str(time.clock() - start_time) + ' seconds\n'
        
class Sarsa_Factory(object):
    def __init__(self, **argk):
        self.params = argk

    def __call__(self, **argk):
        params = dict(self.params)
        params.update([x for x in argk.items()])
        valuefn = params.get('valuefn')
        policy = params.get('policy')
        return Sarsa(**params)