import numpy as np
from rltools.math import discrete_sample

class HMM(object):
    def __init__(self):
        pass
    def step(self):
        pass
    def reset(self):
        pass
    
class DiscreteHMM(HMM):
    def __init__(self, p0, T, O):
        super(DiscreteHMM, self).__init__()
        self.T = T
        self.O = O
        self.p0 = p0
        self.reset()
        
        
    def filter(self, obs_seq, b0):
        if obs_seq.size == 0:
            return b0
        
        o_t = obs_seq[-1]
        b_t = self.O[:,o_t] * self.T.T.dot(self.filter(obs_seq[:-1], b0))
        return b_t/np.sum(b_t)
        
    def filter_all(self, obs_seq, b0):
        bs = np.empty((self.T.shape[0], len(obs_seq)+1))
        bs[:,0] = b0
        for i in xrange(0, len(obs_seq)):
            o_t = obs_seq[i]
            bs[:,i+1] = self.O[:,o_t] * self.T.T.dot(bs[:,i])
            bs[:,i+1] /= np.sum(bs[:,i+1])
        return bs
    
    def step(self):
        self.s = discrete_sample(self.T[self.s,:])
        return discrete_sample(self.O[self.s,:])
        
    def reset(self):
        self.s = discrete_sample(self.p0)
        
    def get_obs_prob(self, s):
        return self.O[s,:]
    
    def get_transition_prob(self, s):
        return self.T[s,:]
        
        
def generate_sequence(hmm, size):
    hmm.reset()
    obs_s = [(hmm.s, hmm.step()) for i in xrange(size)]
    states, obs = zip(*obs_s)
    states = states + (hmm.s,) 
    return np.array(obs), np.array(states)
    
        
    