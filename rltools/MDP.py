import numpy as np

class MDP(object):
    def __init__(self, Ps, R, terminate=None, p0=None, **argk):
        self.Ps = Ps
        R = R
        self.p0 = p0
        self.size = Ps[0].shape[1]
        if self.p0 == None:
            self.p0 = np.ones(self.size)/self.size

        self.terminate = terminate
        if terminate == None:
            self.terminate = lambda s: False

        # this is very wasteful for large state spaces (memory)
        self.indices = np.arange(self.size)


    def start(self):
        self.s = np.random.choice(self.indices, p=self.p0)
        return self.s

    def step(self, a):
        s_t = self.s
        p = self.Ps[a][:, self.s]
        self.s = np.random.choice(self.indices, p=p)

        if self.terminate(self.s):
            self.s = None

        return self.s, self.R(s_t, a, self.s)


class MDPRunner(object):
    def __init__(self, MDP, policy, **argk):
        self.MDP = MDP
        self.policy = policy

    def start(self):
        self.s = MDP.start(self)
        return self.s

    def step(self):
        a_t = self.policy(self.s)
        s_t = self.s
        self.s, r = self.MDP.step(a_t)
        return (s_t, a_t, r, self.s)

class MDPSequenceGenerator(object):
    def __init__(self, MDP, policy, maxlength = None, **argk):
        self.MDPr = MDPRunner(MDP, policy)
        self.maxlength = maxlength

    def getsequence(self):
        sequence = []
        s_t = self.MDPr.start()
        count = 0
        while s_t != None and (self.maxlength == None or self.maxlength> count):
            (s_t, a_t, r_t, s_tp1) = self.MDPr.step()
            sequence.add((s_t, a_t, r_t, s_tp1))
            count += 1

        return sequence

class TransitionMapMDP(MDP):
    def __init__(self,
                 num_s,
                 num_a,
                 T,
                 R,
                 terminate=None,
                 p0=None,
                 **argk):
        Ps = []
        for a in xrange(num_a):
            P = np.array([T(s, a) for s in xrange(num_s)]).T
            Ps.add(P)
        super(TransitionMapMDP, self).__init__(Ps, R, terminate, p0)




