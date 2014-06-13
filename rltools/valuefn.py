from rltools.representation import IdentityProj
import numpy as np
from rltools.pyneuralnet import NeuralNet

class ValueFn(object):
    def __init__(self):
        pass

    def __call__(self, state, actions):
        pass

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        pass

class LinearTD0(ValueFn):
    def __init__(self, projector, **argk):
        super(LinearTD0, self).__init__()
        self.projector = projector
        self.gamma = argk.get('gamma', 0.9)
        self.alpha = argk.get('alpha', 0.1)

    def __call__(self, state, actions):
        projected = np.array(self.projector(state, actions))
        return projected.dot(self.w)

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if s_t == None:
            return

        phi_t = self.projector(s_t, a_t)
        v_t = phi_t.dot(self.w)
        if s_tp1 == None:
            v_tp1 = 0
        else:
            v_tp1 = self(s_tp1, a_tp1)

        td = r + self.gamma * v_tp1 - v_t

        self.w += self.alpha * td * phi_t


class RBFValueFn(ValueFn):
    def __init__(self,
                 alpha,
                 c,
                 w,
                 projector,
                 gamma,
                 sigma,
                 eta,
                 **kargs):
        super(RBFValueFn, self).__init__()
        self.alpha = alpha
        self.projector = projector
        self.gamma = gamma
        self.width = sigma **-2
        self.eta =  eta
        self.c = np.array(c)
        self.w = np.array(w)


    def __call__(self, state, action):
        projected = self.projector(state, action)
        return self.w.dot(self.computeRBFs(projected, self.c))

    def computeRBFs(self, x, c):
        s =  np.sum( (x-c)**2, axis=1)
        s *= (-self.width)
        rbfs = np.exp(s)
        rbfs /= np.sum(rbfs)
        return rbfs

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if s_t == None:
            return

        phi_t = self.projector(s_t)
        rbfs = self.computeRBFs(phi_t, self.c)
        if s_tp1 == None:
            v_tp1 = 0
            drbfs = np.zeros_like(rbfs)
        else:
            phi_tp1 = self.projector(s_tp1, a_tp1)
            rbfs_tp1 = self.computeRBFs(phi_tp1, self.c)
            v_tp1 = self.w.dot(rbfs_tp1)
            drbfs = rbfs_tp1 - rbfs

        v_t =  self.w.dot(rbfs)


        delta = r + self.gamma * v_tp1 - v_t
        self.w -= (self.eta * self.alpha * delta * drbfs)
        self.w += ((1-self.eta) * self.alpha * delta * rbfs)
        # no change to c for now!

class TabularRBFValueFn(ValueFn):
    def __init__(self,
                 alpha,
                 c,
                 w,
                 projector,
                 gamma,
                 sigma,
                 eta,
                 actions,
                 **kargs):
        super(RBFValueFn, self).__init__()
        self.alpha = alpha
        self.projector = projector
        self.gamma = gamma
        self.width = sigma **-2
        self.eta =  eta

        self.actions = actions

        self.c = [ np.array(c) for i in xrange(len(actions))]
        self.w = [ np.array(w) for i in xrange(len(actions))]


    def __call__(self, state, action):
        projected = self.projector(state)
        index = self.getactionindex(action)
        return self.w[index].dot(self.computeRBFs(projected, self.c[index]))

    def getactionindex(self, action):
        return np.argmax(np.all(self.actions == action, axis = 1))

    def computeRBFs(self, x, c):
        s =  np.sum( (x-c)**2, axis=1)
        s *= (-self.width)
        rbfs = np.exp(s)
        rbfs /= np.sum(rbfs)
        return rbfs

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if s_t == None:
            return

        index_t = self.getactionindex(a_t)
        phi_t = self.projector(s_t)
        rbfs = self.computeRBFs(phi_t, self.c[index_t])
        if s_tp1 == None:
            v_tp1 = 0
            drbfs = np.zeros_like(rbfs)
        else:
            index_tp1 = self.getactionindex(a_tp1)
            phi_tp1 = self.projector(s_tp1)
            rbfs_tp1 = self.computeRBFs(phi_tp1, self.c[index_tp1])
            v_tp1 = self.w[index_t].dot(rbfs_tp1)
            drbfs = rbfs_tp1 - rbfs

        v_t =  self.w[index_t].dot(rbfs)


        delta = r + self.gamma * v_tp1 - v_t
        self.w -= (self.eta * self.alpha * delta * drbfs)
        self.w += ((1-self.eta) * self.alpha * delta * rbfs)
        # no change to c for now!



class NeuroSFTD(ValueFn):
    def __init__(self, projector, **argk):
        super(NeuroSFTD, self).__init__()
        self.projector = projector
        self.gamma = argk.get('gamma', 0.9)
        if 'layers' not in argk:
            argk['layers'] = [projector.size, 30, 1]
        self.net = NeuralNet( **argk)

    def __call__(self, state, action):
        projected = self.projector(state, action)
        return self.net.evaluate(projected)[0]

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if s_t == None:
            return

        phi_t = self.projector(s_t, a_t)

        if s_tp1 == None:
            dphi = np.zeros_like(phi_t)
            v_tp1 = 0
        else:
            phi_tp1 = self.projector(s_tp1, a_tp1)
            v_tp1 = self.net.evaluate(phi_tp1)[0]
            dphi = phi_tp1 - phi_t

        v_t = self.net.evaluate(phi_t)[0]

        dV = v_tp1 * (1 - self.gamma) - r

        target = r + self.gamma * v_tp1

        self.net.backprop(target, dphi, dV)



class NeuroSFTD_Factory(object):
    def __init__(self, **argk):
        self.params = argk

    def __call__(self, **argk):
        params = dict(self.params)
        params.update([x for x in argk.items()])
        return NeuroSFTD( **params)
