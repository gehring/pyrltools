from rltools.representation import IdentityProj
import numpy as np
from rltools.pyneuralnet import NeuralNet
# from GNeuralNet import NeuralNet

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
            phi_tp1 = np.zeros_like(phi_t)
            v_tp1 = 0
        else:
            phi_tp1 = self.projector(s_tp1, a_tp1)
            v_tp1 = self.net.evaluate(phi_tp1)[0]

        v_t = self.net.evaluate(phi_t)[0]

        dphi = phi_tp1 - phi_t
#         norm = np.linalg.norm(dphi)
#         dphi /= norm
        dV = v_t * (1 - self.gamma) - r
#         dV /= norm

        target = r + self.gamma * v_tp1

        self.net.backprop(target, dphi, dV)

class NeuroSFTD_Factory(object):
    def __init__(self, **argk):
        self.params = argk

    def __call__(self, **argk):
        params = dict(self.params)
        params.update([x for x in argk.items()])
        return NeuroSFTD( **params)
