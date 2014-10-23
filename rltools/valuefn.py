import numpy as np
import scipy
from rltools.pyneuralnet import NeuralNet

class ValueFn(object):
    def __init__(self):
        pass

    def __call__(self, state, actions):
        pass

class incrementalValueFn(ValueFn):
    def __init__(self):
        super(incrementalValueFn, self).__init__()

    def __call__(self, state, actions):
        pass

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        pass

class linearValueFn(ValueFn):
    def __init__(self, weights, projector):
        super(linearValueFn, self).__init__()
        self.theta= weights
        self.proj = projector
    def __call__(self, state):
        return self.proj(state).dot(self.theta)


def LSTDlambda(policy,
           environment,
           gamma,
           feature,
           projector,
           number_episodes,
           max_episode_length,
           lamb = 0.0,
           **args):
    phi = projector
    A = np.zeros((projector.size, projector.size))
    b = np.zeros(projector.size)

    for i in xrange(number_episodes):
        x_t = environment.reset()[1]
        p_t = phi(x_t)
        z = p_t
        t=0
        while not environment.isterminal() and t<max_episode_length:
            r, x_tp1 = environment.step(policy(x_t))
            p_tp1 = phi(x_tp1)
            A += np.outer(z, p_t - gamma*p_tp1)
            b += z * (feature(x_t) if feature != None else r)
            z = gamma*lamb * z + p_tp1
            x_t = x_tp1
            p_t = p_tp1
            t += 1

    theta = np.linalg.lstsq(A, b)[0]
    return linearValueFn(theta, phi)

# def BatchLSTDlambda(policy,
#            environment,
#            gamma,
#            feature,
#            projector,
#            lamb = 0.0,
#            batch
#            **args):
#     phi = projector
#     A = np.zeros((projector.size, projector.size))
#     b = np.zeros(projector.size)
#
#     for data in batch:
#         x_t = environment.reset()[1]
#         p_t = phi(x_t)
#         z = p_t
#         t=0
#         while not environment.isterminal() and t<max_episode_length:
#             x_tp1 = environment.step(policy(x_t))(1)
#             p_tp1 = phi(x_tp1)
#             A += np.outer(z, p_t - gamma*p_tp1)
#             b += z * feature(x_t)
#             z = gamma*lamb * z + p_tp1
#             x_t = x_tp1
#             p_t = p_tp1
#             t += 1
#
#     theta = np.linalg.solve(A, b)
#     return linearValueFn(theta, phi)

class LinearTD(ValueFn):
    def __init__(self,
                 num_actions,
                 projector,
                 alpha,
                 lamb,
                 gamma,
                 **argk):
        super(LinearTD, self).__init__()
        self.projector = projector
        self.gamma = gamma
        self.alpha = alpha
        self.lamb = lamb
        self.phi = projector
        self.theta = np.zeros((num_actions, projector.size))
        self.e = np.zeros_like(self.theta)
        self.num_actions = num_actions

    def __call__(self, state, action= None):
        if state == None:
            return 0
        elif action == None:
            if issubclass(state, np.uint):
                return np.sum(self.theta[:,state], axis=1)
            else:
                return self.theta.dot(self.phi(state))
        else:
            if issubclass(state, np.uint):
                return np.sum(self.theta[action,state])
            else:
                return self.theta[action,:].dot(self.phi(state))

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if s_t == None:
            self.e = np.zeros_like(self.theta)
            return

        self.e[a_t,:] += self.phi(s_t)
        delta = r + self.gamma*self(s_tp1, a_tp1) - self(s_t, a_t)
        self.theta += self.alpha*delta*self.e
        self.e *= self.gamma*self.lamb


class LinearTDPolicyMixture(ValueFn):
        def __init__(self,
                 num_actions,
                 projector,
                 alpha,
                 lamb,
                 gamma,
                 **argk):
        super(LinearTDPolicyMixture, self).__init__()
        self.projector = projector
        self.gamma = gamma
        self.alpha = alpha
        self.lamb = lamb
        self.phi = projector
        self.theta = np.zeros((num_actions, projector.size))
        self.e = np.zeros_like(self.theta)
        self.num_actions = num_actions

    def __call__(self, state, action = None):
        if state == None:
            return 0
        elif action == None:
            if issubclass(state, np.uint):
                return np.sum(self.theta[:,state], axis=1)
            else:
                return self.theta.dot(self.phi(state))
        else:
            if issubclass(state, np.uint):
                return np.sum(self.theta[action,state])
            else:
                return self.theta[action,:].dot(self.phi(state))

    def update(self, s_t, r, s_tp1, rho):
        if s_t == None:
            self.e[:,:] = 0
            return
        phi_t = self.phi(s_t)
        self.e *= self.gamma*self.lamb
        if issubclass(state, np.uint):
            self.e[:,phi_t] += rho[:,None]
        else:
             self.e += rho[:,None]*phi_t
        delta = r + self.gamma*self(s_tp1)- self(s_t)
        self.theta += self.alpha*delta[:,None]*self.e



class RBFValueFn(incrementalValueFn):
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

class TabularRBFValueFn(incrementalValueFn):
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
        super(TabularRBFValueFn, self).__init__()
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
            v_tp1 = self.w[index_tp1].dot(rbfs_tp1)
            drbfs = rbfs_tp1 - rbfs

        v_t =  self.w[index_t].dot(rbfs)


        delta = r + self.gamma * v_tp1 - v_t
        self.w[index_t] -= (self.eta * self.alpha * delta * drbfs)
        self.w[index_t] += ((1-self.eta) * self.alpha * delta * rbfs)
        # no change to c for now!



class NeuroSFTD(incrementalValueFn):
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

class TabularNeuroSFTD(incrementalValueFn):
    def __init__(self, actions, projector, **argk):
        super(TabularNeuroSFTD, self).__init__()
        self.projector = projector
        self.gamma = argk.get('gamma', 0.9)
        if 'layers' not in argk:
            argk['layers'] = [projector.size, 30, 1]

        self.actions = actions
        self.nets = [NeuralNet( **argk) for i in xrange(len(actions))]

    def __call__(self, state, action):
        projected = self.projector(state)
        index = self.getactionindex(action)
        return self.nets[index].evaluate(projected)[0]

    def getactionindex(self, action):
        return np.argmax(np.all(self.actions == action, axis = 1))

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if s_t == None:
            return

        index_t = self.getactionindex(a_t)
        phi_t = self.projector(s_t)

        if s_tp1 == None:
            dphi = np.zeros_like(phi_t)
            v_tp1 = 0
        else:
            phi_tp1 = self.projector(s_tp1)
            index_tp1 = self.getactionindex(a_tp1)
            v_tp1 = self.nets[index_tp1].evaluate(phi_tp1)[0]
            dphi = phi_tp1 - phi_t

        v_t = self.nets[index_t].evaluate(phi_t)[0]

        dV = v_tp1 * (1 - self.gamma) - r

        target = r + self.gamma * v_tp1

        self.nets[index_t].backprop(target, dphi, dV)



class TabularAvgRewNeuroSFTD(incrementalValueFn):
    def __init__(self, actions, projector, alphamu, **argk):
        super(TabularAvgRewNeuroSFTD, self).__init__()
        self.projector = projector
        self.alphamu = alphamu
        if 'layers' not in argk:
            argk['layers'] = [projector.size, 30, 1]

        self.actions = actions
        self.nets = [NeuralNet( **argk) for i in xrange(len(actions))]
        self.mu=None

    def __call__(self, state, action):
        projected = self.projector(state)
        index = self.getactionindex(action)
        return self.nets[index].evaluate(projected)[0]

    def getactionindex(self, action):
        return np.argmax(np.all(self.actions == action, axis = 1))

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if self.mu == None:
            self.mu = r
        else:
            self.mu = self.mu*(1-self.alphamu) + self.alphamu*r

        if s_t == None:
            return

        index_t = self.getactionindex(a_t)
        phi_t = self.projector(s_t)

        if s_tp1 == None:
            dphi = np.zeros_like(phi_t)
            v_tp1 = 0
        else:
            phi_tp1 = self.projector(s_tp1)
            index_tp1 = self.getactionindex(a_tp1)
            v_tp1 = self.nets[index_tp1].evaluate(phi_tp1)[0]
            dphi = phi_tp1 - phi_t

        v_t = self.nets[index_t].evaluate(phi_t)[0]

        dV = - r + self.mu

        target = r - self.mu + v_tp1

        self.nets[index_t].backprop(target, dphi, dV)

class TabularAvgRewNeuroSFTD_Factory(object):
    def __init__(self, **argk):
        self.params = argk

    def __call__(self, **argk):
        params = dict(self.params)
        params.update([x for x in argk.items()])
        params['actions'] = params['domain'].discrete_actions
        return TabularAvgRewNeuroSFTD( **params)


class TabularAvgRewSFTD(incrementalValueFn):
    def __init__(self, actions, projector, alpha, alphamu, eta, **argk):
        super(TabularAvgRewSFTD, self).__init__()
        self.projector = projector
        self.alpha = alpha
        self.alphamu = alphamu
        self.eta = eta

        self.actions = actions
        self.theta = [np.random.normal(loc = 0,
                                       scale = 0.05,
                                       size = projector.size)
                      for a in actions]
        self.mu=None

    def __call__(self, state, action):
        projected = self.projector(state)
        index = self.getactionindex(action)
        return self.theta[index].dot(projected)

    def getactionindex(self, action):
        return np.argmax(np.all(self.actions == action, axis = 1))

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if self.mu == None:
            self.mu = r
        else:
            self.mu = self.mu*(1-self.alphamu) + self.alphamu*r

        if s_t == None:
            return

        index_t = self.getactionindex(a_t)
        phi_t = self.projector(s_t)

        if s_tp1 == None:
            v_t = self.theta[index_t].dot(phi_t)
            v_tp1 = 0
            delta = r - self.mu + v_tp1 - v_t
            self.theta[index_t] += self.alpha * delta * phi_t
        else:
            phi_tp1 = self.projector(s_tp1)
            index_tp1 = self.getactionindex(a_tp1)
            v_tp1 = self.theta[index_tp1].dot(phi_tp1)
            v_t = self.theta[index_t].dot(phi_t)
            delta = r - self.mu + v_tp1 - v_t
            self.theta[index_t] += self.alpha * delta * phi_t
            self.theta[index_t] -= self.alpha * self.eta * delta * phi_tp1

class TabularAvgRewSFTD_Factory(object):
    def __init__(self, **argk):
        self.params = argk

    def __call__(self, **argk):
        params = dict(self.params)
        params.update([x for x in argk.items()])
        params['actions'] = params['domain'].discrete_actions
        return TabularAvgRewSFTD( **params)


class TabularAvgRewNSFTD(incrementalValueFn):
    def __init__(self, actions, projector, alpha, alphamu, eta, **argk):
        super(TabularAvgRewNSFTD, self).__init__()
        self.projector = projector
        self.alpha = alpha
        self.alphamu = alphamu
        self.eta = eta

        self.actions = actions
        self.theta = [np.random.normal(loc = 0,
                                       scale = 0.05,
                                       size = projector.size)
                      for a in actions]
        self.mu=None

    def __call__(self, state, action):
        projected = self.projector(state)
        index = self.getactionindex(action)
        return self.theta[index].dot(projected)

    def getactionindex(self, action):
        return np.argmax(np.all(self.actions == action, axis = 1))

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if self.mu == None:
            self.mu = r
        else:
            self.mu = self.mu*(1-self.alphamu) + self.alphamu*r

        if s_t == None:
            return

        index_t = self.getactionindex(a_t)
        phi_t = self.projector(s_t)

        if s_tp1 == None:
            v_t = self.theta[index_t].dot(phi_t)
            v_tp1 = 0
            delta = r - self.mu + v_tp1 - v_t
            self.theta[index_t] += self.alpha * delta * phi_t
        else:
            phi_tp1 = self.projector(s_tp1)
            index_tp1 = self.getactionindex(a_tp1)
            v_tp1 = self.theta[index_tp1].dot(phi_tp1)
            v_t = self.theta[index_t].dot(phi_t)
            delta = r - self.mu + v_tp1 - v_t
            beta = phi_tp1.dot(phi_tp1) + phi_t.dot(phi_t) - 2*(phi_tp1.dot(phi_t))
            self.theta[index_t] += self.alpha * delta/beta * phi_t
            self.theta[index_t] -= self.alpha * self.eta * delta/beta * phi_tp1

class TabularAvgRewNSFTD_Factory(object):
    def __init__(self, **argk):
        self.params = argk

    def __call__(self, **argk):
        params = dict(self.params)
        params.update([x for x in argk.items()])
        params['actions'] = params['domain'].discrete_actions
        return TabularAvgRewNSFTD( **params)



class NeuroSFTD_Factory(object):
    def __init__(self, **argk):
        self.params = argk

    def __call__(self, **argk):
        params = dict(self.params)
        params.update([x for x in argk.items()])
        return NeuroSFTD( **params)
