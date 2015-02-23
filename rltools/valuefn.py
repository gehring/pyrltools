import numpy as np
import time
import scipy.sparse as sp
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

class linearQValueFn(ValueFn):
    def __init__(self, weights, projector):
        super(linearQValueFn, self).__init__()
        self.theta= weights
        self.proj = projector
    def __call__(self, state, action = None):
        phi_t = self.proj(state, action)
        if issubclass(phi_t.dtype.type, np.uint):
            return np.sum(self.theta[phi_t], 1)
        else:
            return phi_t.dot(self.theta)

class KernelBasedValueFn(ValueFn):
    def __init__(self, Qvalues, X, kernel, projector):
        super(KernelBasedValueFn, self).__init__()
        self.Q= Qvalues
        self.X = X
        self.kernel = kernel
        self.phi = projector
    def __call__(self, state, action = None):
        phi_t = self.phi(state, action)
        k = self.kernel(self.X, phi_t)
        return k.T.dot(self.Q)

class kernelLinearQValueFn(ValueFn):
    def __init__(self, alpha, projector, X_t, X_tp1, gamma, kernel):
        super(kernelLinearQValueFn, self).__init__()
        self.alpha= alpha
        self.proj = projector
        self.X_t = X_t
        self.X_tp1 = X_tp1
        self.gamma = gamma
        self.kernel = kernel
    def __call__(self, state, action = None):
        phi_t = self.proj(state, action)
        k = self.kernel(self.X_t, phi_t) \
                - self.gamma*self.kernel(self.X_tp1, phi_t)
        return self.alpha.dot(k)

class quadKernelLinearQValueFn(ValueFn):
    def __init__(self, kernel_valuefn):
        self.valuefn = kernel_valuefn
        self.kernel = kernel_valuefn.kernel
        self.X_t = kernel_valuefn.X_t
        self.X_tp1 = kernel_valuefn.X_tp1
        self.alpha = kernel_valuefn.alpha
        self.gamma = kernel_valuefn.gamma
        self.A = self.X_t[:,self.kernel.k2.indices]
        self.Ap = self.X_tp1[:,self.kernel.k2.indices]
        self.alpha = self.valuefn.alpha

    def __call__(self, state, action = None):
        return self.valuefn(state, action)

    def getmaxaction(self, state):
        if state is not None:
            b = self.kernel.k1(self.X_t, state.reshape((1,-1)))[:,0] * self.alpha
            bp = self.kernel.k1(self.X_tp1, state.reshape((1,-1)))[:,0] * self.alpha

            A = self.A
            Ap = self.Ap

            Coef = (A.T*b).dot(A) - self.gamma*(Ap.T*bp).dot(Ap)
            res = -b.dot(A)*self.kernel.k2.kernel.c \
                        + bp.dot(Ap)*self.kernel.k2.kernel.c*self.gamma
            opt_a = np.linalg.lstsq(Coef, res)[0]
            return opt_a
        else:
            return np.zeros(self.A.shape[1])

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

def LSQ(X_t, r_t, X_tp1, gamma, phi, **args):
    print 'Solving for Q-value function'
    start_time= time.clock()
    A = X_t.T.dot(X_t - gamma*X_tp1)
    b = X_t.T.dot(r_t)
    if isinstance(A, sp.spmatrix):
        theta = sp.linalg.lsmr(A, b, damp=0.001)[0]
    else:
        theta = np.linalg.lstsq(A, b)[0]
    print 'Solved in '+str(time.clock() - start_time) + ' seconds'
    return linearQValueFn(theta, phi)

def SFLSQ(X_t, r_t, X_tp1, gamma, phi, theta0= None, **args):
    print 'Solving for Q-value function'
    start_time= time.clock()
    A = (X_t - gamma*X_tp1)
    b = r_t
    if isinstance(A, sp.spmatrix):
        if theta0 is not None:
            b = r_t - A.dot(theta0)
#         C = A.copy()
#         C.data **= 2
#         c_sum =C.sum(0).view(type=np.ndarray)[0,:]
#         index = c_sum > 0.0
#         data = np.sqrt(c_sum[index])
#         D = sp.spdiags(1.0/data, 0, data.size, data.size)
#
#         sol = D.dot(sp.linalg.lsmr(A[:,index].dot(D), b, damp=0.01)[0])
#         theta = np.zeros(A.shape[1])
#         theta[index] = sol

        theta = sp.linalg.lsmr(A, b, damp=0.001)[0]
        if theta0 is not None:
            theta += theta0
    else:
        theta = np.linalg.lstsq(A, b)[0]
    print 'Solved in '+str(time.clock() - start_time) + ' seconds'
    return linearQValueFn(theta, phi)

class MLPValueFn(ValueFn):
    def __init__(self, mlp, actions, phi):
        self.mlp = mlp
        self.phi = phi
        self.actions = actions
    def __call__(self, s, a = None):
        if s is None:
            if a is not None:
                return 0
            else:
                return np.zeros(len(self.actions))
        else:
            if a is not None:
                return self.mlp(self.phi(s,a))
            else:
                sa = np.vstack(( self.phi(s,a) for a in self.actions))
                return self.mlp(sa)

class MLPSparseValueFn(ValueFn):
    def __init__(self, mlp, actions, phi):
        self.mlp = mlp
        self.phi = phi
        self.actions = actions
    def __call__(self, s, a = None):
        if s is None:
            if a is not None:
                return 0
            else:
                return np.zeros(len(self.actions))
        else:
            if a is not None:
                return self.mlp(self.phi(s,a))
            else:
                sa = sp.vstack(( self.phi(s,a) for a in self.actions))
                return self.mlp(sa)

class SklearnValueFn(ValueFn):
    def __init__(self, clf, actions, phi):
        self.clf = clf
        self.phi = phi
        self.actions = actions
    def __call__(self, s, a = None):
        if s is None:
            if a is not None:
                return 0
            else:
                return np.zeros(len(self.actions))
        else:
            if a is not None:
                return self.clf.predict(self.phi(s,a))
            else:
                sa = np.vstack(( self.phi(s,a) for a in self.actions))
                return self.clf.predict(sa)


class LinearTD(ValueFn):
    def __init__(self,
                 num_actions,
                 projector,
                 alpha,
                 lamb,
                 gamma,
                 replacing_trace = True,
                 **argk):
        super(LinearTD, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.lamb = lamb
        self.phi = projector
        self.theta = np.zeros((num_actions, projector.size))
        self.e = np.zeros_like(self.theta)
        self.replacing_trace = replacing_trace
        self.num_actions = num_actions

    def __call__(self, state, action= None):
        if state is None:
            return 0
        elif action is None:
            phi_t = self.phi(state)
            if issubclass(phi_t.dtype.type, np.uint):
                return np.sum(self.theta[:,phi_t], axis=1)
            else:
                return self.theta.dot(phi_t)
        else:
            phi_t = self.phi(state)
            if issubclass(phi_t.dtype.type, np.uint):
                return np.sum(self.theta[action,phi_t])
            else:
                return self.theta[action,:].dot(phi_t)

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if s_t == None:
            self.e = np.zeros_like(self.theta)
            return

        phi_t = self.phi(s_t)
        self.e *= self.gamma*self.lamb

        if issubclass(phi_t.dtype.type, np.uint):
            self.e[a_t,phi_t] += 1
        else:
            self.e[a_t,:] += phi_t

        if self.replacing_trace:
            self.e = np.clip(self.e, 0,1)

        delta = r + self.gamma*self(s_tp1, a_tp1) - self(s_t, a_t)
        self.theta += self.alpha*delta*self.e


class TDCOF(object):
    def __init__(self,
                 projector,
                 alpha,
                 alpha_R,
                 lamb,
                 gamma,
                 n_actions,
                 rank = None,
                 replacing_trace=True):
        self.gamma = gamma
        self.alpha = np.sqrt(alpha)
        self.alpha_R = alpha_R
        self.lamb = lamb
        self.phi = projector

        if rank is None:
            self.rank = max(min(50, projector.size/2), np.log(projector.size))
        else:
            self.rank = rank
        self.matrices = [(np.zeros((self.phi.size, 1)),
                         np.zeros(1),
                         np.zeros((self.phi.size, 1))) for i in xrange(n_actions)]

        self.buffer = [(np.zeros((self.phi.size, self.rank)),
                       np.zeros((self.phi.size, self.rank))) for i in xrange(n_actions)]
        self.count = np.zeros(n_actions, dtype='int')
        self.initialized = np.array([False]*n_actions)

        self.e = np.zeros((n_actions, self.phi.size))
        self.replacing_trace = replacing_trace
        self.R = np.zeros(self.phi.size)

    def __call__(self, state, action = None):
        phi_s = self.phi(state)

        if action is None:
            v = np.zeros(self.count.shape[0])
            for i, ((U,S,V), (A,B)) in enumerate(zip(self.matrices, self.buffer)):
                Uphi = phi_s.T.dot(U)
                phiA = phi_s.T.dot(A)
                v[i] = self.R.dot(V.dot(np.diag(S).dot(Uphi.T))) + self.R.dot(B.dot(phiA.T))
        else:
            U,S,V = self.matrices[action]
            A, B = self.buffer[action]
            Uphi = phi_s.T.dot(U)
            phiA = phi_s.T.dot(A)
            v = self.R.dot(V.dot(np.diag(S).dot(Uphi.T))) + self.R.dot(B.dot(phiA.T))
        return v


    def call_no_act(self, state):
        phi_s = self.phi(state)
        U,S,V = self.matrices[0]
        A, B = self.buffer[0]
        Uphi = phi_s.T.dot(U)
        phiA = phi_s.T.dot(A)
        v = self.R.dot(V.dot(np.diag(S).dot(Uphi.T))) + self.R.dot(B.dot(phiA.T))
        return v

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if s_t == None:
            self.e[:,:] = 0.0
            return

        phi_t = self.phi(s_t)

        self.e *= self.gamma*self.lamb
        self.e[a_t] += phi_t
        if self.replacing_trace:
            self.e = np.clip(self.e, 0, 1)

        # eval V(s_t)
        U,S,V = self.matrices[a_t]
        A, B = self.buffer[a_t]
        Uphi = phi_t.T.dot(U)
        phiA = phi_t.T.dot(A)
        v_t = V.dot(np.diag(S).dot(Uphi.T)) + B.dot(phiA.T)

        if s_tp1 is not None:
            # eval v_tp1
            phi_tp1 = self.phi(s_tp1)
            U,S,V = self.matrices[a_tp1]
            A, B = self.buffer[a_tp1]
            Uphi = phi_tp1.T.dot(U)
            phiA = phi_tp1.T.dot(A)
            v_tp1 = V.dot(np.diag(S).dot(Uphi.T)) + B.dot(phiA.T)

            delta = phi_t + self.gamma*v_tp1 - v_t
        else:
            delta = phi_t - v_t

        # update svd of the occupancy function
        ealpha = self.e * self.alpha
        delta *= self.alpha
        for a, (A,B) in enumerate(self.buffer):
            i = self.count[a]
            if i < self.rank:
                A[:,i] = ealpha[a]
                B[:,i] = delta
                self.count[a] += 1
            else:
                self.matrices[a] = self.update_svd(self.matrices[a],
                                                     self.buffer[a],
                                                     self.initialized[a])
                self.initialized[a] = True
                A[:,:] = 0
                B[:,:] = 0
                self.count[a] = 0
        R = self.R
        self.R = R +  self.alpha_R * (r - np.squeeze(phi_t.T.dot(self.R))) * phi_t
#         self.R = np.squeeze(np.array(self.R))

    def update_svd(self, matrices, ab_buffer, initialized):
        U,S,V = matrices
        A, B = ab_buffer
        if initialized:
            Q_a, R_a = np.linalg.qr(A - U.dot(U.T.dot(A)), mode='reduced')
            Q_b, R_b = np.linalg.qr(B - V.dot(V.T.dot(B)), mode='reduced')

            Ap = np.vstack((U.T.dot(A), R_a))
            Bp = np.vstack((V.T.dot(B), R_b))
            K = np.diag(np.hstack((S, np.zeros(R_a.shape[0])))) + Ap.dot(Bp.T)
            Up, Sp, Vp = np.linalg.svd(K, full_matrices = False)

            U = np.hstack((U, Q_a)).dot(Up)
            V = np.hstack((V, Q_b)).dot(Vp.T)

        else:
            Q_a, R_a = np.linalg.qr(A, mode='reduced')
            Q_b, R_b = np.linalg.qr(B, mode='reduced')
            Up, Sp, Vp = np.linalg.svd(R_a.dot(R_b.T), full_matrices = False)

            U = Q_a.dot(Up)
            V = Q_b.dot(Vp.T)
        S = Sp[:self.rank]
        U = U[:,0:self.rank]
        V = V[:,0:self.rank]
        return (U, S, V)

    def correct_orthogonality(self):
        U, S, V = self.matrices
        Vq, Vr = np.linalg.qr(V)
        Uq, Ur = np.linalg.qr(U)
        tU, tS, tV = np.linalg.svd(Ur.dot(np.diag(S)).dot(Vr.T), full_matrices = False)
        V = Vq.dot(tV)
        U = Uq.dot(tU)
        S = tS
        self.matrices = (U,S,V)
#         if self.use_U_only:
#             self.R = self.R.dot(tV)
    def get_values(self):
        v = np.zeros((self.phi.size, self.count.shape[0]))
        for i, ((U,S,V), (A,B)) in enumerate(zip(self.matrices, self.buffer)):
            v[:,i] = U.dot(np.diag(S).dot(V.T.dot(self.R))) + A.dot(B.T.dot(self.R))
        return v

    @property
    def theta(self):
        return self.get_values().T

class gradTDCOF(object):
    def __init__(self,
                 projector,
                 alpha,
                 alpha_R,
                 lamb,
                 gamma,
                 n_actions,
                 truth,
                 rank = None,
                 replacing_trace=True):
        self.gamma = gamma
        self.alpha = np.sqrt(alpha)
        self.alpha_R = alpha_R
        self.lamb = lamb
        self.phi = projector

        self.truth = truth
        if rank is None:
            self.rank = max(min(50, projector.size/2), np.log(projector.size))
        else:
            self.rank = rank
        self.matrices = [(np.zeros((self.phi.size, 1)),
                         np.zeros(1),
                         np.zeros((self.phi.size, 1))) for i in xrange(n_actions)]

        self.buffer = [(np.zeros((self.phi.size, self.rank)),
                       np.zeros((self.phi.size, self.rank))) for i in xrange(n_actions)]
        self.count = np.zeros(n_actions, dtype='int')
        self.initialized = np.array([False]*n_actions)

        self.e = np.zeros((n_actions, self.phi.size))
        self.replacing_trace = replacing_trace
        self.R = np.zeros(self.phi.size)

    def __call__(self, state, action = None):
        phi_s = self.phi(state)

        if action is None:
            v = np.zeros(self.count.shape[0])
            for i, ((U,S,V), (A,B)) in enumerate(zip(self.matrices, self.buffer)):
                Uphi = phi_s.T.dot(U)
                phiA = phi_s.T.dot(A)
                v[i] = self.R.dot(V.dot(np.diag(S).dot(Uphi.T))) + self.R.dot(B.dot(phiA.T))
        else:
            U,S,V = self.matrices[action]
            A, B = self.buffer[action]
            Uphi = phi_s.T.dot(U)
            phiA = phi_s.T.dot(A)
            v = self.R.dot(V.dot(np.diag(S).dot(Uphi.T))) + self.R.dot(B.dot(phiA.T))
        return v


    def call_no_act(self, state):
        phi_s = self.phi(state)
        U,S,V = self.matrices[0]
        A, B = self.buffer[0]
        Uphi = phi_s.T.dot(U)
        phiA = phi_s.T.dot(A)
        v = self.R.dot(V.dot(np.diag(S).dot(Uphi.T))) + self.R.dot(B.dot(phiA.T))
        return v

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if s_t == None:
            self.e[:,:] = 0.0
            return

        phi_t = self.phi(s_t)



        # eval V(s_t)
        U,S,V = self.matrices[a_t]
        A, B = self.buffer[a_t]
        Uphi = phi_t.T.dot(U)
        phiA = phi_t.T.dot(A)
        v_t = V.dot(np.diag(S).dot(Uphi.T)) + B.dot(phiA.T)

        if s_tp1 is not None:
            # eval v_tp1
            phi_tp1 = self.phi(s_tp1)
            U,S,V = self.matrices[a_tp1]
            A, B = self.buffer[a_tp1]
            Uphi = phi_tp1.T.dot(U)
            phiA = phi_tp1.T.dot(A)
            v_tp1 = V.dot(np.diag(S).dot(Uphi.T)) + B.dot(phiA.T)

            delta = phi_t + self.gamma*v_tp1 - v_t
        else:
            delta = phi_t - v_t

        # update svd of the occupancy function
        self.e *= self.gamma*self.lamb
        U,S,V = self.truth #self.matrices[a_t]
        if s_tp1 is None or not self.initialized[a_t]:
            self.e[a_t] += phi_t
        else:
            self.e[a_t] += U.dot(np.diag(1.0/(S+0.00001)).dot(V.T.dot(phi_t)))
#             self.e[a_t] += self.project_inv_trans(U, S, V, phi_t)


        ealpha = self.e * self.alpha
        delta *= self.alpha
        for a, (A,B) in enumerate(self.buffer):
            i = self.count[a]

            if self.replacing_trace:
                self.e = np.clip(self.e, 0, 1)

            if i < self.rank:
                A[:,i] = ealpha[a]
                B[:,i] = delta
                self.count[a] += 1
            else:
                self.matrices[a] = self.update_svd(self.matrices[a],
                                                     self.buffer[a],
                                                     self.initialized[a])
                self.initialized[a] = True
                A[:,:] = 0
                B[:,:] = 0
                self.count[a] = 0
        R = self.R
        self.R = R +  self.alpha_R * (r - np.squeeze(phi_t.T.dot(self.R))) * phi_t
#         self.R = np.squeeze(np.array(self.R))

    def update_svd(self, matrices, ab_buffer, initialized):
        U,S,V = matrices
        A, B = ab_buffer
        if initialized:
            Q_a, R_a = np.linalg.qr(A - U.dot(U.T.dot(A)), mode='reduced')
            Q_b, R_b = np.linalg.qr(B - V.dot(V.T.dot(B)), mode='reduced')

            Ap = np.vstack((U.T.dot(A), R_a))
            Bp = np.vstack((V.T.dot(B), R_b))
            K = np.diag(np.hstack((S, np.zeros(R_a.shape[0])))) + Ap.dot(Bp.T)
            Up, Sp, Vp = np.linalg.svd(K, full_matrices = False)

            U = np.hstack((U, Q_a)).dot(Up)
            V = np.hstack((V, Q_b)).dot(Vp.T)

        else:
            Q_a, R_a = np.linalg.qr(A, mode='reduced')
            Q_b, R_b = np.linalg.qr(B, mode='reduced')
            Up, Sp, Vp = np.linalg.svd(R_a.dot(R_b.T), full_matrices = False)

            U = Q_a.dot(Up)
            V = Q_b.dot(Vp.T)
        S = Sp[:self.rank]
        U = U[:,0:self.rank]
        V = V[:,0:self.rank]
        return (U, S, V)

    def correct_orthogonality(self):
        U, S, V = self.matrices
        Vq, Vr = np.linalg.qr(V)
        Uq, Ur = np.linalg.qr(U)
        tU, tS, tV = np.linalg.svd(Ur.dot(np.diag(S)).dot(Vr.T), full_matrices = False)
        V = Vq.dot(tV)
        U = Uq.dot(tU)
        S = tS
        self.matrices = (U,S,V)
#         if self.use_U_only:
#             self.R = self.R.dot(tV)
    def get_values(self):
        v = np.zeros((self.phi.size, self.count.shape[0]))
        for i, ((U,S,V), (A,B)) in enumerate(zip(self.matrices, self.buffer)):
            v[:,i] = U.dot(np.diag(S).dot(V.T.dot(self.R))) + A.dot(B.T.dot(self.R))
        return v

    def project_inv_trans(self, U,S,V, x, ratio=0.999):
        cs = np.cumsum(S)
        cs /= cs[-1]
        Sp = S[ (cs <= ratio).nonzero()]
        return U[:,0:Sp.size].dot(np.diag(1.0/(Sp+0.01)).dot(V[:,:Sp.size].T.dot(x)))

    @property
    def theta(self):
        return self.get_values().T

class TDOF(object):
    def __init__(self,
                 projector,
                 alpha,
                 alpha_R,
                 lamb,
                 gamma,
                 n_actions,
                 replacing_trace=True):
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_R = alpha_R
        self.lamb = lamb
        self.phi = projector

        self.matrices = [np.zeros((self.phi.size, self.phi.size))for i in xrange(n_actions)]


        self.e = np.zeros((n_actions, self.phi.size))
        self.replacing_trace = replacing_trace
        self.R = np.zeros(self.phi.size)

    def __call__(self, state, action = None):
        phi_s = self.phi(state)

        if action is None:
            v = np.zeros(self.count.shape[0])
            for i, Theta in enumerate(self.matrices):
                phiTheta = phi_s.dot(Theta)
                v[i] = self.R.dot(phiTheta.T)
        else:
            Theta = self.matrices[action]
            phiTheta = phi_s.dot(Theta)
            v = self.R.dot(phiTheta.T)
        return v

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if s_t == None:
            self.e[:,:] = 0.0
            return

        phi_t = self.phi(s_t)

        self.e *= self.gamma*self.lamb
        self.e[a_t] += phi_t
        if self.replacing_trace:
            self.e = np.clip(self.e, 0, 1)

        # eval V(s_t)
        Theta = self.matrices[a_t]
        phiTheta = phi_t.dot(Theta)
        v_t = phiTheta.T

        if s_tp1 is not None:
            # eval v_tp1
            phi_tp1 = self.phi(s_tp1)
            Theta = self.matrices[a_t]
            phiTheta = phi_tp1.dot(Theta)
            v_tp1 = phiTheta.T

            delta = phi_t + self.gamma*v_tp1 - v_t
        else:
            delta = phi_t - v_t

        # update svd of the occupancy function
        for i, Theta in enumerate(self.matrices):
            Theta += self.alpha*(self.e[i,:].reshape((-1,1))).dot(delta.reshape((1,-1)))
        R = self.R
        self.R = R +  self.alpha_R * (r - np.squeeze(phi_t.T.dot(self.R))) * phi_t
#         self.R = np.squeeze(np.array(self.R))

    def get_values(self):
        v = np.zeros((self.phi.size, self.count.shape[0]))
        for i, ((U,S,V), (A,B)) in enumerate(zip(self.matrices, self.buffer)):
            v[:,i] = U.dot(np.diag(S).dot(V.T.dot(self.R))) + A.dot(B.T.dot(self.R))
        return v

class TDSR(object):
    def __init__(self,
                 projector,
                 alpha,
                 alpha_R,
                 lamb,
                 gamma,
                 rank = None,
                 threshold = 1e-6,
                 replacing_trace=True,
                 use_U_only = False):
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_R = alpha_R
        self.lamb = lamb
        self.phi = projector

        if rank is None:
            self.rank = max(min(50, projector.size/2), np.log(projector.size))
        else:
            self.rank = rank
        self.matrices = (np.zeros((self.phi.size, 1)),
                         np.zeros(1),
                         np.zeros((self.phi.size, 1)))
        self.initialized = False
        self.threshold = threshold

        self.e = np.zeros(self.phi.size)
        self.replacing_trace = replacing_trace

        self.use_U_only = use_U_only
        if use_U_only:
            self.R = np.zeros(1)
        else:
            self.R = np.zeros(projector.size)

    def __call__(self, state, action = None):
        phi_sa = self.phi(state, action)

        U,S,V = self.matrices
        Uphi = phi_sa.T.dot(U)
        if self.use_U_only:
#             v = self.R.dot(np.diag(S).dot(Uphi.T))
            v = self.R.dot((Uphi.T))
        else:
            v = self.R.dot(V.dot(np.diag(S).dot(Uphi.T)))
        return v


    def call_no_act(self, state):
        phi_sa = self.phi(state)
        U,S,V = self.matrices
        Uphi = phi_sa.T.dot(U)
        if self.use_U_only:
            v = self.R.dot(np.diag(S).dot(Uphi.T))
        else:
            v = self.R.dot(V.dot(np.diag(S).dot(Uphi.T)))
        return v

    def update(self, s_t, a_t, r, s_tp1, a_tp1):
        if s_t == None:
            self.e[:] = 0.0
            return

        phi_t = self.phi(s_t, a_t)

        self.e *= self.gamma*self.lamb
        if phi_t.ndim > 1:
            e = self.e[:,None]
        else:
            e = self.e
        self.e = np.squeeze(np.array(e + phi_t))

        if self.replacing_trace:
            self.e = np.clip(self.e, 0, 1)

        U,S,V = self.matrices
        Uphi_t = phi_t.T.dot(U)
        if s_tp1 is not None:
            phi_tp1 = self.phi(s_tp1, a_tp1)
            Uphi_tp1 = phi_tp1.T.dot(U)
            delta = phi_t + self.gamma*V.dot(np.diag(S).dot(Uphi_tp1.T)) \
                        - V.dot(np.diag(S).dot(Uphi_t.T))
        else:
            delta = phi_t - V.dot(np.diag(S).dot(Uphi_t.T))

        delta = np.squeeze(np.array(delta))
        # update svd of the successor state representation
        ealpha = self.e * self.alpha
        if not self.initialized:
            self.matrices = ((ealpha/np.linalg.norm(ealpha)).reshape((-1,1)),
                             np.array([np.linalg.norm(ealpha) * np.linalg.norm(delta)]),
                             (delta/np.linalg.norm(delta)).reshape((-1,1)))
            self.initialized = True

        else:
            m = U.T.dot(ealpha)
            p = ealpha - U.dot(m)
            ra = np.linalg.norm(p)

            n = V.T.dot(delta)
            q = delta - V.dot(n)
            rb = np.linalg.norm(q)

            K = np.hstack((m, ra))[:,None] * np.hstack((n, rb))[None,:] \
                        + np.diag(np.hstack((S, 0)))

            C, Sp, Dt = np.linalg.svd(K, full_matrices = False)
            D = Dt.T
            if np.abs(Sp[-1]) < self.threshold or Sp.shape[0] > self.rank:
                U = U.dot(C[:-1, :-1])
                V = V.dot(D[:-1, :-1])
                S = Sp[:-1]
                if self.use_U_only:
                    self.R = self.R.dot(D[:-1, :-1])

            else:
                Pm = p/ra
                Qm = q/rb
                U = np.hstack((U, Pm.reshape((-1,1)))).dot(C)
                V = np.hstack((V, Qm.reshape((-1,1)))).dot(D)
                S = Sp
                if self.use_U_only:
                    self.R = np.hstack((self.R, 0)).dot(D)

            self.matrices = (U,S,V)
        Uphi_t = phi_t.T.dot(U)
        if self.use_U_only:
#             if s_tp1 is not None:
#                 Uphi_tp1 = phi_tp1.T.dot(U)
#                 delta = r + self.gamma*self.R.dot(np.diag(S).dot(Uphi_tp1.T)) \
#                         - self.R.dot(np.diag(S).dot(Uphi_t.T))
#             else:
#                 print self.R.dot(np.diag(S).dot(Uphi_t.T)), np.linalg.norm(np.diag(S).dot(Uphi_t.T))
#                 delta = r - self.R.dot(np.diag(S).dot(Uphi_t.T))
#             delta = np.squeeze(delta)
#             M = (np.diag(S).dot(Uphi_t.T)).squeeze()
#             self.R += self.alpha_R*delta*M
            if s_tp1 is not None:
                Uphi_tp1 = phi_tp1.T.dot(U)
                delta = r + self.gamma*self.R.dot((Uphi_tp1.T)) \
                        - self.R.dot((Uphi_t.T))
            else:
                delta = r - self.R.dot((Uphi_t.T))
            delta = np.squeeze(delta)
            M = ((Uphi_t.T)).squeeze()
            self.R += self.alpha_R*delta*M
        else:
            if phi_t.ndim > 1:
                R = self.R[:,None]
            else:
                R = self.R
            self.R = R +  self.alpha_R * (r - np.squeeze(phi_t.T.dot(self.R))) * phi_t
            self.R = np.squeeze(np.array(self.R))

    def update_no_act(self, s_t, r, s_tp1):
        if s_t == None:
            self.e[:] = 0.0
            return

        phi_t = self.phi(s_t)

        self.e *= self.gamma*self.lamb
        if phi_t.ndim > 1:
            e = self.e[:,None]
        else:
            e = self.e
        self.e = np.squeeze(np.array(e + phi_t))

        if self.replacing_trace:
            self.e = np.clip(self.e, 0, 1)

        U,S,V = self.matrices
        Uphi_t = phi_t.T.dot(U)
        if s_tp1 is not None:
            phi_tp1 = self.phi(s_tp1)
            Uphi_tp1 = phi_tp1.T.dot(U)
            delta = phi_t + self.gamma*V.dot(np.diag(S).dot(Uphi_tp1.T)) \
                        - V.dot(np.diag(S).dot(Uphi_t.T))
        else:
            delta = phi_t - V.dot(np.diag(S).dot(Uphi_t.T))

        delta = np.squeeze(np.array(delta))* np.sqrt(self.alpha)
        # update svd of the successor state representation
        ealpha = self.e * np.sqrt(self.alpha)
        if not self.initialized:
            self.matrices = ((ealpha/np.linalg.norm(ealpha)).reshape((-1,1)),
                             np.array([np.linalg.norm(ealpha) * np.linalg.norm(delta)]),
                             (delta/np.linalg.norm(delta)).reshape((-1,1)))
            self.initialized = True

        else:
            m = U.T.dot(ealpha)
            p = ealpha - U.dot(m)
            ra = np.linalg.norm(p)

            n = V.T.dot(delta)
            q = delta - V.dot(n)
            rb = np.linalg.norm(q)

#             print ra, rb
#             print p
#             print q
            K = np.hstack((m, ra))[:,None] * np.hstack((n, rb))[None,:] \
                        + np.diag(np.hstack((S, 0)))
            if np.any(np.isinf(K)):
                print 's', ra, rb
            C, Sp, Dt = np.linalg.svd(K, full_matrices = False)
            D = Dt.T
            if np.abs(Sp[-1]) < self.threshold or Sp.shape[0] > self.rank:
                U = U.dot(C[:-1, :-1])
                V = V.dot(D[:-1, :-1])
                S = Sp[:-1]
                if self.use_U_only:
                    self.R = self.R.dot(D[:-1, :-1])

            else:
                Pm = p/ra
                Qm = q/rb
                U = np.hstack((U, Pm.reshape((-1,1)))).dot(C)
                V = np.hstack((V, Qm.reshape((-1,1)))).dot(D)
                S = Sp
                if self.use_U_only:
                    self.R = np.hstack((self.R, 0)).dot(D)

            self.matrices = (U,S,V)
            if np.any(np.isinf(U)) or np.any(np.isinf(S)) or np.any(np.isinf(V)):
                print ra, rb
        Uphi_t = phi_t.T.dot(U)
        if self.use_U_only:
            if s_tp1 is not None:
                Uphi_tp1 = phi_tp1.T.dot(U)
                delta = r + self.gamma*self.R.dot((Uphi_tp1.T)) \
                        - self.R.dot((Uphi_t.T))
            else:
                delta = r - self.R.dot((Uphi_t.T))
            delta = np.squeeze(delta)
            M = ((Uphi_t.T)).squeeze()
            self.R += self.alpha_R*delta*M
        else:
            if phi_t.ndim > 1:
                R = self.R[:,None]
            else:
                R = self.R
            self.R = R +  self.alpha_R * (r - np.squeeze(phi_t.T.dot(self.R))) * phi_t
            self.R = np.squeeze(np.array(self.R))


    def correct_orthogonality(self):
        U, S, V = self.matrices
        Vq, Vr = np.linalg.qr(V)
        Uq, Ur = np.linalg.qr(U)
        tU, tS, tV = np.linalg.svd(Ur.dot(np.diag(S)).dot(Vr.T), full_matrices = False)
        V = Vq.dot(tV)
        U = Uq.dot(tU)
        S = tS
        self.matrices = (U,S,V)
#         if self.use_U_only:
#             self.R = self.R.dot(tV)

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
            phi_t = self.phi(state)
            if issubclass(phi_t.dtype.type, np.uint):
                return np.sum(self.theta[:,phi_t], axis=1)
            else:
                return self.theta.dot(phi_t)
        else:
            phi_t = self.phi(state)
            if issubclass(phi_t.dtype.type, np.uint):
                return np.sum(self.theta[action,phi_t])
            else:
                return self.theta[action,:].dot(phi_t)


    def update(self, s_t, r, s_tp1, rho):
        if s_t == None:
            self.e[:,:] = 0
            return
        phi_t = self.phi(s_t)
        self.e *= self.gamma*self.lamb
        if issubclass(phi_t.dtype.type, np.uint):
            self.e[:,phi_t] += rho[:,None]
        else:
            self.e += rho[:,None]*phi_t
        delta = r + self.gamma*self(s_tp1)- self(s_t)
        self.theta += self.alpha*delta[:,None]*self.e



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
