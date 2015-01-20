import numpy as np
from numpy.random import random_sample

def discrete_sample(probabilities, size=1):
    bins = np.cumsum(probabilities)
    if size >1:
        return np.digitize(random_sample(size), bins)
    else:
        return np.digitize(random_sample(size), bins)[0]

class iSVD(object):
    def __init__(self, max_rank, shape, init = False, threshold = 1e-12):
        self.r = max_rank
        if not init:
            self.matrices = None
        else:
            self.matrices = (np.ones((shape[0],1))/shape[0],
                             np.ones((1,1)),
                             np.ones(1),
                             np.ones((1,1)),
                             np.ones((shape[1], 1))/shape[1])
        self.thres = threshold
        self.shape = shape

    def update_wrong(self, a, b):
        rank = self.r
        if self.matrices is None:
            self.matrices = ((a/np.linalg.norm(a)).reshape((-1,1)),
                             np.eye(1),
                             np.array([np.linalg.norm(a) * np.linalg.norm(b)]),
                             np.eye(1),
                             (b/np.linalg.norm(b)).reshape((-1,1)))
            return
        U, Up, S, Vp, V = self.matrices
        m = U.T.dot(a)
        p = a - U.dot(m)
        ra = np.linalg.norm(p)
        Pm = p/ra

        n = V.T.dot(b)
        q = b - V.dot(n)
        rb = np.linalg.norm(q)
        Qm = q/rb

        K =  np.hstack((m, ra))[:,None] * np.hstack((n, rb))[None,:] \
                    + np.diag(np.hstack((S, 0)))

        C, Sp, Dt = np.linalg.svd(K, full_matrices = False)
        D = Dt.T

        if np.abs(Sp[-1]) < self.thres or Sp.shape[0] > rank:
            Up = Up.dot(C[:-1, :-1])
            Vp = Vp.dot(D[:-1, :-1])
            S = Sp[:-1]
        else:
            newUp = np.zeros((Up.shape[0]+1, Up.shape[1]+1))
            newUp[:-1,:-1] = Up
            newUp[-1, -1] = 1
            Up = newUp.dot(C)
            U = np.hstack((U, Pm.reshape((-1,1))))

            newVp = np.zeros((Vp.shape[0]+1, Vp.shape[1]+1))
            newVp[:-1,:-1] = Vp
            newVp[-1, -1] = 1
            Vp = newVp.dot(D)
            V = np.hstack((V, Qm.reshape((-1,1))))

            S = Sp

        self.matrices = (U, Up, S, Vp, V)

    def update(self, a, b):
        rank = self.r
        if self.matrices is None:
            self.matrices = ((a/np.linalg.norm(a)).reshape((-1,1)),
                             np.eye(1),
                             np.array([np.linalg.norm(a) * np.linalg.norm(b)]),
                             np.eye(1),
                             (b/np.linalg.norm(b)).reshape((-1,1)))
            return
        U, Up, S, Vp, V = self.matrices
        m = U.T.dot(a)
        p = a - U.dot(m)
        ra = np.linalg.norm(p)
        Pm = p/ra

        n = V.T.dot(b)
        q = b - V.dot(n)
        rb = np.linalg.norm(q)
        Qm = q/rb

        K =  np.hstack((m, ra))[:,None] * np.hstack((n, rb))[None,:] \
                    + np.diag(np.hstack((S, 0)))

        C, Sp, Dt = np.linalg.svd(K, full_matrices = False)
        D = Dt.T
        Rot = D
        if np.abs(Sp[-1]) < self.thres or Sp.shape[0] > rank:
            U = U.dot(C[:-1, :-1])
            V = V.dot(D[:-1, :-1])
            S = Sp[:-1]
        else:
            newUp = np.zeros((Up.shape[0]+1, Up.shape[1]+1))
            newUp[:-1,:-1] = Up
            newUp[-1, -1] = 1
            Up = newUp
            U = np.hstack((U, Pm.reshape((-1,1)))).dot(C)

            newVp = np.zeros((Vp.shape[0]+1, Vp.shape[1]+1))
            newVp[:-1,:-1] = Vp
            newVp[-1, -1] = 1
            Vp = newVp
            V = np.hstack((V, Qm.reshape((-1,1)))).dot(D)

            S = Sp

        Vq, Vr = np.linalg.qr(V)
        Uq, Ur = np.linalg.qr(U)
        tU, tS, tV = np.linalg.svd(Ur.dot(np.diag(S)).dot(Vr.T), full_matrices = False)
        V = Vq.dot(tV)
        U = Uq.dot(tU)
        S = tS

        self.matrices = (U, Up, S, Vp, V)
        return Rot

    def get_decomp(self):
        if self.matrices is not None:
            U, Up, S, Vp, V = self.matrices
            return U.dot(Up), S, Vp.T.dot(V.T)
        else:
            return np.zeros((self.shape[0],1)), np.zeros(1), np.zeros((1,self.shape[1]))
