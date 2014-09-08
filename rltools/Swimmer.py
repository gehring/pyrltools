"""multi-link swimmer moving in a fluid. Taken from rlpy"""

import numpy as np
from itertools import product

__copyright__ = "Copyright 2013, RLPy http://www.acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann"


class Swimmer(object):

    """
    A swimmer consisting of a chain of d links connected by rotational joints.
    Each joint is actuated. The goal is to move the swimmer to a specified goal
    position.

    *States*:
        | 2 dimensions: position of nose relative to goal
        | d -1 dimensions: angles
        | 2 dimensions: velocity of the nose
        | d dimensions: angular velocities

    *Actions*:
        each joint torque is discretized in 3 values: -2, 0, 2

    .. note::
        adapted from Yuval Tassas swimmer implementation in Matlab available at
        http://www.cs.washington.edu/people/postdocs/tassa/code/

    .. seealso::
        Tassa, Y., Erez, T., & Smart, B. (2007).
        *Receding Horizon Differential Dynamic Programming.*
        In Advances in Neural Information Processing Systems.
    """
    dt = 0.03
    episodeCap = 1000
    discount_factor = 0.98

    def __init__(self, d=3, k1=7.5, k2=0.3, **kargs):
        """
        d:
            number of joints
        """
        self.d = d
        self.k1 = k1
        self.k2 = k2
        self.nose = 0
        self.masses = np.ones(d)
        self.lengths = np.ones(d)
        self.inertia = self.masses * self.lengths * self.lengths / 12.
        self.goal = np.zeros(2)

        # reward function parameters
        self.cu = 0.04
        self.cx = 2.

        Q = np.eye(self.d, k=1) - np.eye(self.d)
        Q[-1, :] = self.masses
        A = np.eye(self.d, k=1) + np.eye(self.d)
        A[-1, -1] = 0.
        self.P = np.dot(np.linalg.inv(Q), A * self.lengths[None, :]) / 2.

        self.U = np.eye(self.d) - np.eye(self.d, k=-1)
        self.U = self.U[:, :-1]
        self.G = np.dot(self.P.T * self.masses[None, :], self.P)

        # incidator variables for angles in a state representation
        self.angles = np.zeros(2 + self.d * 2 + 1, dtype=np.bool)
        self.angles[2:2 + self.d - 1] = True
        self.angles[-self.d - 2:] = True

        self.actions = [np.array(a) for a in product(*((d - 1) * [[-2., 0., 2]]))]
        self.actions_num = len(self.actions)

        self.statespace_limits = [[-15, 15]] * 2 + [[-np.pi, np.pi]] * (d - 1) \
            + [[-2, 2]] * 2 + [[-np.pi * 2, np.pi * 2]] * d
        self.statespace_limits = np.array(self.statespace_limits)
        self.continuous_dims = range(self.statespace_limits.shape[0])
        super(Swimmer, self).__init__()

    def reset(self):
        self.theta = np.zeros(self.d)
        self.pos_cm = np.array([10, 0])
        self.v_cm = np.zeros(2)
        self.dtheta = np.zeros(self.d)
        self.step_count = 0
        return 0, self.state

    @property
    def state(self):
        return np.hstack(self._body_coord())

    def isTerminal(self):
        return False

    def possibleActions(self, s=None):
        return np.arange(self.actions_num)


    def _body_coord(self):
        """
        transforms the current state into coordinates that are more
        reasonable for learning
        returns a 4-tupel consisting of:
        nose position, joint angles (d-1), nose velocity, angular velocities

        The nose position and nose velocities are referenced to the nose rotation.
        """
        cth = np.cos(self.theta)
        sth = np.sin(self.theta)
        M = self.P - 0.5 * np.diag(self.lengths)
        #  stores the vector from the center of mass to the nose
        c2n = np.array([np.dot(M[self.nose], cth), np.dot(M[self.nose], sth)])
        #  absolute position of nose
        T = -self.pos_cm - c2n - self.goal
        #  rotating coordinate such that nose is axis-aligned (nose frame)
        #  (no effect when  \theta_{nose} = 0)
        c2n_x = np.array([cth[self.nose], sth[self.nose]])
        c2n_y = np.array([-sth[self.nose], cth[self.nose]])
        Tcn = np.array([np.sum(T * c2n_x), np.sum(T * c2n_y)])

        #  velocity at each joint relative to center of mass velocity
        vx = -np.dot(M, sth * self.dtheta)
        vy = np.dot(M, cth * self.dtheta)
        #  velocity at nose (world frame) relative to center of mass velocity
        v2n = np.array([vx[self.nose], vy[self.nose]])
        #  rotating nose velocity to be in nose frame
        Vcn = np.array([np.sum((self.v_cm + v2n) * c2n_x),
                        np.sum((self.v_cm + v2n) * c2n_y)])
        #  angles should be in [-pi, pi]
        ang = np.mod(
            self.theta[1:] - self.theta[:-1] + np.pi,
            2 * np.pi) - np.pi
        return Tcn, ang, Vcn, self.dtheta

    def step(self, a):
        d = self.d
#         a = self.actions[a]
        s = np.hstack((self.pos_cm, self.theta, self.v_cm, self.dtheta))
        ns = rk4(
            dsdt, s, [0,
                      self.dt], a, self.P, self.inertia, self.G, self.U, self.lengths,
            self.masses, self.k1, self.k2)[-1]

        self.theta = ns[2:2 + d]
        self.v_cm = ns[2 + d:4 + d]
        self.dtheta = ns[4 + d:]
        self.pos_cm = ns[:2]
        self.step_count += 1
        return (self._reward(a),
                self.state if self.step_count < self.episodeCap else None)


    def _dsdt(self, s, a):
        """ just a convenience function for testing and debugging, not really used"""
        return dsdt(
            s, 0., a, self.P, self.inertia, self.G, self.U, self.lengths,
            self.masses, self.k1, self.k2)

    def _reward(self, a):
        """
        penalizes the l2 distance to the goal (almost linearly) and
        a small penalty for torques coming from actions
        """

        xrel = self._body_coord()[0] - self.goal
        dist = np.sum(xrel ** 2)
        return (
            - self.cx * dist / (np.sqrt(dist) + 1) - self.cu * np.sum(a ** 2)
        )

    def copy(self):
        newswim = Swimmer(self.d, self.k1, self.k2)
        newswim.theta[:] = self.theta
        newswim.pos_cm[:] = self.pos_cm
        newswim.v_cm[:] = self.v_cm
        newswim.dtheta[:] = self.dtheta
        newswim.step_count = self.step_count
        return newswim

    @property
    def state_range(self):
        # np.array(zip(*self.statespace_limits))
        return [np.array([ r[0] for r in self.statespace_limits]),
                np.array([ r[1] for r in self.statespace_limits])]

    @property
    def discrete_actions(self):
        return self.actions

    @property
    def state_dim(self):
        return len(self.state_range[0])

    @property
    def action_dim(self):
        return len(self.action_range[0])


def dsdt(s, t, a, P, I, G, U, lengths, masses, k1, k2):
    """
    time derivative of system dynamics
    """
    d = len(a) + 1
    theta = s[2:2 + d]
    vcm = s[2 + d:4 + d]
    dtheta = s[4 + d:]

    cth = np.cos(theta)
    sth = np.sin(theta)
    rVx = np.dot(P, -sth * dtheta)
    rVy = np.dot(P, cth * dtheta)
    Vx = rVx + vcm[0]
    Vy = rVy + vcm[1]

    Vn = -sth * Vx + cth * Vy
    Vt = cth * Vx + sth * Vy

    EL1 = np.dot((v1Mv2(-sth, G, cth) + v1Mv2(cth, G, sth)) * dtheta[None, :]
                 + (v1Mv2(cth, G, -sth) + v1Mv2(sth, G, cth)) * dtheta[:, None], dtheta)
    EL3 = np.diag(I) + v1Mv2(sth, G, sth) + v1Mv2(cth, G, cth)
    EL2 = - k1 * np.dot((v1Mv2(-sth, P.T, -sth) + v1Mv2(cth, P.T, cth)) * lengths[None, :], Vn) \
          - k1 * np.power(lengths, 3) * dtheta / 12. \
          - k2 * \
        np.dot((v1Mv2(-sth, P.T, cth) + v1Mv2(cth, P.T, sth))
               * lengths[None, :], Vt)
    ds = np.zeros_like(s)
    ds[:2] = vcm
    ds[2:2 + d] = dtheta
    ds[2 + d] = - \
        (k1 * np.sum(-sth * Vn) + k2 * np.sum(cth * Vt)) / np.sum(masses)
    ds[3 + d] = - \
        (k1 * np.sum(cth * Vn) + k2 * np.sum(sth * Vt)) / np.sum(masses)
    ds[4 + d:] = np.linalg.solve(EL3, EL1 + EL2 + np.dot(U, a))
    return ds


def v1Mv2(v1, M, v2):
    """
    computes diag(v1) dot M dot diag(v2).
    returns np.ndarray with same dimensions as M
    """
    return v1[:, None] * M * v2[None, :]

def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`. Taken from rlpy.

    *y0*
        initial state vector

    *t*
        sample times

    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``

    *args*
        additional arguments passed to the derivative function

    *kwargs*
        additional keyword arguments passed to the derivative function

    Example 1 ::

        ## 2D system

        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)

    Example 2::

        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)

        y0 = 1
        yout = rk4(derivs, y0, t)


    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0
    i = 0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout