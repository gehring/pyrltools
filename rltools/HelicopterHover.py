import numpy as np
import copy
from math import pi
from itertools import product


class Quaternion(object):
    def __init__(self, xyzw):
        if xyzw.size <4:
            self.xyzw = np.zeros(4)
            self.xyzw[:3] = xyzw
        else:
            self.xyzw = xyzw.copy()

    def conj(self):
        xyzw= self.xyzw
        xyzw[:3] *= -1.0
        return Quaternion(xyzw)

    def complex_part(self):
        return self.xyzw[:3]

    def mult(self, quat):
        xyzw = np.zeros(4)
        xyzw[0] = self.xyzw[3] * quat.xyzw[0] + self.xyzw[0] * quat.xyzw[3] \
                    + self.xyzw[1] * quat.xyzw[2] - self.xyzw[2] * quat.xyzw[1]
        xyzw[1] = self.xyzw[3] * quat.xyzw[1] - self.xyzw[0] * quat.xyzw[2] \
                    + self.xyzw[1] * quat.xyzw[3] + self.xyzw[2] * quat.xyzw[0]
        xyzw[2] = self.xyzw[3] * quat.xyzw[2] + self.xyzw[0] * quat.xyzw[1] \
                    - self.xyzw[1] * quat.xyzw[0] + self.xyzw[2] * quat.xyzw[3]
        xyzw[3] = self.xyzw[3] * quat.xyzw[3] - self.xyzw[0] * quat.xyzw[0] \
                    - self.xyzw[1] * quat.xyzw[1] - self.xyzw[2] * quat.xyzw[2]
        return Quaternion(xyzw)

    def rotate(self, x):
        return self.mult(Quaternion(x)).mult(self.conj()).complex_part()

    def express_in_frame(self, x):
        return self.conj().rotate(x)

    def copy(self):
        return Quaternion(self.xyzw)


class HelicopterHover(object):
    MAX_VEL = 5.0
    MAX_POS = 20.0
    MAX_RATE = 4 * pi
    MAX_QUAT = 1.0
    MIN_QW_BEFORE_HITTING_TERMINAL_STATE = np.cos(15.0 * pi / 180.0)
    MAX_ACTION = 1.0
    WIND_MAX = 5.0
    WIND = np.zeros(2)
    __state_range = [np.array([-MAX_VEL] * 3
                            + [-MAX_POS] * 3
                            + [-MAX_RATE] * 3
                            + [-MAX_QUAT] * 4),
                   np.array([MAX_VEL] * 3
                            + [MAX_POS] * 3
                            + [MAX_RATE] * 3
                            + [MAX_QUAT] * 4)]
    drag = np.array([0.18, 0.42, 0.49, 12.78, 10.12, 8.16])
    u_coeff = np.array([33.04, -33.32, 70.54, -42.15])
    tail_thrust = -0.54
    d_t = 0.1
    max_steps = 6000
    num_steps = 0
    vel = np.zeros(3)
    pos = np.zeros(3)
    ang_rate = np.zeros(3)
    q = Quaternion(np.zeros(4))
    noise = np.zeros(6)


    def __init__(self,
                 noise_memory = 0.8,
                 noise_mult = 2.0,
                 noise_std = None,
                 max_steps = 6000,
                 **kargs):
        if noise_std == None:
            noise_std = np.array([0.1941, 0.2975, 0.6058, 0.1508, 0.2492, 0.0734])
        self.noise_mult = noise_mult
        self.noise_std = noise_std
        self.noise_memory = noise_memory
        self.max_steps = max_steps

    def reset(self):
        self.vel = np.zeros(3)
        self.pos = np.zeros(3)
        self.ang_rate = np.zeros(3)
        self.q = Quaternion(np.zeros(4))
        self.num_steps = 0
        self.noise = np.zeros(6)

    def box_mull(self):
        x = np.random.uniform( size = (2,6))
        return np.sqrt( -2.0 * np.log(x[0,:])) * np.cos( 2.0 * pi * x[1,:])

    def rand_minus1_plus1(self):
        x = np.random.uniform()
        return 2.0 * x - 1.0

    def makeObs(self):
        obs = np.empty(12)
        obs[:3] = self.q.express_in_frame(self.vel)
        obs[3:6] = self.q.express_in_frame(self.pos)
        obs[6:9] = self.ang_rate
        obs[9:12] = self.q.xyzw[:3]

        return np.clip(obs, self.__state_range[0][:-1], self.__state_range[1][:-1])

    def update(self, a):
        a = np.clip(a, -1, 1)

        self.noise = self.noise * self.noise_memory\
                        + ((1.0 - self.noise_memory) * self.box_mull()
                           * self.noise_mult * self.noise_std)

        dt = 0.01
        for i in xrange(10):
            # intergrate position
            self.pos += self.vel * dt

            # rotate frame of velocity
            uvw = self.q.express_in_frame(self.vel)

            # compute wind resistance from vel and background wind
            wind_ned = np.zeros(3)
            wind_ned[:2] = self.WIND
            wind_uvw = self.q.express_in_frame(wind_ned)

            uvw_force_from_heli = -self.drag[:3] * (uvw - wind_uvw) + self.noise[:3]
            uvw_force_from_heli[1] += self.tail_thrust
            uvw_force_from_heli[2] += self.u_coeff[3] * a[3]

            # this is a correction to match the original implementation
            # I do not believe this is the correct thing to do but I include
            # it for consistency with previous results. If WIND is zero, then
            # this will have no effect
            uvw_force_from_heli[2] -= -self.drag[2] * wind_uvw[2]

            ned_force_from_heli = self.q.rotate(uvw_force_from_heli)

            # integrate vel
            self.vel += ned_force_from_heli * dt
            self.vel[2] += 9.81 * dt

            # update orientation
            axis_rot = self.ang_rate * dt
            rot_quat = self.to_quaternion(axis_rot)
            self.q = self.q.mult(rot_quat)

            # update angular velocity
            pqr_dot = self.drag[3:6] * self.ang_rate \
                        + self.u_coeff[:3] * a[:3] \
                        + self.noise[3:6]
            self.ang_rate += pqr_dot * dt

    def is_terminal(self):
        return np.any(np.abs(self.pos) > self.MAX_POS) \
            or np.any(np.abs(self.vel) > self.MAX_VEL) \
            or np.any(np.abs(self.ang_rate) > self.MAX_RATE) \
            or np.any(np.abs(self.q.xyzw[3]) > self.MIN_QW_BEFORE_HITTING_TERMINAL_STATE)

    def get_reward(self):
        return - self.pos.dot(self.pos) - self.vel.dot(self.vel) \
                    - self.ang_rate.dot(self.ang_rate) \
                    - self.q.xyzw[:3].dot(self.q.xyzw[:3])

    def get_terminal_reward(self):
        reward = 3*self.MAX_POS**2 + 3*self.MAX_VEL**2 + 3*self.MAX_RATE**2\
                        + (1-self.MIN_QW_BEFORE_HITTING_TERMINAL_STATE**2)
        reward *= (self.max_steps - self.num_steps)
        return reward

    def to_quaternion(self, x):
        angle = np.linalg.norm(x)
        if angle < 1e-4:
            quat = Quaternion(x/2.0)
            quat.xyzw[3] = np.sqrt(1 - quat.xyzw[:3].dot(quat.xyzw[:3]))
        else:
            quat = Quaternion(np.sin(angle/2.0)*(x/angle))
            quat.xyzw[3] = np.cos(angle)

    def copy(self):
        heli_copy = copy.copy(self)
        heli_copy.vel = self.vel.copy()
        heli_copy.pos = self.pos.copy()
        heli_copy.ang_rate = self.ang_rate.copy()
        heli_copy.q = self.q.copy()
        heli_copy.num_steps = self.num_steps.copy()
        heli_copy.noise = self.noise.copy()


class InfiniteHorizonHelicopter(object):

    num_steps_penalty = 30
    action_range = [np.array([-1.0]*4), np.array([1.0]*4)]


    def __init__(self, actions = None, **kargs):
        kargs['max_steps'] = float('inf')
        if actions == None:
            actions_per_dim = kargs['actions_per_dim']
            actions = product(*[np.linspace(-1.0, 1.0, actions_per_dim),
                                np.linspace(-1.0, 1.0, actions_per_dim),
                                np.linspace(-1.0, 1.0, actions_per_dim),
                                np.linspace(-1.0, 1.0, actions_per_dim)])
            self.__discrete_actions = map(np.array, actions)
        else:
            self.__discrete_actions = actions

        self.helicopter = HelicopterHover(**kargs)

    def reset(self):
        self.helicopter.reset()
        return 0, self.helicopter.makeObs()

    def step(self, a):
        self.helicopter.update(a)
        reward = self.helicopter.get_reward()
        if self.helicopter.is_terminal():
            reward *= 30
            self.helicopter.reset()

        return reward, self.helicopter.makeObs()

    def copy(self):
        IHheli = InfiniteHorizonHelicopter()
        IHheli.helicopter = self.helicopter.copy()
        IHheli.__discrete_actions = [ a.copy() for a in self.__discrete_actions]
        return IHheli

    @property
    def state_range(self):
        return [self.helicopter.__state_range[0][:-1],
                self.helicopter.__state_range[1][:-1]]

    @property
    def discrete_actions(self):
        return self.__discrete_actions

    @property
    def state_dim(self):
        return len(self.state_range[0])

    @property
    def action_dim(self):
        return len(self.action_range[0])

