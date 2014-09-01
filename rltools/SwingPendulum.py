import numpy as np
from math import pi
import math

class SwingPendulum(object):
    min_pos = -pi
    max_pos = pi

    umax = 2.0
    mass = 1.0
    length = 1.0
    G = 9.8
    timestep = 0.01
    required_up_time = 10.0
    up_range = pi/4.0
    max_speed = (pi/4.0)/timestep

    pos_start = pi/2.0
    vel_start = 0.0

    damping = 0.2

    state_range =[ np.array([min_pos, -max_speed]),
                   np.array([max_pos, max_speed])]


    action_range = [[-umax], [umax]]

    __discrete_actions = [np.array([-umax]),
                          np.array([0]),
                          np.array([umax])]

    def __init__(self,
                 random_start = False,
                 required_up_time=10.0,
                 **argk):
        self.state= np.zeros(2)
        self.random_start = random_start
        self.required_up_time = required_up_time
        self.reset()


    def step(self, action):
        self.update(action)
        if self.inGoal():
            next_state = None
        else:
            next_state = self.state.copy()

        return np.cos(self.state[0]), next_state

    def reset(self):
        if self.random_start:
            self.state[:] = [np.random.uniform(self.state_range[0][0], self.state_range[1][0]),
                             0]
        else:
            self.state[:] = [self.pos_start, self.vel_start]

        self.uptime = 0
        return np.cos(self.state[0]), self.state.copy()

    def update(self, action):
        torque = np.clip(action, *self.action_range)
        theta_acc = self.timestep * (- self.state[1]*self.damping
                        + self.mass * self.G * self.length * math.sin(self.state[0])
                        + torque)
        self.state[1] = np.clip(self.state[1] + theta_acc, self.state_range[0][1], self.state_range[1][1])
        self.state[0] += self.state[1] * self.timestep
        self.adjustTheta()
        self.uptime = 0 if np.abs(self.state[0]) > self.up_range else self.uptime + self.timestep

    def adjustTheta(self):
        if self.state[0] >= pi:
            self.state[0] -= 2*pi
        if self.state[0] < -pi:
            self.state[0] += 2*pi

    def inGoal(self):
        return self.uptime >= self.required_up_time

    def copy(self):
        newpendulum = SwingPendulum(random_start = self.random_start)
        newpendulum.state[:] = self.state
        return newpendulum

    @property
    def discrete_actions(self):
        return self.__discrete_actions

    @property
    def state_dim(self):
        return len(self.state_range[0])

    @property
    def action_dim(self):
        return len(self.action_range[0])

class SwingPendulum_Factory(object):
    def __init__(self, **argk):
        self.param = argk

    def __call__(self, **argk):
        params = dict(self.param)
        params.update([x for x in argk.items()])
        return SwingPendulum(**params)
