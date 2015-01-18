import numpy as np
from control.statefbk import lqr

class SwingPendulum(object):
    min_pos = -np.pi
    max_pos = np.pi

    umax = 2.0
    mass = 1.0
    length = 1.0
    G = 9.8
    integ_rate = 0.01
    control_rate = 0.05
    
    required_up_time = 2.0
    up_range = np.pi/8.0
    max_speed = np.pi*3

    pos_start = np.pi/2.0
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

        return -np.cos(self.state[0]), next_state

    def reset(self):
        if self.random_start:
            self.state[:] = [np.random.uniform(self.state_range[0][0], self.state_range[1][0]),
                             0]
        else:
            self.state[:] = [self.pos_start, self.vel_start]

        self.uptime = 0
        return -np.cos(self.state[0]), self.state.copy()

    def update(self, action):
        torque = np.clip(action, *self.action_range)
        moment = self.mass*self.length**2
        for i in xrange(int(np.ceil(self.control_rate/self.integ_rate))):
            theta_acc = (torque - self.damping * self.state[1] \
                        - self.mass*self.G *self.length*np.sin(self.state[0]))/moment
            
            theta_delta_acc = self.integ_rate * theta_acc
            self.state[1] = np.clip(self.state[1] + theta_delta_acc, self.state_range[0][1], self.state_range[1][1])
            self.state[0] += self.state[1] * self.integ_rate
            self.adjustTheta()
            self.uptime = 0 if np.abs(self.state[0]) > self.up_range else self.uptime + self.integ_rate

    def get_energy(self, state = None):
        if state is None:
            state = self.state
        moment = self.mass*self.length**2
        E = 0.5*moment*state[1]**2 \
                    - self.mass*self.G*self.length*np.cos(state[0])
        return E

    def adjustTheta(self):
        if self.state[0] >= np.pi:
            self.state[0] -= 2*np.pi
        if self.state[0] < -np.pi:
            self.state[0] += 2*np.pi

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
    

class Energy_pumping(object):
    def __init__(self, pendulum, k=20.0):
        self.k = k
        self.pendulum = pendulum
        self.required = pendulum.mass*pendulum.G*pendulum.length
        
    def __call__(self, state):
        dE = self.pendulum.get_energy(state) - self.required
        return -self.k * state[1] * dE
    
class Swing_stabilize(object):
    def __init__(self, pendulum, k=20.0):
        self.pendulum = pendulum
        self.required = pendulum.mass*pendulum.G*pendulum.length
        moment = pendulum.mass * pendulum.length**2
        A = np.array([[0, 1], [pendulum.mass*pendulum.length*pendulum.G/moment, -pendulum.damping/moment]])
        B = np.array([[0], [1/moment]])
        self.K, self.S, _ = lqr(A, B, np.eye(2)*50, np.eye(1))
        self.pump = Energy_pumping(pendulum, k)
    
    def get_qbar(self, q):
        q_bar = q.copy()
        q_bar[0] = np.remainder(q[0], 2*np.pi) - np.pi
        return q_bar
    
    def __call__(self, state):
        q_bar = self.get_qbar(state)
        if q_bar.dot(self.S.dot(q_bar)) < 30:
            torque = -self.K.dot(q_bar)
        else:
            torque = self.pump(state)
        return torque        

class SwingPendulum_Factory(object):
    def __init__(self, **argk):
        self.param = argk

    def __call__(self, **argk):
        params = dict(self.param)
        params.update([x for x in argk.items()])
        return SwingPendulum(**params)
