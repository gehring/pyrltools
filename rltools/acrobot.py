import numpy as np
# from scipy.integrate import odeint
from scipy.integrate import ode

class Acrobot(object):

    umax = 10
    umin = -10

    dt = np.array([ 0.1])

    start_state = np.array([0,0,0,0])
    __discrete_actions = [np.array([umin]),
                          np.array([0]),
                          np.array([umax])]

    action_range = [np.array([umin]),
                    np.array([umax]),]

    steps = 100


    def __init__(self,
                 random_start = False,
                 max_episode = 1000,
                 m1 = 1,
                 m2 = 1,
                 l1 = 1,
                 l2 = 1,
                 g = 9.81,
                 b = 0.0,
                 **argk):
        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.g =g
        self.b =b
        self.Ic1 = m1*l1**2/3
        self.Ic2 = m2*l2**2/3

        self.solver = ode(self.state_dot)
        self.solver.set_integrator('dop853')

        self.state = np.zeros(4)
        self.random_start = random_start
        self.max_episode = max_episode
        self.reset()

    def step(self, action):
        self.update(action)
        if self.isterminal():
            next_state = None
        else:
            next_state = self.state.copy()

        self.step_count += 1

        return -1, next_state

    def reset(self):
        if self.random_start:
            self.state[:] = [np.random.uniform(self.state_range[0][0], self.state_range[1][0]),
                             np.random.uniform(self.state_range[0][1], self.state_range[1][1]),
                             np.random.uniform(self.state_range[0][2], self.state_range[1][2]),
                             np.random.uniform(self.state_range[0][3], self.state_range[1][3])]
        else:
            self.state[:] = self.start_state

        self.step_count = 0
        self.t = 0
        self.solver.set_initial_value(self.state)

        return 0, self.state.copy()

    def state_dot(self, t, q, u):
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        lc1 = self.l1/2
        lc2 = self.l2/2
        I1 = self.Ic1 + m1*lc1**2
        I2 = self.Ic2 + m2*lc2**2

        g = self.g
        c = np.cos(q[:2])
        s = np.sin(q[:2])
        s12 = np.sin(q[0]+q[1])
#         m2l1l2c2 = m2*l1*l2*c2
        m2l1lc2 = m2*l1*lc2
        # double pendulum
#         a = (m1+m2)*l1**2 + m2*l2**2 + 2*m2l1l2c2
#         b =  m2*l2**2 + m2l1l2c2
#         c = m2*l2**2 + m2l1l2c2
#         d = m2*l2**2
#         Hinv= np.array(((d, -b),
#                      (-c, a))
#                     )/ (a*d - b*c)
#         C= np.array(((self.b, -m2*l1*l2*(2*q[2]+q[3])*s2),
#                      (m2*l1*l2*q[2]*s2, self.b))
#                     )
#         G = g* np.array(((m1+m2)*l1*s1 + m2*l2*s12, m2*l2*s12))
#
#         u = np.array((0, u[0]))

        a = I1 + I2 + m2*l1**2 + 2*m2l1lc2*c[1]
        b =  I2 + m2l1lc2*c[1]
        d = I2
        Hinv= np.array(((d, -b),
                     (-b, a))
                    )/ (a*d - b*c)
        C= np.array(((self.b -2*m2l1lc2*s[1]*q[3], -m2l1lc2*q[3]*s[1]),
                     (m2l1lc2*q[2]*s[1], self.b))
                    )
        G = g* np.array(((m1*lc1 + m2*l1)*s[0] + m2*lc2*s12, m2*lc2*s12))

        u = np.array((0, u[0]))

        qdot = Hinv.dot( u - G- C.dot(q[2:]))

        return np.hstack((q[2:], qdot))

    def update(self, action):
        u = np.clip(action, *self.action_range)
#         self.state = odeint( lambda x, t: self.state_dot(x, u), y0 = self.state, t = np.hstack(((0.0), self.dt)))[-1]
#         self.state[:2] = np.remainder(self.state[:2], 2*np.pi)
        self.solver.set_f_params(u)
#         self.solver.set_initial_value(self.state)
        self.state = self.solver.integrate(self.t + self.dt, step=self.steps)
        self.t += self.dt
        self.state[:2] = np.remainder(self.state[:2], 2*np.pi)

    def inGoal(self):
        pass

    def copy(self):
        pass

    def isterminal(self):
        pass

    @property
    def discrete_actions(self):
        return self.__discrete_actions

    @property
    def state_dim(self):
        return len(self.state_range[0])

    @property
    def action_dim(self):
        return len(self.action_range[0])
