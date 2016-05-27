import numpy as np
from scipy.integrate import odeint

class Cartpole(object):

    xmax = 20
    xmin = -20

    umax = 3
    umin = -3

    dt = np.array([0, 0.01])

    start_state = np.array([0.0,0.0,0.0,0.0])
    __discrete_actions = [np.array([umin]),
                          np.array([0]),
                          np.array([umax])]

    action_range = [np.array([umin]),
                    np.array([umax]),]

    thres = np.pi/4
    goal_range = [np.array([np.pi - thres, -thres, -thres, -thres]),
                  np.array([np.pi + thres, thres, thres, thres]),]

    x_max_speed = 10
    theta_max_speed = 20
    state_range = [np.array([xmin, 0, -x_max_speed, -theta_max_speed]),
                   np.array([xmax, np.pi*2, x_max_speed, theta_max_speed])]


    def __init__(self,
                 random_start = False,
                 max_episode = 1000,
                 m_c = 1,
                 m_p = 1,
                 l = 1,
                 g = 9.81,
                 x_damp = 0.1,
                 theta_damp = 0.1,
                 start_sampler = None,
                 **argk):
        self.l = l
        self.m_c = m_c
        self.m_p = m_p
        self.g = g
        self.x_damp = x_damp
        self.theta_damp = theta_damp

        self.state = np.zeros(4)
        self.random_start = random_start
        self.max_episode = max_episode
        self.start_sampler = start_sampler
        self.reset()

    def step(self, action):
        self.update(action)
        if self.isterminal():
            next_state = None
        else:
            next_state = self.state.copy()

        self.step_count += 1

        return (-1 if not self.inGoal() else 0), next_state

    def reset(self):
        if self.random_start:
            if self.start_sampler is None:
                self.state[:] = [np.random.uniform(self.state_range[0][0], self.state_range[1][0]),
                                 np.random.uniform(self.state_range[0][1], self.state_range[1][1]),
                                 np.random.uniform(self.state_range[0][2], self.state_range[1][2]),
                                 np.random.uniform(self.state_range[0][3], self.state_range[1][3])]
            else:
                self.state[:] = self.start_sampler()
        else:
            self.state[:] = self.start_state

        self.step_count = 0

        return 0, self.state.copy()

    def get_manipulator(self, q):
        m_c = self.m_c
        m_p = self.m_p
        l = self.l
        g = self.g
        theta = self.state[1::2]       
        x_damp = self.x_damp
        theta_damp = self.theta_damp
        
        s_theta = np.sin(theta[0])

        a = m_c + m_p
        b = m_p*l*np.cos(theta[0])
        d = m_p*l**2
        H= np.array(((a, b),
                     (b, d))
                    )
        C= np.array(((x_damp, -m_p*l*theta[1]*s_theta),
                     (0, theta_damp))
                    )
        G = np.array((0, m_p*g*l*s_theta))

        B = np.array((1, 0))
        return (H, C, G, B)

    def state_dot(self, q, t, u):
        H, C, G, B = self.get_manipulator(q)
        Hinv= -H
        Hinv[0,0]= H[1,1]
        Hinv[1,1]= H[0,0]
        Hinv /= (H[0,0]*H[1,1] - H[0,1]**2)

        qdot = Hinv.dot(B*u - C.dot(q[2:])- G)

        return np.hstack((q[2:], qdot))

    def update(self, action):
        u = np.clip(action, *self.action_range)
        self.state = odeint(self.state_dot, y0 = self.state, t = self.dt, args=(u,))[-1]
        self.state[1] = np.remainder(self.state[1], 2*np.pi)
        if self.state[0] > self.state_range[1][0] or self.state[0]< self.state_range[0][0]:
            self.state[2] = 0.0
        
        self.state = np.clip(self.state, self.state_range[0], self.state_range[1])

    def inGoal(self):
        return np.all(np.hstack((self.state[2:]>self.goal_range[0][2:],
                                     self.state[2:]<self.goal_range[1][2:],
                                     angle_range_check(self.goal_range[0][:2],
                                                 self.goal_range[1][:2],
                                                 self.state[:2]))))

    def copy(self):
        return copy.deepcopy(self)

    def isterminal(self):
        angle = np.abs(np.pi - self.state[1])
        angle = np.pi - angle if angle > np.pi else angle
        return angle< np.pi/12 and np.abs(self.state[3]) < 0.5

    @property
    def discrete_actions(self):
        return self.__discrete_actions

    @property
    def state_dim(self):
        return len(self.state_range[0])

    @property
    def action_dim(self):
        return len(self.action_range[0])
        
def angle_range_check( a, b, x):
    a = np.mod(a, 2*np.pi)
    b = np.mod(b, 2*np.pi)
    theta_bar = np.mod(b-a, 2*np.pi)
    return np.mod(x-a, 2*np.pi)<=theta_bar