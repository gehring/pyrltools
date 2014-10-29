import numpy as np
from scipy.integrate import odeint
from control import lqr

class Acrobot(object):

    umax = 10
    umin = -10

    dt = np.array([0, 0.1])

    start_state = np.array([0.0,0.0,0.0,0.0])
    __discrete_actions = [np.array([umin]),
                          np.array([0]),
                          np.array([umax])]

    action_range = [np.array([umin]),
                    np.array([umax]),]



    def __init__(self,
                 random_start = False,
                 max_episode = 1000,
                 m1 = 1,
                 m2 = 1,
                 l1 = 1,
                 l2 = 1,
                 g = 9.81,
                 b1 = 0.1,
                 b2 = 0.1,
                 **argk):
        self.l1 = l1
        self.l2 = l2
        self.lc1 = self.l1/2.0
        self.lc2 = self.l2/2.0
        self.m1 = m1
        self.m2 = m2
        self.g =g
        self.b1 =b1
        self.b2 = b2
        self.Ic1 = m1*l1**2/12.0
        self.Ic2 = m2*l2**2/12.0

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

        return 0, self.state.copy()

    def get_manipulator(self, q):
        m1 = self.m1
        m2 = self.m2
        l1 = self.l1
        lc1 = self.lc1
        lc2 = self.lc2
        I1 = self.Ic1 + m1*lc1**2.0
        I2 = self.Ic2 + m2*lc2**2.0

        g = self.g
        c = np.cos(q[:2])
        s = np.sin(q[:2])
        s12 = np.sin(q[0]+q[1])
        m2l1lc2 = m2*l1*lc2

        a = I1 + I2 + m2*l1**2 + 2*m2l1lc2*c[1]
        b =  I2 + m2l1lc2*c[1]
        d = I2
        H= np.array(((a, b),
                     (b, d))
                    )
        C= np.array(((self.b1 -2*m2l1lc2*s[1]*q[3], -m2l1lc2*q[3]*s[1]),
                     (m2l1lc2*q[2]*s[1], self.b2))
                    )
        G = g* np.array(((m1*lc1 + m2*l1)*s[0] + m2*lc2*s12, m2*lc2*s12))

        B = np.array((0, 1))
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
        self.state[:2] = np.remainder(self.state[:2], 2*np.pi)

    def get_E(self, q):
        H, C, G, B = self.get_manipulator(q)
        c = np.cos(q[0])
        U = -self.m1*self.g*self.lc1*c - self.m2*self.g*(self.l1*c +
                                                         self.lc2*np.cos(q[0] + q[1]))
        return 0.5* q[2:].dot(H.dot(q[2:])) + U

    def get_pumping_policy(self):
        return Acrobot_energyshaping(self)

    def get_swingup_policy(self):
        return Acrobot_LQR_enerygyshaping(self)

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

class Acrobot_energyshaping(object):
    desired_pos = np.array([np.pi,0,0,0])
    def __init__(self, acrobot, k1=2.0, k2=1.0, k3=0.1, desired_pos = None):
        if desired_pos == None:
            desired_pos= np.array([np.pi,0,0,0])
        self.desired_pos = desired_pos
        self.acrobot = acrobot
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.Ed = acrobot.get_E(self.desired_pos)

    def __call__(self, q):
        H, C, G, B = self.acrobot.get_manipulator(q)

        C= C.dot(q[2:]) + G
        detinv = 1/(H[1,1]*H[0,0] - H[0,1]**2)
        a3 = (H[0,0]*detinv)
        a2 = -(H[0,1]*detinv)
        e_tilde = (self.acrobot.get_E(q) - self.Ed)
        q2wrapped = np.remainder(q[1]+np.pi, np.pi*2) - np.pi

        y=-self.k1*q2wrapped - self.k2*q[3]
        u = y/a3 + a2*C[0]/a3 + C[1] - self.k3*e_tilde*q[3]

        return u

class Acrobot_LQR(object):
    def __init__(self,
                 acrobot,
                 Q= None,
                 R= None,
                 desired_pos = None):
        if desired_pos == None:
            desired_pos= np.array([np.pi,0,0,0])
        self.desired_pos = desired_pos
        if Q == None:
            Q = np.diag([10,10,1,1])
        if R == None:
            R = np.eye(1)
        self.acrobot = acrobot
        A, B = self.get_linear(desired_pos, np.zeros(1))
        print A
        print B
        self.lqr = lqr(A,B[:,None],Q,R)


    def get_dG(self, q):
        g = self.acrobot.g
        m1 = self.acrobot.m1
        m2 = self.acrobot.m2
        lc1 = self.acrobot.lc1
        l1 = self.acrobot.l1
        lc2 = self.acrobot.lc2
        dG = np.array([(-g*(m1*lc1 + m2*l1 + m2*lc2), -m2*lc2*g),
                       (-m2*g*lc2, -m2*g*lc2)])
        return dG

    def get_linear(self, q, u):
        H, C, G, B = self.acrobot.get_manipulator(q)
        Hinv= -H
        Hinv[0,0]= H[1,1]
        Hinv[1,1]= H[0,0]
        Hinv /= (H[0,0]*H[1,1] - H[0,1]**2)

        dG  = self.get_dG(q)

        A = np.zeros((4,4))
        A[2:,:2] = -Hinv.dot(dG)
        A[:2,2:] = np.eye(2)
        A[2:,2:] = -Hinv.dot(C)

        Blin = np.zeros(4)
        Blin[2:] = Hinv.dot(B)
        return A, Blin

    def naive_test(self, q):
        qbar = self.get_qbar(q)
        return qbar.dot(self.lqr[1].dot(qbar))< 1000

    def __call__(self, q):
        return -self.lqr[0].dot(self.get_qbar(q))

    def get_qbar(self, q):
        q_bar = q.copy()
        q_bar[:2] = np.remainder(q[:2]- self.desired_pos[:2] + np.pi, 2*np.pi) - np.pi
        q_bar[2:] -= self.desired_pos[2:]
        return q_bar

class Acrobot_LQR_enerygyshaping(object):
    def __init__(self,
                 acrobot,
                 k1=2.0,
                 k2=1.0,
                 k3=0.1,
                 Q = None,
                 R = None,
                 desired_pos = None):
        if desired_pos == None:
            desired_pos= np.array([np.pi,0,0,0])
        self.desired_pos = desired_pos

        if Q == None:
            Q = np.diag([1,1,1,1])*50
        if R == None:
            R = np.eye(1)
        self.energyshaping = Acrobot_energyshaping(acrobot,k1,k2,k3)
        self.lqr = Acrobot_LQR(acrobot, Q, R, desired_pos)

    def __call__(self, q):
        if self.lqr.naive_test(q):
            return self.lqr(q)
        else:
            return self.energyshaping(q)



