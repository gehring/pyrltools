import numpy as np
from scipy.integrate import odeint
from control import lqr
import copy

from sklearn import linear_model
import scipy.optimize

class Acrobot(object):

    umax = 20
    umin = -20

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

    max_speed = 24
    state_range = [np.array([0, 0, -max_speed, -max_speed]),
                   np.array([np.pi*2, np.pi*2, max_speed, max_speed])]


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
                 start_sampler = None,
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

    def get_dG(self, q):
        g = self.g
        m1 = self.m1
        m2 = self.m2
        lc1 = self.lc1
        l1 = self.l1
        lc2 = self.lc2
        dG = np.array([(-g*(m1*lc1 + m2*l1 + m2*lc2), -m2*lc2*g),
                       (-m2*g*lc2, -m2*g*lc2)])
        return dG

    def get_pumping_policy(self):
        return Acrobot_energyshaping(self)

    def get_swingup_policy(self, k1=10,k2=10, k3=1, lqr_thres=10000):
        return Acrobot_LQR_enerygyshaping(self,k1, k2, k3, lqr_thres)

    def inGoal(self):
        return np.all(np.hstack((self.state[2:]>self.goal_range[0][2:],
                                     self.state[2:]<self.goal_range[1][2:],
                                     angle_range_check(self.goal_range[0][:2],
                                                 self.goal_range[1][:2],
                                                 self.state[:2]))))

    def copy(self):
        return copy.deepcopy(self)

    def isterminal(self):
        return self.step_count>self.max_episode

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

def compute_acrobot_from_data(q,
                              qdot,
                              qdotdot,
                              tao,
                              method = 'dynamics',
                              random_start = False,
                              max_episode = 1000,
                              start_sampler = None):
    if method == 'dynamics':
        A, tao = get_matrix_dynamics(q,
                                      qdot,
                                      qdotdot,
                                      tao)
    if method == 'energy':
        A, tao = get_matrix_energy(q,
                                  qdot,
                                  qdotdot,
                                  tao)
    if method == 'power':
        A, tao = get_matrix_power(q,
                                  qdot,
                                  qdotdot,
                                  tao)

#     a = np.linalg.lstsq(A, tao)[0]
    a = linear_model.ridge_regression(A, tao, 0.00001)
#     a = scipy.optimize.nnls(A.T.dot(A) + 0.01 * np.eye(A.shape[1]), A.T.dot(tao))[0]
    
    if method != 'dynamics':
        a = np.array([a[0] + a[1],
                      a[2],
                      a[1],
                      a[3],
                      a[4],
                      a[5],
                      a[6]])
    
    return Acobot_from_data(a,
                            random_start,
                            max_episode,
                            start_sampler)

def get_matrix_dynamics(q,
                      qdot,
                      qdotdot,
                      tao):
    c = np.cos(q)
    c12 = np.sin(np.sum(q, axis=1))

    c1 = np.sin(q[:,0])
    c2 = c[:,1]
    s2 = np.sin(q[:,1])

    qd1 = qdot[:,0]
    qd2 = qdot[:,1]
    qdd1 = qdotdot[:,0]
    qdd2 = qdotdot[:,1]

    n= q.shape[0]
    u = np.zeros((n*2, 7))
    u[:n,0] = qdd1
    u[:n,1] = 2*c2*qdd1 + c2*qdd2 - s2*qd2**2 - 2*s2*qd2*qd1
    u[:n,2] = qdd2
    u[:n,3] = c1
    u[:n,4] = c12
    u[:n,5] = qd1

    u[n:,1] = c2*qdd1 + s2*qd1**2
    u[n:,2] = qdd2 + qdd1
    u[n:,4] = c12
    u[n:,6] = qd2

    tao = np.hstack((np.zeros(n), tao))
    return u, tao

def get_matrix_power(q,
                      qdot,
                      qdotdot,
                      tao):
    s = np.sin(q)
    s1 = s[:,0]
    s2 = s[:,1]
    s12 = np.sin(q[:,0] + q[:,1])
    c2 = np.cos(q[:,1])
    
    
    qd1 = qdot[:,0]
    qd2 = qdot[:,1]
    qdd1 = qdotdot[:,0]
    qdd2 = qdotdot[:,1]
    
    A = np.zeros((q.shape[0], 7))
    A[:,0] = qd1*qdd1
    A[:,1] = A[:,0] + qd2*qdd2 + qd1*qdd2 + qdd1*qd2
    A[:,2] = c2*(2*A[:,0] + qd1*qdd2 + qdd1*qd2) - s2*qd2*(qd1**2 + qd1*qd2)
    A[:,3] = s1*qd1
    A[:,4] = s12 * (qd1 + qd2)
    A[:,5] = qd1**2
    A[:,6] = qd2**2
    
    tao = tao*qd2
    
    return A, tao

def get_matrix_energy(q,
                      qdot,
                      qdotdot,
                      tao):
    q1a = q[:-1,0]
    q1b = q[1:,0]
    q2a = q[:-1,1]
    q2b = q[1:,1]
    
    qd1a = qdot[:-1,0]
    qd1b = qdot[:-1,0]
    qd2a = qdot[:-1,1]
    qd2b = qdot[:-1,1]
    
    A = np.zeros((q.shape[0]-1, 7))
    A[:,0] = 0.5*(qd1b**2 - qd1a**2)
    A[:,1] = 0.5*(qd1b**2 - qd1a**2 + qd2b**2 - qd2a**2) + qd1b*qd2b - qd1a*qd2a
    A[:,2] = np.cos(q2b) * (qd1b**2 + qd1b*qd2b)  - np.cos(q2a) * (qd1a**2 + qd1a*qd2a)
    A[:,3] = np.cos(q1b) - np.cos(q1a)
    A[:,4] = np.cos(q1b + q2b) - np.cos(q1a + q2a)
    A[:,5] = 0.5 * (qd1b**2 + qd1a**2)
    A[:,6] = 0.5 * (qd1b**2 + qd1a**2)
#     A[:,5] = qd1b**2 - qd1a*qd1b + qd1a**2
#     A[:,6] = qd2b**2 - qd2a*qd2b + qd2a**2
    
    
    tao = tao[:-1] * (q2b - q2a)
    
    return A, tao
    

class Acobot_from_data(Acrobot):
    def __init__(self,
                 a,
                 random_start = False,
                 max_episode = 1000,
                 start_sampler = None,
                 **argk):
        super(Acobot_from_data, self).__init__(random_start =random_start,
                                               max_episode = max_episode,
                                               start_sampler = start_sampler,
                                               **argk)
        self.a = a

    def get_manipulator(self, q):
        c1 = np.sin(q[0])
        c12 = np.sin(q[0]+q[1])

        c2 = np.cos(q[1])
        s2 = np.sin(q[1])

        a = self.a

        aa = a[0] + a[1]*2*c2
        b = a[1]*c2 + a[2]
        d = a[2]
        H= np.array(((aa, b),
                     (b, d)))

        C= np.array(((a[5]-2*a[1]*s2*q[3], -a[1]*s2*q[3]),
                     (a[1]*s2*q[2], a[6]))
                    )

        G = np.array((a[3]*c1 + a[4]*c12, a[4]*c12))

        B = np.array((0, 1))
        return (H, C, G, B)

    def get_E(self, q):
        H, C, G, B = self.get_manipulator(q)
        c = np.cos(q[0])
        U = -self.a[3]*c - self.a[4]*np.cos(np.sum(q[:2]))
        return 0.5* q[2:].dot(H.dot(q[2:])) + U

    def get_dG(self, q):
        dG = np.array([(-self.a[3] - self.a[4], -self.a[4]),
                       (-self.a[4], -self.a[4])])
        return dG


class Acrobot_energyshaping(object):
    desired_pos = np.array([np.pi,0,0,0])
    def __init__(self, acrobot, k1=1.14, k2=3.2, k3=1.1, desired_pos = None):
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
#         detinv = 1/(H[1,1]*H[0,0] - H[0,1]**2)
#         a3 = (H[0,0]*detinv)
#         a2 = -(H[0,1]*detinv)
#         e_tilde = (self.acrobot.get_E(q) - self.Ed)
#         q2wrapped = np.remainder(q[1]+np.pi, np.pi*2) - np.pi
#
#         y=-self.k1*q2wrapped - self.k2*q[3]
#         u = y/a3 + a2*C[0]/a3 + C[1] - self.k3*e_tilde*q[3]
        Ed = self.Ed
        E = self.acrobot.get_E(q)

#         Ed = 2*9.81*1.25
#         E = 9.81*(np.cos(q[0]) + 2*(np.cos(q[0] + q[1])+np.cos(q[0]))) + 0.5*q[2:].dot(H.dot(q[2:]))
        M22bar = H[1,1] - H[1,0]/H[0,0]*H[0,1]
        h2bar = C[1] - H[1,0]/H[0,0]*C[0]
        k1 = 10
        k2 = 10
        k3 = 1
        ubar = (Ed -E) * q[3]
        u = M22bar*(-k1*q[1] - k2*q[3]) + h2bar + k3*ubar
        return u

class Acrobot_PD(object):
    def __init__(self, k1=10, k2=20, alpha=0.30):
        self.k1 = k1
        self.k2 =k2
        self.alpha = alpha

    def __call__(self, q):
        q2d = 2*self.alpha/np.pi * np.arctan(q[2])
        u = self.k1*(q2d - q[1]) - self.k2*q[1]
        return u
class Acrobot_LQR(object):
    def __init__(self,
                 acrobot,
                 Q= None,
                 R= None,
                 desired_pos = None):
        if desired_pos is not None:
            desired_pos= np.array([np.pi,0,0,0])
        self.desired_pos = desired_pos
        if Q is not None:
            Q = np.diag([1,1,10,10])
        if R is not None:
            R = np.eye(1)
        self.acrobot = acrobot
        A, B = self.get_linear(desired_pos, np.zeros(1))
        self.lqr = lqr(A,B[:,None],Q,R)


    def get_linear(self, q, u):
        H, C, G, B = self.acrobot.get_manipulator(q)
        Hinv= -H
        Hinv[0,0]= H[1,1]
        Hinv[1,1]= H[0,0]
        Hinv /= (H[0,0]*H[1,1] - H[0,1]**2)

        dG  = self.acrobot.get_dG(q)

        A = np.zeros((4,4))
        A[2:,:2] = -Hinv.dot(dG)
        A[:2,2:] = np.eye(2)
        A[2:,2:] = -Hinv.dot(C)

        Blin = np.zeros(4)
        Blin[2:] = Hinv.dot(B)
        return A, Blin

    def naive_test(self, q, lqr_thres):
        qbar = self.get_qbar(q)
        return qbar.dot(self.lqr[1].dot(qbar))< lqr_thres

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
                 k1=10,
                 k2=10,
                 k3=1,
                 lqr_thres=10000,
                 Q = None,
                 R = None,
                 desired_pos = None):
        if desired_pos == None:
            desired_pos= np.array([np.pi,0,0,0])
        self.desired_pos = desired_pos

        if Q is None:
            Q = np.diag([1,1,15,15])*50
        if R is None:
            R = np.eye(1)
        self.energyshaping = Acrobot_energyshaping(acrobot,k1,k2,k3)
        self.lqr = Acrobot_LQR(acrobot, Q, R, desired_pos)
        self.lqr_thres = lqr_thres

    def __call__(self, q):
        if self.lqr.naive_test(q, self.lqr_thres):
            return self.lqr(q)
        else:
            return self.energyshaping(q)

    def set_param(self, param):
        self.energyshaping.k1 = param[0]
        self.energyshaping.k2 = param[1]
        self.energyshaping.k3 = param[2]
        self.lqr_thres = param[3]

def get_trajectories(acrobot,
                     number = 1,
                     length = 100,
                     controller = None):
    if controller is None:
        controller = acrobot.get_swingup_policy()


    traj = []
    for i in xrange(number):
        acrobot.reset()

        u = controller(acrobot.state)
        steps = [(acrobot.state, u)]

        for j in xrange(length):

            acrobot.step(u)
            u = np.clip(controller(acrobot.state), *acrobot.action_range)

            steps.append((acrobot.state, u))

        traj.append(steps)

    states = [np.vstack(zip(*t)[0]) for t in traj]
    torques = [np.hstack(zip(*t)[1]) for t in traj]
    return (states, torques) if number > 1 else (states[0], torques[0])

def get_qs_from_traj(states, torques, dt):
    if isinstance(states, list):
        qdd = np.vstack([ (s[2:,2:] - s[:-2,2:])/(2*dt) for s in states])
        qd = np.vstack([ s[1:-1,2:] for s in states])
        q = np.vstack([ s[1:-1,:2] for s in states])
        y = np.hstack([ t[1:-1] for t in torques])
    else:
        qdd = (states[2:,2:] - states[:-2,2:])/(2*dt)
        qd = states[1:-1,2:]
        q = states[1:-1,:2]
        y = torques[1:-1]

    return q, qd, qdd, y


