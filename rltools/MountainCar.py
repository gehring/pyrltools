import numpy as np

class MountainCar(object):
    min_pos = -1.2
    max_pos = 0.6

    max_speed = 0.07

    goal_pos = 0.5

    pos_start = -0.5
    vel_start = 0.0

    state_range =[ np.array([min_pos, -max_speed]),
                   np.array([max_pos, max_speed])]
    action_range = [[-0.001], [0.001]]

    __discrete_actions = [ np.array([-0.001]),
                          np.array([0]),
                          np.array([0.001])]

    def __init__(self, random_start = False, max_episode = 1000, **argk):
        self.state = np.zeros(2)
        self.random_start = random_start
        self.max_episode = max_episode
        self.reset()

    def step(self, action):
        self.update(action)
        if self.inGoal() or self.step_count >= self.max_episode:
            next_state = None
        else:
            next_state = self.state.copy()

        self.step_count += 1

        return -1, next_state

    def reset(self):
        if self.random_start:
            self.state[:] = [np.random.uniform(self.state_range[0][0], self.state_range[1][0]),
                             np.random.uniform(self.state_range[0][1], self.state_range[1][1])]
        else:
            self.state[:] = [self.pos_start, self.vel_start]

        self.step_count = 0

        return 0, self.state.copy()

    def update(self, action):
        self.state[1] += (np.clip(action[0], *self.action_range)
                            + np.cos(3*self.state[0])*-0.0025)

        self.state[:] = np.clip(self.state, *self.state_range)
        self.state[0] += self.state[1]

        self.state[:] = np.clip(self.state, *self.state_range)
        if self.state[0] <= self.min_pos and self.state[1] < 0:
            self.state[1] = 0

    def inGoal(self):
        return self.state[0] >= self.goal_pos

    def copy(self):
        mountaincopy = MountainCar(random_start = self.random_start,
                                   max_episode = self.max_episode)
        mountaincopy.state[:] = self.state
        mountaincopy.step_count = self.step_count
        return mountaincopy

    @property
    def discrete_actions(self):
        return self.__discrete_actions

    @property
    def state_dim(self):
        return len(self.state_range[0])

    @property
    def action_dim(self):
        return len(self.action_range[0])

class MountainCar_Factory(object):
    def __init__(self, **argk):
        self.param = argk

    def __call__(self, **argk):
        params = dict(self.param)
        params.update([x for x in argk.items()])
        return MountainCar(**params)

class PumpingPolicy(object):
    def __init__(self):
        pass
    def __class__(self, state):
        return np.array([0.001]) if state[1] >= 0 else np.array([-0.001])

class InifiniteMountainCar(object):
    min_pos = -1.2
    max_pos = 0.6

    max_speed = 0.07

    goal_pos = 0.5

    pos_start = -0.5
    vel_start = 0.0

    state_range =[ np.array([min_pos, -max_speed]),
                   np.array([max_pos, max_speed])]
    action_range = [[-0.001], [0.001]]

    __discrete_actions = [ np.array([-0.001]),
                          np.array([0]),
                          np.array([0.001])]

    def __init__(self, random_start = True, **argk):
        self.state = np.zeros(2)
        self.random_start = random_start
        self.reset()

    def step(self, action):
        self.update(action)
        if self.inGoal():
            return self.reset()
        else:
            return -1, self.state.copy()


    def reset(self):
        if self.random_start:
            self.state[:] = [np.random.uniform(self.state_range[0][0], self.state_range[1][0]),
                             np.random.uniform(self.state_range[0][1], self.state_range[1][1])]
        else:
            self.state[:] = [self.pos_start, self.vel_start]


        return 1, self.state.copy()

    def update(self, action):
        self.state[1] += (np.clip(action[0], *self.action_range)
                            + np.cos(3*self.state[0])*-0.0025)

        self.state[:] = np.clip(self.state, *self.state_range)
        self.state[0] += self.state[1]

        self.state[:] = np.clip(self.state, *self.state_range)
        if self.state[0] <= self.min_pos and self.state[1] < 0:
            self.state[1] = 0

    def inGoal(self):
        return self.state[0] >= self.goal_pos

    def copy(self):
        mountaincopy = InifiniteMountainCar(random_start = self.random_start)
        mountaincopy.state[:] = self.state
        return mountaincopy

    @property
    def discrete_actions(self):
        return self.__discrete_actions

    @property
    def state_dim(self):
        return len(self.state_range[0])

    @property
    def action_dim(self):
        return len(self.action_range[0])
