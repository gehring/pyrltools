import numpy as np

dx = np.array([ [1,0],
                [-1,0],
                [0,1],
                [0,-1]], dtype='int32')


class GridWorld(object):
    def __init__(self,
                 reward,
                 islegal,
                 terminal,
                 start_range,
                 transition = deterministic_transitions,
                 random_start = False,
                 max_episode = 1000,
                 **argk):
        self.state = np.zeros(2, dtype= 'int32')
        self.random_start = random_start
        self.max_episode = max_episode
        self.reward = reward
        self.islegal = islegal
        self.start_range = start_range
        self.transition = transition
        self.reset()

    def step(self, action):
        prev_state = self.state
        self.update(action)
        next_state = self.state.copy()
        self.step_count += 1

        return self.reward(prev_state, action), next_state

    def reset(self):
        if self.random_start:
            self.state[:] = [np.random.randint(self.start_range[0][0], self.start_range[1][0]),
                             np.random.randint(self.start_range[0][1], self.start_range[1][1])]
        else:
            self.state[:] = [0,0]

        self.step_count = 0

        return 0, self.state.copy()

    def update(self, action):
        next_state = self.transition(self.state, action)
        if self.islegal(next_state):
            self.state = next_state

    def isterminal(self):
        return self.terminal(self.state) or self.count >= self.max_episode


def deterministic_transitions(state, action):
    return state + dx[action]

class boundary_condition(object):
    def __init__(self, state_range):
        self.state_range = state_range

    def __call__(self, state):
        s = np.clip(state, self.state_range[0], self.state_range[1])
        return s == state

class obstacle_condition(object):
    def __init__(self, obstacle):
        self.obstacle = obstacle

    def __call__(self, state):
        return not state in self.obstacle

