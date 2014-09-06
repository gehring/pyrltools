import numpy as np

class POMaze(object):

    bad_move_rew = -10
    step_rew = -1
    goal_rew = 0

    dims = np.ones(4)*2

    def __init__(self, transitions, goal):
        self.state= 0
        self.transitions = transitions
        self.goal = goal

    def reset(self):
        self.state = 0

    def step(self, a):
        # if legal action
        if self.state in self.transitions[a]:
            self.state = self.transitions[a][self.state]

            # if reached goal
            if self.state in self.goal:
                # give goal reward
                return self.goal_rew, None
            else:
                # give step reward and new observation
                return self.step_rew, self.getObs()

        else:
            # give bad move reward and same observation
            return self.bad_move_rew, self.getObs()

    def getObs(self):
        can_transit = np.array([ self.state in t for t in self.transitions],
                               dtype = 'int32')
        return np.ravel_multi_index(can_transit, dims = self.dims)

