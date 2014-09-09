import numpy as np
from itertools import product
import copy

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

    def copy(self):
        newmaze = POMaze([copy.copy(t) for t in self.transitions], copy.copy(self.goal))
        return newmaze

    @property
    def discrete_actions(self):
        return np.arange(4)

def createMazeFromLines(walls, goal, size, wrap = 'clip'):
    actions = [(0,1), (1,0), (-1,0), (0,-1)]
    transitions = [{ (x,y) : (x+a[0], y+a[1]) for x,y in product(range(size[0]),
                                                                 range(size[1]))}
                   for a in actions]

    for w in walls:
        for t in transitions:
            if w[1] in t and w[0] == t[w[1]]:
                del t[w[1]]
            if w[0] in t and w[1] == t[w[0]]:
                del t[w[0]]

    if wrap == 'clip':
        for x in range(size[0]):
            del transitions[3][(x,0)]
            del transitions[0][(x,size[1]-1)]
        for y in range(size[1]):
            del transitions[2][(0,y)]
            del transitions[1][(size[0]-1, y)]
    elif wrap == 'wrap':
        for x in range(size[0]):
            transitions[3][(x,0)] = (x,size[1]-1)
            transitions[0][(x,size[1]-1)] = (x,0)
        for y in range(size[1]):
            transitions[2][(0,y)] = (size[0]-1, y)
            transitions[1][(size[0]-1,y)] = (0, y)

    return POMaze(transitions, goal)

