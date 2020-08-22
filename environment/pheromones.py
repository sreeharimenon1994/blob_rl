import numpy as np
from .perception import Perception

np.seterr(divide='ignore', invalid='ignore')

class Pheromones:
    """docstring for Pheromones"""
    def __init__(self, color, w, h, padding):
        super(Pheromones, self).__init__()
        self.color = color
        self.w = w
        self.h = h
        self.max = 1
        self.padding = padding
        self.pos_surround = 2 * self.padding
        self.xy = np.zeros([self.w + self.pos_surround, self.h + self.pos_surround])
        self.perception = Perception()

    @property
    def xy_pos_count(self, pos):
        """indexing 'padding' number of values around the pos"""
        return self.xy[pos[0]:pos[0] + self.pos_surround,\
                       pos[1]:pos[1] + self.pos_surround]

    @property
    def xy_actual(self):
        return self.xy[self.padding: -self.padding, self.padding: -self.padding]
    
    
    def perish_update(self):
        self.xy = self.xy - 0.00002
        self.xy[self.xy < 0] = 0

    def observation(self, pos, rotation):
        # arr = self.xy[pos[0][0]:pos[0][0] + self.pos_surround + 1,\
        #            pos[0][1]:pos[0][1] + self.pos_surround + 1].reshape(1,-1)
        # for i in pos[1:]:
        #     tmp = self.xy[i[0]:i[0] + self.pos_surround + 1,\
        #                i[1]:i[1] + self.pos_surround + 1].reshape(1,-1)
        #     arr = np.vstack([arr, tmp])
        # arr = arr.reshape(1, -1)
        arr = self.perception.perceive_data(self.xy, pos + self.padding, rotation)
        return arr

    def add(self, pos):
        self.xy[pos[0] + self.padding, pos[1] + self.padding] += 0.005
        self.xy[self.xy > self.max] = self.max

    def reset(self):
        self.xy = np.zeros([self.w + self.pos_surround, self.h + self.pos_surround])