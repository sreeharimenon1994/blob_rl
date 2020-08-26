import numpy as np
from .perception import Perception

class Food:
    """docstring for Food"""
    def __init__(self, w, h, hill, padding):
        super(Food, self).__init__()
        self.w = w
        self.h = h
        self.hill = hill
        self.padding = padding
        self.pos_surround = 2 * self.padding + 1
        self.xy = None
        self.reset()
        self.perception = Perception()

    @property
    def pos_count(self):
        return self.xy

    def observation(self, pos, rotation):
        arr = self.xy[pos[0][0]:pos[0][0] + self.pos_surround + 1,\
                   pos[0][1]:pos[0][1] + self.pos_surround + 1].reshape(1,-1)
        for i in pos[1:]:
            tmp = self.xy[i[0]:i[0] + self.pos_surround + 1,\
                       i[1]:i[1] + self.pos_surround + 1].reshape(1,-1)
            arr = np.vstack([arr, tmp])
        # arr = self.perception.perceive_data(self.xy, pos + self.padding, rotation)
        return arr

    def reset(self):
        self.xy = np.zeros([self.w, self.h])
        tmp = np.random.random(self.xy.shape)
        thesh = 0.9
        tmp[tmp > thesh] = 1
        tmp[tmp <= thesh] = 0
        self.xy += tmp
        self.xy = np.pad(self.xy, self.padding, 'constant', constant_values=0)


    def visualise_food(self):
        xy = self.xy[self.padding: -self.padding, self.padding: -self.padding]
        xy = np.where(xy > 0)
        return xy
