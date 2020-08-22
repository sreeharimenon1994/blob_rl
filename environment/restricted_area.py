import numpy as np
restriction = 1
from .perception import Perception


class Restricted_Area:
    """docstring for Restricted_Area"""
    def __init__(self, w, h):
        super(Restricted_Area, self).__init__()
        self.w = w
        self.h = h
        self.xy = np.zeros([self.w-2, self.h-2])
        self.xy = np.pad(self.xy, restriction, 'constant', constant_values=1)

    @property
    def possibility(self, pos):
        """indexing 'padding' number of values around the pos."""
        return self.xy[pos[0]:pos[0] + pos_surround,\
                       pos[1]:pos[1] + pos_surround]

    def observation(self, pos, rotation):
        arr = self.xy[pos[0][0]:pos[0][0] + self.pos_surround + 1,\
                   pos[0][1]:pos[0][1] + self.pos_surround + 1].reshape(1,-1)
        for i in pos[1:]:
            tmp = self.xy[i[0]:i[0] + self.pos_surround + 1,\
                       i[1]:i[1] + self.pos_surround + 1].reshape(1,-1)
            arr = np.vstack([arr, tmp])
        # arr = arr.reshape(1, -1)
        # arr = self.perception.perceive_data(self.xy, pos, rotation)
        return arr

        