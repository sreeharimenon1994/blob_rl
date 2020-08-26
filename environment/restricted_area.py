import numpy as np
restriction = 3
from .perception import Perception


class Restricted_Area:
    """docstring for Restricted_Area"""
    def __init__(self, w, h):
        super(Restricted_Area, self).__init__()
        self.w = w
        self.h = h
        self.xy = np.zeros([self.w-1, self.h-1])
        self.xy = np.pad(self.xy, restriction, 'constant', constant_values=1)
        self.perception = Perception()

    @property
    def possibility(self, pos):
        """indexing 'padding' number of values around the pos."""
        return self.xy[pos[0]:pos[0] + pos_surround,\
                       pos[1]:pos[1] + pos_surround]

    def observation(self, pos, rotation):
        # arr = self.xy[pos[0][0]-1:pos[0][0] + 2,\
        #            pos[0][1]-1:pos[0][1]+ 2].reshape(1,-1)
        # for i in pos[1:]:
        #     tmp = self.xy[i[0]-1:i[0] + 2,\
        #                i[1]-1:i[1] + 2].reshape(1,-1)
        #     arr = np.vstack([arr, tmp])

        arr = self.perception.perceive_data(self.xy, pos, rotation)
        # print(arr)
        return arr

        
