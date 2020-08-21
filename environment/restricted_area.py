import numpy as np
restriction = 1

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

        