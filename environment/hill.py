import numpy as np
# padding = 2
# pos_surround = 2 * padding


class Hill:
    """docstring for Hill"""
    def __init__(self, w, h, padding):
        super(Hill, self).__init__()
        self.w = w
        self.h = h
        self.padding = padding
        self.pos_surround = 2 * self.padding
        self.xy = np.zeros([self.w + self.pos_surround, self.h + self.pos_surround])
        self.xy[4:14, 4:14] = 1
    
    def observation(self, pos):
        arr = self.xy[pos[0][0]:pos[0][0] + self.pos_surround + 1,\
                   pos[0][1]:pos[0][1] + self.pos_surround + 1].reshape(1,-1)
        for i in pos[1:]:
            tmp = self.xy[i[0]:i[0] + self.pos_surround + 1,\
                       i[1]:i[1] + self.pos_surround + 1].reshape(1,-1)
            arr = np.vstack([arr, tmp])
        # arr = arr.reshape(1, -1)
        return arr
        