import numpy as np

class Perception:
    """docstring for Perception"""
    def __init__(self, min_r=0.125):
        super(Perception, self).__init__()
        self.min_r = min_r
        self.indices = {
            0: {
                "x": np.array([0, 1, 2, 2, 2, 2, 2, 1, 0, 0, 1, 1, 1, 0]),
                "y": np.array([2, 2, 2, 1, 0, -1, -2, -2, -2, 1, 1, 0, -1, -1])},
            1: {
                "x": np.array([-2, -1, 0, 1, 2, 2, 2, 2, 2, -1, 0, 1, 1, 1]),
                "y": np.array([2, 2, 2, 2, 2, 1, 0, -1, -2, 1, 1, 1, 0, -1])},
            2: {
                "x": np.array([-2, -2, -2, -1, 0, 1, 2, 2, 2, -1, -1, 0, 1, 1]),
                "y": np.array([0, 1, 2, 2, 2, 2, 2, 1, 0, 0, 1, 1, 1, 0])},
            3: {
                "x": np.array([-2, -2, -2, -2, -2, -1, 0, 1, 2, -1, -1, -1, 0, 1]),
                "y": np.array([-2, -1, 0, 1, 2, 2, 2, 2, 2, -1, 0, 1, 1, 1])},
            4: {
                "x": np.array([0, -1, -2, -2, -2, -2, -2, -1, 0, 0, -1, -1, -1, 0]),
                "y": np.array([-2, -2, -2, -1, 0, 1, 2, 2, 2, -1, -1, 0, 1, 1])},
            5: {
                "x": np.array([2, 1, 0, -1, -2, -2, -2, -2, -2, 1, 0, -1, -1, -1]),
                "y": np.array([-2, -2, -2, -2, -2, -1, 0, 1, 2, -1, -1, -1, 0, 1])},
            6: {
                "x": np.array([2, 2, 2, 1, 0, -1, -2, -2, -2, 1, 1, 0, -1, -1]),
                "y": np.array([0, -1, -2, -2, -2, -2, -2, -1, 0, 0, -1, -1, -1, 0])},
            7: {
                "x": np.array([2, 2, 2, 2, 2, 1, 0, -1, -2, 1, 1, 1, 0, -1]),
                "y": np.array([2, 1, 0, -1, -2, -2, -2, -2, -2, 1, 0, -1, -1, -1])},
        }
    def perceive_data(self, xy, pos, rotation):
        res = None
        for pos_t, rot in zip(pos, rotation):
            # if rot < 0:
            #     rot = 1 - (rot % 1)
            # else:
            #     rot = rot % 1
            r = abs(int((rot%1) / self.min_r))
            i = self.indices[r]
            x = i['x'] + pos_t[0]
            y = i['y'] + pos_t[1]
            if res is None:
                res = xy[x, y]
            else:
                res = np.vstack([res, xy[x, y]])
        return res

