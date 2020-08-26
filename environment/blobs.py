import numpy as np
from .restricted_area import Restricted_Area
from .perception import Perception

class Blobs:
    """docstring for Blobs"""
    def __init__(self, n_blobs, w, h, padding, n_steps):
        super(Blobs, self).__init__()
        self.n_blobs = n_blobs
        self.w = w
        self.h = h
        self.padding = padding
        self.n_steps = n_steps
        self.xyfa = None
        """ 0. number of steps, 1. prev distance with food, add more...
        """
        self.extra = np.ones([self.n_blobs, 2], dtype=np.float) 
        # self.xy = np.zeros([self.w, self.h], dtype=np.int)
        # self.xy = np.pad(self.xy, int(padding/2), 'constant', constant_values=-1)
        self.rotation = np.zeros(self.n_blobs)
        self.jump = np.ones((self.n_blobs, 1))
        self.pheromones = []
        self.restricted_area = Restricted_Area(w=self.w, h=self.h)
        self.reset()
        self.perception = Perception()

    def add_pheromones(self, pheromones):
        self.pheromones.append(pheromones)

    def observation(self):
        pos = self.xyfa[:, 0]
        # arr = self.pheromones[0].observation(pos)
        arr = self.pheromones[0].observation(pos, self.rotation)
        for i in self.pheromones[1:]:
            tmp = i.observation(pos, self.rotation)
            arr = np.append(arr, tmp, axis=1)
        arr = np.append(arr, self.rotation.reshape(-1, 1), axis=1)
        # arr = np.append(arr, self.jump.reshape(-1, 1), axis=1)
        # arr = np.append(arr, self.xyfa[:, 1, 0].reshape(-1, 1), axis=1)
        # arr = np.append(arr, self.xyfa[:, 1, 1].reshape(-1, 1)/3, axis=1)
        tmp = self.restricted_area.observation(pos)
        arr = np.append(arr, tmp, axis=1)
        # arr = np.append(arr, self.extra[:, 0].reshape(-1, 1), axis=1)

        return arr

    def update_pos(self, rotation, jump):
        self.rotation += rotation
        self.rotation = self.rotation % 1
        # print(self.rotation)
        # self.jump = jump
        jump = 1.0
        # self.jump = 1.0
        x, y = (jump * np.cos(self.rotation * 2 * np.pi),\
                jump * np.sin(self.rotation * 2 * np.pi))

        x1 = (self.xyfa[:, 0, 0] + x).round()
        y1 = (self.xyfa[:, 0, 1] + y).round()
        # print(x1.shape, y1.shape, '------->', x.shape, y.shape, '---------->', rotation, jump)
        x1[x1 < 0] = 0
        y1[y1 < 0] = 0
        x1[x1 > self.h - 1] = self.h - 1
        y1[y1 > self.w - 1] = self.w - 1

        restricted = self.restricted_area.xy[x1.astype(np.int), y1.astype(np.int)]
        not_restricted = np.invert(restricted.astype(np.bool))

        self.xyfa[:, 0 , 0] = x1 * not_restricted + self.xyfa[:, 0 , 0] * restricted
        self.xyfa[:, 0 , 1] = y1 * not_restricted + self.xyfa[:, 0 , 1] * restricted
        self.xyfa[self.xyfa[:, 0, 0] < 0, 0, 0] = 0
        self.xyfa[self.xyfa[:, 0, 1] < 0, 0, 1] = 0
        
        self.xyfa[self.xyfa[:, 0, 0] > self.h - 1] = self.h - 1
        self.xyfa[self.xyfa[:, 0, 1] > self.w - 1] = self.w - 1

        # self.extra[0] -= 0.0002
        # if self.extra[0] < .4:
        #     self.extra[0] = 0.4

    def update_pheromones(self, pheromones):
        for i, p in enumerate(self.pheromones):
            p.perish_update()
            tmp = pheromones == i
            ind = np.where(tmp)
            x = self.xyfa[:, 0, 0][ind]
            y = self.xyfa[:, 0, 1][ind]
            if len(x) > 0 and len(y) > 0:
                p.add([x, y])

    def reset(self):
        self.xyfa = np.ones([self.n_blobs, 3, 2], dtype=np.int)
        self.xyfa[:,1] = 0
        self.xyfa[:,0] = 3
        self.xy = np.zeros([self.w, self.h], dtype=np.int)
        self.xy = np.pad(self.xy, int(self.padding/2), 'constant', constant_values=-1)
        for x in self.pheromones:
            x.reset()

    def visualise_blobs(self):
        xy = self.xyfa[:, 0]
        return xy

    def visualise_pheromones(self):
        ph = []
        for x in self.pheromones:
            tmp = np.where(x.xy_actual > 0)
            if len(tmp[0]) > 0:
                ph.append((np.dstack((tmp[0], tmp[1])))[0])
        return ph






