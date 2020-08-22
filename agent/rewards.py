import numpy as np
padding = 2

class Exploitation:
    """docstring for Exploitation"""
    def __init__(self, blobs, w, h):
        super(Exploitation, self).__init__()
        self.blobs = blobs
        self.xy = np.ones([w, h], dtype=np.int) * 1 # explore reward
        self.discount = 1.0
        self.discount_dec = 0.0025

    def calculate(self):
        s = self.xy[self.blobs.xyfa[:, 0, 0], self.blobs.xyfa[:, 0, 1]] * self.discount
        self.xy[self.blobs.xyfa[:, 0, 0], self.blobs.xyfa[:, 0, 1]] = 0
        self.discount = self.discount - self.discount_dec if self.discount >\
                        self.discount_dec else self.discount_dec
        return s


class Food_Reward:
    """docstring for Food_Reward"""
    def __init__(self, food, blobs, hill, padding):
        super(Food_Reward, self).__init__()
        self.food = food
        self.blobs = blobs
        self.hill = hill
        self.padding = padding
        self.single_food = 1

    def calculate(self):
        fx = self.blobs.xyfa[:, 0, 0] + self.padding
        fy = self.blobs.xyfa[:, 0, 1] + self.padding
        f = self.food.xy[fx, fy].astype(np.bool)
        
        # bf = self.blobs.xyfa[:, 1, 0] # blob food status
        # ba = self.blobs.xyfa[:, 1, 1] # blob action status

        # r1 = np.bitwise_and(np.bitwise_and(ba == 1, bf == 0), f)
        tmp = np.where(f)
        self.food.xy[fx[tmp], fy[tmp]] -= self.single_food
        self.blobs.xyfa[tmp, 1, 0] = 1

        r1 = f.astype(np.int) * 2 # picking food

        # r2 = np.bitwise_and(ba == 2, bf == 1)
        # tmp = np.where(r2)
        # self.food.xy[fx[tmp], fy[tmp]] += self.single_food
        # self.blobs.xyfa[tmp, 1, 0] = 0
        # r2 = r2.astype(np.int) * -2 # dropping food

        h = self.hill.xy[fx, fy].astype(np.bool)
        g = self.blobs.xyfa[:, 1, 0].astype(np.bool)

        gh = np.bitwise_and(g, h)
        if np.any(gh):
            tmp = np.where(gh)
            self.blobs.xyfa[tmp, 1, 0] = 0
            # self.food.xy[fx[tmp], fy[tmp]] -= self.single_food
            r3 = gh.astype(np.float) * 10 # hill drop reward
            # r3 *= self.blobs.extra[:, 0] * gh
            self.blobs.extra[gh, 0] = 1 # reset steps after dropping
            self.blobs.extra[gh, 1] = 0 # reseting prev distance for future reward
        else:
            r3 = np.zeros(self.blobs.n_blobs)

        dist = np.sqrt(self.blobs.xyfa[:, 0, 0]**2 + self.blobs.xyfa[:, 0, 1]**2)
        tmp = self.blobs.extra[:, 1] > dist
        r4 = tmp * 3.0 * self.blobs.xyfa[:, 1, 0]
        # print(r4, dist, tmp, self.blobs.xyfa[:, 1, 0])
        self.blobs.extra[:, 1] = dist
        return r1 + r3 + r4


class Reward:
    """docstring for Reward"""
    def __init__(self, blobs, food, w, h, hill, padding):
        super(Reward, self).__init__()
        self.w = w
        self.h = h
        self.blobs = blobs
        self.food = food
        self.hill = hill
        self.padding = padding
        self.exploitation_reward = Exploitation(self.blobs, w=self.w, h=self.h)
        self.food_reward = Food_Reward(food=self.food, blobs=self.blobs, hill=self.hill, padding=self.padding)

    def calculate(self):
        return self.exploitation_reward.calculate() + self.food_reward.calculate()

