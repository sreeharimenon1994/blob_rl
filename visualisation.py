import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import json
import torch
from environment.base import Base
import time

delay = 0.05


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, base):
        self.base = base
        self.w = self.base.w
        self.h = self.base.h
        self.numpoints = self.base.n_blobs
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(figsize=(11, 9))
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=5, 
                                          init_func=self.setup_plot, blit=True)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, s, c = next(self.stream).T
        self.scat = self.ax.scatter(x, y, c=c, s=s, vmin=0, vmax=1)
        self.ax.axis([0, self.w, 0, self.h])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""

        xy = (np.random.random((self.numpoints, 2))-0.5)*10
        phero_color = [0.01, 0.5, 0.99]
        # c = [.150, .50]
        self.base.observation_aggregate()
        ind = 0
        while True:
            self.base.step()
            self.base.learn()
            time.sleep(delay)

            xy = self.base.blobs.visualise_blobs()
            s = np.ones(self.numpoints) * 100.0
            c = np.ones(self.numpoints) * .25

            # if ind % 500000 == 0:
            #     print(self.base.food.xy.sum())

            tmp = self.base.food.visualise_food()
            xy = np.append(xy, (np.dstack((tmp[0], tmp[1])))[0] + 0.1, axis=0)
            s = np.append(s, np.ones(tmp[0].shape[0]) * 40, axis=0)
            c = np.append(c, np.ones(tmp[0].shape[0]) * 0.75, axis=0)

            yield np.c_[xy[:,0], xy[:,1], s, c]

    def update(self, i):
        """Update the scatter plot."""
        data = next(self.stream)
        # print(data.shape)

        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        self.scat.set_sizes(abs(data[:, 2]))
        # Set colors..
        self.scat.set_array(data[:, 3])

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,



# if __name__ == '__main__':
with open("config.json") as f:
    cfg = json.load(f)
    print('\n\n', cfg, '\n')
    cfg['base']['epsilon'] = 3.1
    cfg['base']['eps_dec'] = .5
    cfg['main']['n_steps'] = 200000
    f.close()

model_path = 'model/model.pt'
base = Base(epsilon=cfg['base']['epsilon'], eps_dec=cfg['base']['eps_dec'], padding=cfg['main']['padding'],\
            eps_min=cfg['base']['eps_min'], lr=cfg['base']['lr'], gamma=cfg['base']['gamma'],\
            w=cfg['main']['width'], h=cfg['main']['height'], batch_size=cfg['main']['batch_size'],\
            n_blobs=cfg['main']['n_blobs'], n_pheromones=cfg['base']['n_pheromones'], visualise=True,\
            n_steps=cfg["main"]["n_steps"], model_path=model_path, n_prev=cfg['base']['n_prev'])

base.setup()
model_param = torch.load(model_path, map_location=torch.device('cpu'))
base.agent.model.load_state_dict(model_param)
base.agent.target.load_state_dict(model_param)

a = AnimatedScatter(base=base)
plt.show()