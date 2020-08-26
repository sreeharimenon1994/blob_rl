#from .model import Model
from .memory import Memory, Prev_Observation
from .model import Model
import torch
from torch import nn
import numpy as np


class Agent:
    """docstring for Agent"""
    def __init__(self, input_size, rotation, n_pheromones, lr, batch_size,\
                 epsilon, eps_dec, eps_min, gamma, n_prev, n_blobs, model_path):
        super(Agent, self).__init__()
        self.n_pheromones = n_pheromones
        self.lr = lr
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.gamma = gamma
        self.input_size = 71 * n_prev
        # self.input_size = 121
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.model = Model(input_size=self.input_size, rotation=5,\
                           n_pheromones=self.n_pheromones, batch_size=batch_size)
        self.target = Model(input_size=self.input_size, rotation=5,\
                           n_pheromones=self.n_pheromones, batch_size=batch_size)
        # try:
        #     model_params = torch.load(model_path, map_location=self.device)
        #     self.model.load_state_dict(model_params)
        # except:
        #     print('invalid path:', model_path)

        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        self.memory = Memory(input_dims=[self.input_size], batch_size=self.batch_size,\
                             mem_size_per_agent=5000, n_blobs=n_blobs)
        self.prev_observation = Prev_Observation(n_prev=n_prev)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.criterion_1 = nn.MSELoss()
        self.criterion_1 = nn.SmoothL1Loss()
        self.iter_cntr = 0
        self.replace_target = 1990
        self.batch_list = np.arange(self.batch_size)
        # self.loss_mean = 0.0
        # self.reward_mean = 0.0

    def train(self):
        max_mem = len(self.memory)
        if max_mem < self.batch_size:
            return 0

        state_batch, new_state_batch, action_batch, reward_batch, done_batch = self.memory.retrieve()

        state_batch = torch.tensor(state_batch).to(self.device)
        new_state_batch = torch.tensor(new_state_batch).to(self.device)
        reward_batch = torch.tensor(reward_batch).to(self.device).reshape(1, -1)
        done_batch = torch.tensor(done_batch).to(self.device)

        with torch.no_grad():
            # rotation, pheromone, pickup_drop = self.model.forward(state_batch)
            rotation, pheromone = self.model.forward(state_batch)

            rotation_t, pheromone_t = self.target.forward(new_state_batch)
            pheromone_t = torch.max(pheromone_t, dim=1).values
            rotation_t = torch.max(rotation_t, dim=1).values

            # print(rotation_t.shape, rotation.shape, action_batch.shape)
            # print('reward_batch', reward_batch.shape)
            # print('\ndone_batch', done_batch.shape)
            # print('\nrotation_t * ~done_batch', (rotation_t * ~done_batch).shape, '\n\n\n\n')

            rotation_t = reward_batch + self.gamma * rotation_t * ~done_batch
            pheromone_t = reward_batch + self.gamma * pheromone_t * ~done_batch
            
            # print(rotation_t.shape, rotation.shape)

            rotation[self.batch_list, action_batch[:, 0]] = rotation_t
            pheromone[self.batch_list, action_batch[:, 1]] = pheromone_t
        
        output = self.model.forward(state_batch)        
        # print(output[0].shape, 'poli\n')
        loss_1 = self.criterion_1(output[0], rotation)
        loss_2 = self.criterion_1(output[1], pheromone)
        # loss_3 = self.criterion_1(jump, jump_t)
        # loss_4 = self.criterion_1(pickup_drop, pickup_drop_t)
        loss = loss_1 + loss_2
        self.optimizer.zero_grad()
        loss.backward()
        # self.reward_mean = torch.mean(reward_batch)
        # self.loss_mean = loss.item()
        
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()

        self.iter_cntr += 1
        if self.iter_cntr == self.replace_target:
            # print('model replaced')
            self.iter_cntr = 0
            self.target.load_state_dict(self.model.state_dict())

    def reset(self, epsilon):
        self.epsilon = epsilon

    def save_model(self, temp=False, i=0):
        if temp:
            # torch.save(self.model.state_dict(), 'model/temp/model_'+str(i)+'.pt')
            pass
        else:
            torch.save(self.model.state_dict(), 'model/model.pt')

