from .model import Model
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
        self.input_size = 130 * n_prev
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model(input_size=self.input_size, rotation=5,\
                           n_pheromones=self.n_pheromones, batch_size=batch_size)
        self.target = Model(input_size=self.input_size, rotation=5,\
                           n_pheromones=self.n_pheromones, batch_size=batch_size)
        try:
            model_params = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(model_params)
        except:
            print('invalid path:', model_path)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()

        self.memory = Memory(input_dims=[self.input_size], batch_size=self.batch_size,\
                             mem_size_per_agent=5000, n_blobs=n_blobs)
        self.prev_observation = Prev_Observation(n_prev=n_prev)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion_1 = nn.MSELoss()
        # self.criterion_1 = nn.SmoothL1Loss()
        self.iter_cntr = 0
        self.replace_target = 1950
        self.batch_list = np.arange(self.batch_size)
        self.loss_mean = 0.0
        self.reward_mean = 0.0

    def train(self):
        max_mem = len(self.memory)
        if max_mem < self.batch_size:
            return 0

        state_batch, new_state_batch, action_batch, reward_batch, done_batch = self.memory.retrieve()

        state_batch = torch.tensor(state_batch).to(self.device)
        new_state_batch = torch.tensor(new_state_batch).to(self.device)
        reward_batch = torch.tensor(reward_batch).to(self.device)
        done_batch = torch.tensor(done_batch).to(self.device)

        rotation, pheromone, pickup_drop = self.model.forward(state_batch)

        rotation = rotation[self.batch_list, action_batch[:, 0]].reshape(-1, 1)
        pickup_drop = pickup_drop[self.batch_list, action_batch[:, 1]].reshape(-1, 1)
        pheromone = pheromone[self.batch_list, action_batch[:, 2]].reshape(-1, 1)

        rotation_t, pheromone_t, pickup_drop_t = self.target.forward(new_state_batch)
        rotation_t, pheromone_t, pickup_drop_t = rotation_t.detach().clone(),\
                                                         pheromone_t.detach().clone(),\
                                                         pickup_drop_t.detach().clone()

        pheromone_t = torch.max(pheromone_t, dim=1)[0].reshape(-1, 1)
        rotation_t = torch.max(rotation_t, dim=1)[0].reshape(-1, 1)
        pickup_drop_t = torch.max(pickup_drop_t, dim=1)[0].reshape(-1, 1)
        
        rotation_t[done_batch] = 0.0
        # jump_t[done_batch] = 0.0
        pheromone_t[done_batch] = 0.0
        pickup_drop_t[done_batch] = 0.0

        rotation_t = reward_batch + rotation_t * self.gamma
        # jump_t = reward_batch + jump_t * self.gamma
        pheromone_t = reward_batch + pheromone_t * self.gamma
        pickup_drop_t = reward_batch + pickup_drop_t * self.gamma

        loss_1 = self.criterion_1(rotation, rotation_t)
        loss_2 = self.criterion_1(pheromone, pheromone_t)
        # loss_3 = self.criterion_1(jump, jump_t)
        loss_4 = self.criterion_1(pickup_drop, pickup_drop_t)
        loss = loss_1 + loss_2 + loss_4
        self.optimizer.zero_grad()
        loss.backward()
        self.reward_mean = torch.mean(reward_batch)
        self.loss_mean = loss.item()
        
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()

        self.iter_cntr += 1
        if self.iter_cntr % self.replace_target == 0:
            self.target.load_state_dict(self.model.state_dict())

    def reset(self, epsilon):
        self.epsilon = epsilon

    def save_model(self, temp=False, i=0):
        if temp:
            # torch.save(self.model.state_dict(), 'model/temp/model_'+str(i)+'.pt')
            pass
        else:
            torch.save(self.model.state_dict(), 'model/model.pt')

