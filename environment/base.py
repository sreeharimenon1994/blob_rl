from .blobs import Blobs
from .pheromones import Pheromones
from .food import Food
from agent.agent import Agent
from agent.rewards import Reward
from .hill import Hill
import numpy as np
import torch
padding = 2

class Base:
    """Base for all the env objects to communicate"""
    def __init__(self, epsilon, eps_dec, eps_min, lr, w, h, n_blobs, n_pheromones, batch_size,\
                 gamma, visualise, padding, n_steps, model_path, n_prev):
        super(Base, self).__init__()
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.lr = lr
        self.n_blobs = n_blobs
        self.w = w
        self.h = h
        self.gamma = gamma
        self.padding = padding
        self.blobs = None
        self.agent = None
        self.n_pheromones = n_pheromones
        self.food = None
        self.hill = None
        self.model = None
        self.target = None
        self.optimizer = None
        self.observation = None
        self.reward = None
        self.batch_size = batch_size
        self.jump_strength = 1
        self.n_prev = n_prev
        self.visualise = visualise
        self.n_steps = n_steps
        self.model_path = model_path
        self.done_steps = n_steps - 1
        self.step_cntr = 0
    
    def reset(self):
        self.blobs.reset()
        self.food.reset()
        self.agent.reset(epsilon=self.epsilon)

    def setup(self):
        self.blobs = Blobs(n_blobs=self.n_blobs, w=self.w, h=self.h, padding=self.padding, n_steps=self.n_steps)
        self.hill = Hill(w=self.w, h=self.h, padding=self.padding)
        self.food = Food(w=self.w, h=self.h, hill=self.hill, padding=self.padding)
        for x in range(self.n_pheromones):
            pheromones = Pheromones(color=x, w=self.w, h=self.h, padding=self.padding)
            self.blobs.add_pheromones(pheromones)

        self.reward = Reward(blobs=self.blobs, food=self.food, w=self.w, h=self.h, hill=self.hill, padding=self.padding)
        self.agent = Agent(input_size=10, rotation=1, n_pheromones=self.n_pheromones,\
                           lr=self.lr, epsilon=self.epsilon, gamma=self.gamma,\
                           eps_dec=self.eps_dec, eps_min=self.eps_min, batch_size=self.batch_size,\
                           n_blobs=self.n_blobs, n_prev=self.n_prev, model_path=self.model_path)

    def observation_aggregate(self):
        xy = self.blobs.xyfa[:, 0]
        food = self.food.observation(xy)
        blobs = self.blobs.observation()
        hill = self.hill.observation(xy)
        obs = np.append(blobs, hill, axis=1)
        obs = np.append(obs, food, axis=1)

        self.agent.prev_observation.store(obs)
        self.observation = self.agent.prev_observation.state

    def choose_action(self, state):
        if np.random.random() > self.agent.epsilon:
            state = torch.tensor(state).to(self.agent.device)
            with torch.no_grad():
                rotation, pheromones, jump, pick_drop = self.agent.target.forward(state)
                rotation = (torch.argmax(rotation, dim=1).detach().cpu().numpy()).ravel() 
                jump = jump.detach().cpu().numpy().reshape(1, -1)
                pheromones = (torch.argmax(pheromones, dim=1).detach().cpu().numpy()).ravel()
                pick_drop = (torch.argmax(pick_drop, dim=1).detach().cpu().numpy()).reshape(1, -1)
            rotation = rotation - 3//2
        else:
            rotation = np.random.randint(low=0, high=3, size=self.n_blobs) - 3//2
            jump = np.random.random(size=self.n_blobs) * self.jump_strength
            pick_drop = np.random.randint(low=0, high=3, size=self.n_blobs)
            pheromones = np.random.randint(low=0, high=self.n_pheromones, size=self.n_blobs)

        return rotation, jump, pick_drop, pheromones

    def step(self):
        self.step_cntr += 1
        done = True

        state = self.observation.copy()
        rotation, jump, pick_drop, pheromones = self.choose_action(state)
        self.blobs.xyfa[:, 1, 1] = pick_drop
        self.blobs.update_pos(rotation=rotation * 0.16355283, jump=jump)
        self.blobs.update_pheromones(pheromones=pheromones)

        self.observation_aggregate()
        new_state = self.observation.copy()

        # print('\n\nself.agent.epsilon:', self.agent.epsilon)
        reward = self.reward.calculate()
        # reward = np.array([2.0, 3.0])
        # print('\n',reward.sum(), '\nfood sum ------>', self.food.xy[padding: -padding, padding: -padding].sum(), '\n\n')
        if self.step_cntr == self.done_steps:
            done = False

        if len(self.agent.prev_observation) == self.n_prev:
            action = np.dstack((rotation, jump, pick_drop, pheromones))
            self.agent.memory.store(state=state, action=action, reward=reward, state_=new_state, done=done)

    def learn(self):
        self.agent.epsilon = self.agent.epsilon - self.agent.eps_dec if self.agent.epsilon > self.agent.eps_min\
                             else self.agent.eps_min
        # print(self.agent.epsilon, 'epsilon')
        if not self.visualise:
            self.agent.train()
