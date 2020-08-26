import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, input_size, rotation, n_pheromones, batch_size):
        super(Model, self).__init__()
        power = 3

        # total_outs = [rotation, n_pheromones, 3] # rot, phe, pick
        total_outs = [rotation, n_pheromones] # rot, phe, pick

        self.general = nn.Sequential(
                nn.Linear(input_size, 2**(4 + power)),
                nn.Linear(2**(4 + power), 2**(3 + power)),
                nn.Linear(2**(3 + power), 2**(2 + power)),
                nn.Linear(2**(2 + power), 2**(1 + power)),
                nn.Linear(2**(1 + power), input_size) ).cuda()

        self.branch = nn.ModuleList([])

        for nout in total_outs:
            self.branch.append(nn.Sequential(
                nn.Linear(input_size, 2**(3 + power)),
                nn.Linear(2**(3 + power), 2**(2 + power)),
                nn.Linear(2**(2 + power), 2**(1 + power)),
                nn.Linear(2**(1+ power), nout)).cuda())


    def forward(self, state):
        # print('state', state)
        general = self.general(state)

        outputs = []
        for head in self.branch:
            outputs.append(head(general))

        return outputs

