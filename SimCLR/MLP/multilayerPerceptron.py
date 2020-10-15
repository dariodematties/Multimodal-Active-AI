# This is almost a copy paste from https://www.kaggle.com/pinocookie/pytorch-simple-mlp

import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # convert tensor (batch_size, num_of_chanels, W, H) --> (batch_size, num_of_chanels*W*H)
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x
