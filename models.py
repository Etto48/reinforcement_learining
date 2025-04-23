from typing import Optional
import gymnasium as gym
import torch
import torch.nn as nn

class PolicyModel(nn.Module):
    def __init__(self, state_space: gym.Space, action_space: gym.Space, deterministic=False, hidden_dim=64, depth=3):
        super().__init__()
        self.input_dim = state_space.shape[0]
        self.output_dim = action_space.shape[0]
        self.output_dim = self.output_dim * 2 if not deterministic else self.output_dim
        self.output_scale = torch.tensor(action_space.high - action_space.low)
        self.output_bias = torch.tensor(action_space.low)
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.deterministic = deterministic
        self.layers = nn.Sequential()
        for i in range(depth):
            in_features = self.input_dim if i == 0 else hidden_dim
            out_features = hidden_dim if i < depth - 1 else self.output_dim
            self.layers.append(nn.Linear(in_features, out_features))
            if i < depth - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x: torch.Tensor):
        if x.ndim == 1:
            x = x.view(1, -1)
        x = self.layers(x)
        if not self.deterministic:
            mu, logvar = x.chunk(2, dim=-1)
            std = torch.sigmoid(0.5*logvar) + 0.01
            mu = torch.sigmoid(mu)
            return mu * self.output_scale + self.output_bias, std
        else:
            x = torch.sigmoid(x)
            x = x * self.output_scale + self.output_bias
            return x
    
class CriticModel(nn.Module):
    def __init__(self, state_space: gym.Space, action_space: Optional[gym.Space] = None, hidden_dim=64, depth=3):
        super().__init__()
        action_size = action_space.shape[0] if action_space is not None else 0
        self.input_dim = state_space.shape[0] + action_size
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.layers = nn.Sequential()
        for i in range(depth):
            in_features = self.input_dim if i == 0 else hidden_dim
            out_features = hidden_dim if i < depth - 1 else 1
            self.layers.append(nn.Linear(in_features, out_features))
            if i < depth - 1:
                self.layers.append(nn.ReLU())

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        if state.ndim == 1:
            state = state.view(1, -1)
        if action.ndim == 1:
            action = action.view(1, -1)
        x = torch.cat((state, action), dim=-1)
        x = self.layers(x)
        return x
        