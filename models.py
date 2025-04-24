from typing import Optional
import gymnasium as gym
import torch
import torch.nn as nn

class PolicyModel(nn.Module):
    def __init__(self, state_space: gym.Space, action_space: gym.Space, deterministic=False, hidden_dim=128, depth=4):
        super().__init__()
        self.input_dim = 1
        for dim in state_space.shape:
            self.input_dim *= dim
        self.input_ndim = len(state_space.shape)
        if self.input_ndim == 3:
            self.input_dim = hidden_dim
        self.output_dim = 1
        for dim in action_space.shape:
            self.output_dim *= dim
        self.output_dim = self.output_dim * 2 if not deterministic else self.output_dim
        self.output_scale = torch.tensor(action_space.high - action_space.low)
        self.output_bias = torch.tensor(action_space.low)
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.deterministic = deterministic
        self.layers = nn.Sequential()
        if self.input_ndim == 3:
            self.layers.append(nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1, padding_mode="reflect"))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode="reflect"))
            self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.layers.append(nn.Flatten())
            self.layers.append(nn.ReLU())
            
        for i in range(depth):
            in_features = self.input_dim if i == 0 else hidden_dim
            out_features = hidden_dim if i < depth - 1 else self.output_dim
            self.layers.append(nn.Linear(in_features, out_features))
            if i < depth - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x: torch.Tensor):
        if x.ndim == 1 and self.input_ndim == 1:
            x = x.view(1, -1)
        elif x.ndim == self.input_ndim:
            x = x.unsqueeze(0)
        if self.input_ndim == 3:
            x = x.permute(0, 3, 1, 2)
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
        self.input_dim = 1
        for dim in state_space.shape:
            self.input_dim *= dim
        self.input_dim += action_size
        self.input_ndim = len(state_space.shape)
        if self.input_ndim == 3:
            self.input_dim = hidden_dim + action_size
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.layers = nn.Sequential()
        if self.input_ndim == 3:
            self.conv = nn.Sequential()
            self.conv.append(nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1, padding_mode="reflect"))
            self.conv.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv.append(nn.ReLU())
            self.conv.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode="reflect"))
            self.conv.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.conv.append(nn.Flatten())
            self.conv.append(nn.ReLU())
        for i in range(depth):
            in_features = self.input_dim if i == 0 else hidden_dim
            out_features = hidden_dim if i < depth - 1 else 1
            self.layers.append(nn.Linear(in_features, out_features))
            if i < depth - 1:
                self.layers.append(nn.ReLU())

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        if state.ndim == 1 and self.input_ndim == 1:
            state = state.view(1, -1)
        elif state.ndim == self.input_ndim:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.view(1, -1)
        if self.input_ndim == 3:
            state = state.permute(0, 3, 1, 2)
            state = self.conv(state)
        x = torch.cat((state, action), dim=-1)
        x = self.layers(x)
        return x
        