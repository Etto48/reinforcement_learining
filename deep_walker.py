import copy
from typing import Optional
import gymnasium as gym
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class PolicyModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, depth=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.layers = nn.Sequential()
        for i in range(depth):
            in_features = input_dim if i == 0 else hidden_dim
            out_features = hidden_dim if i < depth - 1 else output_dim * 2
            self.layers.append(nn.Linear(in_features, out_features))
            if i < depth - 1:
                self.layers.append(nn.ReLU())
    
    def forward(self, x: torch.Tensor):
        x = self.layers(x)
        mu, logvar = x.chunk(2, dim=-1)
        std = F.sigmoid(0.5 * logvar) + 1e-6
        mu = F.tanh(mu)
        return mu, std

class ExperienceBuffer(torch.utils.data.Dataset):
    def __init__(self, discount=0.99):
        self.buffer = []
        self.discount = discount
        self.episode_beginnings = []

    def begin_new_episode(self):
        if len(self.episode_beginnings) == 0:
            self.episode_beginnings.append(0)
            return
        gain = 0
        for i in reversed(range(self.episode_beginnings[-1], len(self.buffer))):
            state, action, reward = self.buffer[i]
            gain = reward + self.discount * gain
            self.buffer[i] = (state, action, gain)
        self.episode_beginnings.append(len(self.buffer))

    def add(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def clear(self):
        self.buffer.clear()
        self.episode_beginnings.clear()

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return self.buffer[index]

class REINFORCENeuralAgent:
    def __init__(self, model: nn.Module):
        self.model = model
        self.batch_size = 64
        self.batch_per_replay = 10
        self.gamma = 0.99
        self.replay_buffer = ExperienceBuffer(discount=self.gamma)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, amsgrad=True)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            mu, std = self.model(state)
            d = MultivariateNormal(mu, torch.diag(std))
            action = d.sample()
        return action

    def experience_replay(self):    
        dl = torch.utils.data.DataLoader(self.replay_buffer, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=device))
        loading_bar = tqdm(dl, desc="Experience Replay", total=len(dl))
        self.model.train()
        for batch in loading_bar:    
            states, actions, rewards = batch
            mu, std = self.model(states)
            d = MultivariateNormal(mu, torch.einsum('ij, jh -> ijh', std, torch.eye(self.model.output_dim)))
            log_probs = d.log_prob(actions)
            loss: torch.Tensor = -log_probs * rewards
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.model.eval()

    def fit(self, env: gym.Env, num_episodes: int = 256):
        while True:
            loading_bar = tqdm(range(num_episodes), desc="Training", total=num_episodes)
            for episode in loading_bar:
                state, _ = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = self.select_action(state)
                    action = action.cpu().numpy()
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    self.replay_buffer.add(state, action, reward)
                    episode_reward += reward
                    state = next_state
                loading_bar.set_postfix({"Episode Reward": episode_reward})
                self.replay_buffer.begin_new_episode()
            self.experience_replay()
            self.replay_buffer.clear()

def main():
    
    model = PolicyModel(input_dim=24, output_dim=4)
    agent = REINFORCENeuralAgent(model=model)
    env = gym.make("BipedalWalker-v3")
    agent.fit(env)
    env.close()

if __name__ == "__main__":
    main()