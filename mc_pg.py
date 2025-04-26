import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from get_env_args import get_env_args
from rgb_array_server import RgbArrayServer
from models import PolicyModel, CriticModel
import argparse as ap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

server = RgbArrayServer()

class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, discount):
        self.buffer = []
        self.new_episode_index = 0
        self.discount = discount

    def add(self, state, action, reward):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        self.buffer.append((state, action, reward))

    def clear(self):
        self.buffer.clear()
        self.new_episode_index = 0

    def next_episode(self):
        current_reward = 0
        for i in reversed(range(self.new_episode_index, len(self.buffer))):
            state, action, reward = self.buffer[i]
            current_reward = reward + self.discount * current_reward
            reward = current_reward
            self.buffer[i] = (state, action, reward)
        self.new_episode_index = len(self.buffer)

    def single_batch(self):
        assert len(self.buffer) == self.new_episode_index, "Buffer not finalized, call next_experience() first"
        states, actions, rewards = zip(*self.buffer)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        return states, actions, rewards

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return self.buffer[index]

class REINFORCENeuralAgent:
    def __init__(self, model: nn.Module, baseline: nn.Module = None):
        self.model = model
        self.baseline = baseline
        self.gamma = 0.99
        self.replay_buffer = ReplayBuffer(discount=self.gamma)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.baseline_optimizer = torch.optim.AdamW(self.baseline.parameters(), lr=0.001) if baseline else None

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            mu, std = self.model(state)
            mu = mu.view(-1)
            std = std.view(-1)
            covariance_matrix = torch.diag_embed(std)
            d = MultivariateNormal(mu, covariance_matrix)
            action = d.sample()
        return action

    def experience_replay(self):    
        self.model.train()
        if len(self.replay_buffer) < 1:
            return 0
        batch = self.replay_buffer.single_batch()
        states, actions, returns = batch
        if self.baseline is not None:
            self.baseline.train()
            self.baseline_optimizer.zero_grad()
            baseline_loss = F.mse_loss(self.baseline(states).view(-1), returns)
            baseline_loss.backward()
            nn.utils.clip_grad_norm_(self.baseline.parameters(), 1)
            self.baseline_optimizer.step()
            self.baseline.eval()
        mu, std = self.model(states)
        covariance_matrix = torch.diag_embed(std)
        d = MultivariateNormal(mu, covariance_matrix)
        log_probs = d.log_prob(actions)
        entropy = d.entropy()
        with torch.no_grad():
            advantage = (returns - (self.baseline(states).view(-1) if self.baseline else returns.mean())).detach() / (returns.std() + 1e-8)
        loss: torch.Tensor = -log_probs * advantage
        self.optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.model.eval()
        return loss.item()

    def iter_to_infinity(self):
        i = 0
        while True:
            yield i
            i += 1

    def fit(self, env: gym.Env, num_episodes: int = 10):
        loading_bar = tqdm(self.iter_to_infinity(), desc="Training")
        avg_reward = 0
        best_reward = -np.inf
        steps = 0
        loss = "N/A"
        for episode in loading_bar:
            state, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                action = action.cpu().numpy()
                next_state, reward, terminated, truncated, _ = env.step(action)
                if episode % num_episodes == 0:
                    screenshot = env.render()
                    server.send(screenshot)
                done = terminated or truncated
                self.replay_buffer.add(state, action, reward)
                episode_reward += reward
                steps += 1
                state = next_state
            if episode % num_episodes == 0 and server.is_connected():
                server.send_clear()
            best_reward = max(best_reward, episode_reward)
            avg_reward = 0.9 * avg_reward + 0.1 * episode_reward
            self.replay_buffer.next_episode()
            if (episode + 1) % num_episodes == 0:
                loss = self.experience_replay()
                self.replay_buffer.clear()
                
            loading_bar.set_postfix({"Reward": episode_reward, "Avg Reward": avg_reward, "Best Reward": best_reward, "Loss": loss})
            
            

def main(env_name):
    args = get_env_args(env_name)
    env = gym.make(env_name, **args)
    model = PolicyModel(state_space=env.observation_space, action_space=env.action_space, deterministic=False)
    agent = REINFORCENeuralAgent(model=model)
    
    agent.fit(env)
    env.close()

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Vanilla Policy Gradient")
    parser.add_argument("--env", type=str, default="BipedalWalker-v3", required=False, help="Environment to train on")
    args = parser.parse_args()
    main(args.env)