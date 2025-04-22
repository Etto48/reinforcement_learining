import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from rgb_array_server import RgbArrayServer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

server = RgbArrayServer()

class PolicyModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, depth=3):
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
        if x.ndim == 1:
            x = x.view(1, -1)
        x = self.layers(x) * 0.5
        mu, logvar = x.chunk(2, dim=-1)
        std = F.sigmoid(logvar) + 0.01
        mu = F.tanh(mu)
        return mu, std

class ExperienceBuffer(torch.utils.data.Dataset):
    def __init__(self, discount=0.99):
        self.buffer = []
        self.discount = discount

    def add(self, state, action, reward):
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        self.buffer.append((state, action, reward))

    def clear(self):
        self.buffer.clear()
    
    def single_batch(self):
        states, actions, rewards = zip(*self.buffer)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        current_reward = 0
        for i in reversed(range(len(rewards))):
            current_reward = rewards[i] + self.discount * current_reward
            rewards[i] = current_reward
        return states, actions, rewards

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return self.buffer[index]

class REINFORCENeuralAgent:
    def __init__(self, model: nn.Module):
        self.model = model
        self.gamma = 0.99
        self.replay_buffer = ExperienceBuffer(discount=self.gamma)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            mu, std = self.model(state)
            mu = mu.view(-1)
            std = std.view(-1)
            d = MultivariateNormal(mu, torch.diag(std))
            action = d.sample()
        return action

    def experience_replay(self):    
        self.model.train()
        batch = self.replay_buffer.single_batch()
        states, actions, rewards = batch
        rewards_std, rewards_mean = torch.std_mean(rewards)
        rewards = (rewards - rewards_mean) / (rewards_std + 1e-8)
        mu, std = self.model(states)
        d = MultivariateNormal(mu, torch.einsum('ij, jh -> ijh', std, torch.eye(self.model.output_dim)))
        log_probs = d.log_prob(actions)
        entropy = d.entropy()
        loss: torch.Tensor = -log_probs * rewards
        self.optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        self.model.eval()
        return loss.item()

    def iter_to_infinity(self):
        i = 0
        while True:
            yield i
            i += 1

    def fit(self, env: gym.Env, num_episodes: int = 32):
        loading_bar = tqdm(self.iter_to_infinity(), desc="Training")
        avg_reward = 0
        best_reward = -np.inf
        steps = 0
        for episode in loading_bar:
            state, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                action = action.cpu().numpy()
                next_state, reward, terminated, truncated, _ = env.step(action)
                if episode % 10 == 0:
                    screenshot = env.render()
                    server.send(screenshot)
                done = terminated or truncated
                self.replay_buffer.add(state, action, reward)
                episode_reward += reward
                steps += 1
            episode_reward /= steps
            best_reward = max(best_reward, episode_reward)
            avg_reward = 0.9 * avg_reward + 0.1 * episode_reward
            loss = self.experience_replay()
            loading_bar.set_postfix({"Episode": episode, "Reward": episode_reward, "Avg Reward": avg_reward, "Best Reward": best_reward, "Loss": loss})
            self.replay_buffer.clear()
            

def main():
    
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = PolicyModel(input_dim=state_dim, output_dim=action_dim)
    agent = REINFORCENeuralAgent(model=model)
    
    agent.fit(env)
    env.close()

if __name__ == "__main__":
    main()