import copy
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from get_env_args import get_env_args
from monitor_server import MonitorServer
from models import PolicyModel, CriticModel
import argparse as ap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

server = MonitorServer()

class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, max_len=10000):
        self.buffer = []
        self.max_len = max_len

    def add(self, state, action, reward, next_state, done, log_prob_old):
        if len(self.buffer) >= self.max_len:
            self.buffer.pop(0)
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)
        self.buffer.append((state, action, reward, next_state, done, log_prob_old))

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, index):
        return self.buffer[index]

class ACNeuralAgent:
    def __init__(self, actor: nn.Module, critic: nn.Module):
        self.actor = actor
        self.actor.eval()
        self.critic = critic
        self.target_critic = copy.deepcopy(critic)
        self.critic.eval()
        self.target_critic.eval()
        self.gamma = 0.99
        self.entropy_coefficient = 0.01
        self.replay_buffer = ReplayBuffer()
        self.critic_criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        self.actor.eval()
        with torch.no_grad():
            mu, std = self.actor(state)
            mu = mu.view(-1)
            std = std.view(-1)
            covariance_matrix = torch.diag_embed(std)
            d = MultivariateNormal(mu, covariance_matrix)
            action = d.sample()
        return action, d.log_prob(action)

    def update_target_model(self, tau=0.001):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def experience_replay(self): 
        batch_size = 1024
        single_batch = True
        if len(self.replay_buffer) < batch_size:
            return 0, 0
        self.actor.train()
        self.critic.train()
        dl = torch.utils.data.DataLoader(self.replay_buffer, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device), drop_last=True)
        batches = dl
        critic_avg_loss = 0
        policy_avg_loss = 0
        processed_batches = 0
        for batch in batches:
            states, actions, rewards, next_states, dones, log_prob_old = batch
            with torch.no_grad():
                td_target = rewards + self.gamma * self.target_critic(next_states).view(-1) * (1 - dones.float())
                td_target = td_target.detach()
            value = self.critic(states).view(-1)
            mu, std = self.actor(states)
            covariance_matrix = torch.diag_embed(std)
            d = MultivariateNormal(mu, covariance_matrix)
            critic_loss = self.critic_criterion(value, td_target)

            log_prob = d.log_prob(actions)
            with torch.no_grad():
                delta = td_target - value
                delta = delta.detach()
                importance_weight = torch.exp(log_prob - log_prob_old).detach()
                eps = 0.2
                idx_pos = delta > 0
                g = torch.ones_like(delta) * (1 - eps)
                g[idx_pos] = (1 + eps)
                importance_weight = torch.min(importance_weight, g)
            actor_loss = -log_prob * importance_weight * delta - self.entropy_coefficient * d.entropy()

            self.critic_optimizer.zero_grad()
            critic_loss = critic_loss.mean()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
            critic_avg_loss += critic_loss.item()
            self.critic_optimizer.step()

            self.optimizer.zero_grad()
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
            policy_avg_loss += actor_loss.item()
            self.optimizer.step()
            processed_batches += 1
            if single_batch:
                break
            
        self.update_target_model()
        critic_avg_loss /= processed_batches
        policy_avg_loss /= processed_batches
        self.critic.eval()
        self.actor.eval()
        return policy_avg_loss, critic_avg_loss

    def iter_to_infinity(self):
        i = 0
        while True:
            yield i
            i += 1

    def fit(self, env: gym.Env, num_episodes: int = 10):
        loading_bar = tqdm(self.iter_to_infinity(), desc="Training")
        best_reward = -np.inf
        for episode in loading_bar:
            state, _ = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            while not done:
                action, log_prob = self.select_action(state)
                action = action.cpu().numpy()
                next_state, reward, terminated, truncated, _ = env.step(action)
                if episode % num_episodes == 0 and server.is_connected():
                    screenshot = env.render()
                    server.send(screenshot)
                done = terminated or truncated
                self.replay_buffer.add(state, action, reward, next_state, done, log_prob)
                policy_loss, critic_loss = self.experience_replay()
                episode_reward += reward
                steps += 1
                state = next_state
            if episode % num_episodes == 0 and server.is_connected():
                server.send_paused()
            best_reward = max(best_reward, episode_reward)
            info = {"Reward": episode_reward, "Best Reward": best_reward, "Policy Loss": policy_loss, "Critic Loss": critic_loss}
            loading_bar.set_postfix(info)
            info["Episode"] = episode
            server.send_info(info)

def main(env_name):

    args = get_env_args(env_name)
    env = gym.make(env_name, **args)
    model = PolicyModel(state_space=env.observation_space, action_space=env.action_space, deterministic=False)
    critic = CriticModel(state_space=env.observation_space)
    agent = ACNeuralAgent(actor=model, critic=critic)
    
    agent.fit(env)
    env.close()

if __name__ == "__main__":
    parser = ap.ArgumentParser(description="Actor Critic")
    parser.add_argument("--env", type=str, default="BipedalWalker-v3", required=False, help="Environment to train on")
    args = parser.parse_args()
    main(args.env)