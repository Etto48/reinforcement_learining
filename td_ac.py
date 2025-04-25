import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from rgb_array_server import RgbArrayServer
from models import PolicyModel, CriticModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

server = RgbArrayServer()

class ExperienceBuffer(torch.utils.data.Dataset):
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
    def __init__(self, model: nn.Module, critic: nn.Module):
        self.model = model
        self.critic = critic
        self.gamma = 0.99
        self.entropy_coefficient = 1
        self.replay_buffer = ExperienceBuffer()
        self.critic_criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=1e-3)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            mu, std = self.model(state)
            mu = mu.view(-1)
            std = std.view(-1)
            d = MultivariateNormal(mu, torch.diag(std ** 2))
            action = d.sample()
        return action, d.log_prob(action)

    def experience_replay(self): 
        batch_size = 1024
        single_batch = True
        if len(self.replay_buffer) < batch_size:
            return 0, 0
        self.model.train()
        self.critic.train()
        dl = torch.utils.data.DataLoader(self.replay_buffer, batch_size=batch_size, shuffle=True, generator=torch.Generator(device=device), drop_last=True)
        batches = dl
        critic_avg_loss = 0
        policy_avg_loss = 0
        processed_batches = 0
        for batch in batches:
            states, actions, rewards, next_states, dones, log_prob_old = batch
            with torch.no_grad():
                td_target = rewards + self.gamma * self.critic(next_states).view(-1) * (1 - dones.float())
            td_target = td_target.detach()
            v_of_s = self.critic(states).view(-1)
            with torch.no_grad():
                td_error = td_target - v_of_s
            td_error = td_error.detach()
            mu, std = self.model(states)
            d = MultivariateNormal(mu, torch.einsum('ij, jh -> ijh', std, torch.eye(self.model.output_dim)))
            critic_loss = self.critic_criterion(v_of_s, td_target)
            importance_weight = torch.exp(d.log_prob(actions) - log_prob_old)
            actor_loss = importance_weight * td_error - self.entropy_coefficient * d.entropy()

            self.critic_optimizer.zero_grad()
            critic_loss = critic_loss.mean()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            critic_avg_loss += critic_loss.item()
            self.critic_optimizer.step()

            self.optimizer.zero_grad()
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            policy_avg_loss += actor_loss.item()
            self.optimizer.step()
            processed_batches += 1
            self.entropy_coefficient = max(0.01, self.entropy_coefficient * 0.9)
            if single_batch:
                break
            
        critic_avg_loss /= processed_batches
        policy_avg_loss /= processed_batches
        self.critic.eval()
        self.model.eval()
        return policy_avg_loss, critic_avg_loss

    def iter_to_infinity(self):
        i = 0
        while True:
            yield i
            i += 1

    def fit(self, env: gym.Env, num_episodes: int = 32):
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
                if episode % 10 == 0 and server.is_connected():
                    screenshot = env.render()
                    server.send(screenshot)
                done = terminated or truncated
                self.replay_buffer.add(state, action, reward, next_state, done, log_prob)
                policy_loss, critic_loss = self.experience_replay()
                episode_reward += reward
                steps += 1
                state = next_state
            best_reward = max(best_reward, episode_reward)
            loading_bar.set_postfix({"Avg Reward": episode_reward, "Best Reward": best_reward, "Policy Loss": policy_loss, "Critic Loss": critic_loss})

def main():
    
    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")
    model = PolicyModel(state_space=env.observation_space, action_space=env.action_space, deterministic=False)
    critic = CriticModel(state_space=env.observation_space)
    agent = ACNeuralAgent(model=model, critic=critic)
    
    agent.fit(env)
    env.close()

if __name__ == "__main__":
    main()