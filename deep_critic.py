import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class PolicyModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, depth=4):
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
    
class CriticModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, depth=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.layers = nn.Sequential()
        for i in range(depth):
            in_features = input_dim if i == 0 else hidden_dim
            out_features = hidden_dim if i < depth - 1 else 1
            self.layers.append(nn.Linear(in_features, out_features))
            if i < depth - 1:
                self.layers.append(nn.ReLU())

    def forward(self, x: torch.Tensor):
        if x.ndim == 1:
            x = x.view(1, -1)
        x = self.layers(x)
        return x

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
        self.gamma = 0.9
        self.replay_buffer = ExperienceBuffer()
        self.critic_criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=1e-4)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            mu, std = self.model(state)
            mu = mu.view(-1)
            std = std.view(-1)
            d = MultivariateNormal(mu, torch.diag(std))
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
            v_of_s = self.critic(states).view(-1)
            with torch.no_grad():
                td_error = td_target - v_of_s
            mu, std = self.model(states)
            d = MultivariateNormal(mu, torch.einsum('ij, jh -> ijh', std, torch.eye(self.model.output_dim)))
            critic_loss = self.critic_criterion(v_of_s, td_target)
            with torch.no_grad():
                importance_weight = torch.exp(d.log_prob(actions) - log_prob_old)
                importance_weight = torch.clamp(importance_weight, max=10)
            policy_loss = -d.log_prob(actions) * importance_weight * td_error.detach()

            self.critic_optimizer.zero_grad()
            critic_loss = critic_loss.mean()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            critic_avg_loss += critic_loss.item()
            self.critic_optimizer.step()

            self.optimizer.zero_grad()
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            policy_avg_loss += policy_loss.item()
            self.optimizer.step()
            processed_batches += 1
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
        plt.figure()
        plt.ion()
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
                if episode % 10 == 0:
                    screenshot = env.render()
                    plt.clf()
                    try:
                        del img
                    except UnboundLocalError:
                        pass
                    img = plt.imshow(screenshot)
                    plt.axis('off')
                    plt.pause(0.01)
                done = terminated or truncated
                self.replay_buffer.add(state, action, reward, next_state, done, log_prob)
                policy_loss, critic_loss = self.experience_replay()
                episode_reward += reward
                steps += 1
            episode_reward /= steps
            best_reward = max(best_reward, episode_reward)
            loading_bar.set_postfix({"Avg Reward": episode_reward, "Best Reward": best_reward, "Policy Loss": policy_loss, "Critic Loss": critic_loss})

def main():
    
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    model = PolicyModel(input_dim=state_dim, output_dim=action_dim)
    critic = CriticModel(input_dim=state_dim)
    agent = ACNeuralAgent(model=model, critic=critic)
    
    agent.fit(env)
    env.close()

if __name__ == "__main__":
    main()