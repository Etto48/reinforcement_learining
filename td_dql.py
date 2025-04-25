import copy
from typing import Optional
import gymnasium as gym
import pygame
import torch
import torch.nn as nn
from tqdm import tqdm

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

class FFModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, depth=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.layers = nn.Sequential()
        for i in range(depth):
            in_features = input_dim if i == 0 else hidden_dim
            out_features = hidden_dim if i < depth - 1 else output_dim
            self.layers.append(nn.Linear(in_features, out_features))
            if i < depth - 1:
                self.layers.append(nn.ReLU())
    
    def forward(self, x):
        x = self.layers(x)
        return x

class ReplayBuffer(torch.utils.data.Dataset):
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def add(self, state, action, reward, next_state):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state))
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        state, action, reward, next_state = self.buffer[idx]
        return (torch.tensor(state, dtype=torch.float32),
                torch.tensor(action, dtype=torch.long),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(next_state, dtype=torch.float32) if next_state is not None else torch.zeros_like(torch.tensor(state, dtype=torch.float32)),
                torch.tensor(False, dtype=torch.bool) if next_state is None else torch.tensor(True, dtype=torch.bool))

class DQLNeuralAgent:
    def __init__(self, model: nn.Module):
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.replay_buffer = ReplayBuffer(max_size=10000)
        self.gamma = 0.99
        self.tau = 0.1
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.batch_size = 64

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action_values = self.model(state)
            action_probabilities = torch.softmax(action_values, dim=0)
            action = torch.multinomial(action_probabilities, 1).item()
        return action
    
    def update_target_model(self):
        weights = self.model.state_dict()
        target_weights = self.target_model.state_dict()
        for key in target_weights:
            target_weights[key] = self.tau * weights[key] + (1 - self.tau) * target_weights[key]
        self.target_model.load_state_dict(target_weights)

    def q_value_arrival_state(self, states):
        with torch.no_grad():
            q_values = self.target_model(states)
            q_values = q_values.max(dim=1)[0]
        return q_values

    def experience_replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        dl = torch.utils.data.DataLoader(self.replay_buffer, batch_size=self.batch_size, shuffle=True)
        batch = next(iter(dl))
        states, actions, rewards, next_states, not_done = batch

        next_state_values = torch.zeros((self.batch_size))
        with torch.no_grad():
            next_state_values[not_done] = self.q_value_arrival_state(next_states[not_done])
            target_values = rewards + self.gamma * next_state_values
        
        self.model.train()
        self.optimizer.zero_grad()
        current_values: torch.Tensor  = self.model(states)
        current_values = current_values.gather(1, actions.unsqueeze(1)).squeeze()
        loss: torch.Tensor = self.criterion(current_values, target_values)
        loss.backward()
        self.optimizer.step()
        self.model.eval()
        self.update_target_model()

    def fit(self, env: gym.Env, num_episodes=1000):
        loading_bar = tqdm(range(num_episodes), desc="Training", total=num_episodes)
        for episode in loading_bar:
            state, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.replay_buffer.add(state, action, reward, next_state if not done else None)
                episode_reward += reward
                state = next_state
                self.experience_replay()
            loading_bar.set_postfix({"Episode Reward": episode_reward})

def human_main():
    env = gym.make("LunarLander-v3", render_mode="human")
    env.reset()
    env.render()
    done = False
    while not done:
        action = pygame.key.get_pressed()
        if action[pygame.K_LEFT]:
            action = 1
        elif action[pygame.K_RIGHT]:
            action = 3
        elif action[pygame.K_UP]:
            action = 2
        else:
            action = 0
        observation, reward, terminated, truncated, _ = env.step(action)
        print(observation)
        done = terminated or truncated
        env.render()
        print(f"Reward: {reward}")
    env.close()

def main():
    
    model = FFModel(input_dim=8, output_dim=4)
    agent = DQLNeuralAgent(model=model)
    i = 0
    while True:
        i += 1
        env = gym.make("LunarLander-v3")
        agent.fit(env, num_episodes=9)
        env.close()
        env = gym.make("LunarLander-v3", render_mode="human")
        agent.fit(env, num_episodes=1)
        env.close()
        print(f"Episode {i*10} finished")

if __name__ == "__main__":
    main()