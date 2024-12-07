import gymnasium as gym
import numpy as np
import random
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML
import imageio


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, 210, 160)
            self.flattened_size = self.conv(dummy_input).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Buffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device) / 255.0,
            torch.tensor(actions, dtype=torch.int64, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device) / 255.0,
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, state_shape, n_actions, buffer_capacity, batch_size, gamma, lr, epsilon_start, epsilon_min, epsilon_decay):
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_network = QNetwork(input_dim=state_shape[2], output_dim=n_actions).to(device)
        self.target_network = QNetwork(input_dim=state_shape[2], output_dim=n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = Buffer(capacity=buffer_capacity)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                return torch.argmax(self.q_network(state_tensor)).item()

    def optimize(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.view(self.batch_size, self.state_shape[2], self.state_shape[0], self.state_shape[1]).to(device)
        next_states = next_states.view(self.batch_size, self.state_shape[2], self.state_shape[0], self.state_shape[1]).to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards.unsqueeze(1) + self.gamma * max_next_q_values * (1 - dones.unsqueeze(1))

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_weights(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)

    def load_weights(self, filepath):
        self.q_network.load_state_dict(torch.load(filepath))
        self.target_network.load_state_dict(self.q_network.state_dict())


def chw_transform(state):
    return np.transpose(state, (2, 0, 1))


def train_agent(env, agent, num_episodes, target_update_freq, save_path):
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = chw_transform(state)
        total_reward = 0
        incorrect_guesses = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = chw_transform(next_state)

            if reward == -1:
                incorrect_guesses += 1
            if incorrect_guesses >= 11:
                done = True

            agent.buffer.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.optimize()

            if done:
                break

        agent.update_epsilon()
        if episode % target_update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode}: Total Reward = {total_reward}, Incorrect Guesses = {incorrect_guesses}")

    agent.save_weights(save_path)
    print(f"Weights saved to {save_path}")



def record_video(env, out_directory, fps=30, max_steps=1000):
    images = []
    obs = env.reset()
    state, _ = env.reset()
    rewards_sum = 0

    for _ in range(max_steps):
        action, _states = agent.select_action(obs)
        obs, rewards, dones, info = env.step(action)
        rewards_sum += rewards[0]
        img = env.render("rgb_array")
        images.append(img)

    imageio.mimsave(out_directory, [np.array(image) for image in images], fps=fps)
    print(f"Total reward: {rewards_sum}")


def run_episode(env, agent, load_path):
    agent.load_weights(load_path)
    state, _ = env.reset()
    state = chw_transform(state)
    total_reward = 0
    done = False
    incorrect_guesses = 0

    while not done:
        img = env.render(mode="rgb_array")
        plt.imshow(img)
        plt.axis("off")
        clear_output(wait=True)
        display(plt.gcf())
        plt.pause(0.01)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = torch.argmax(agent.q_network(state_tensor)).item()

        next_state, reward, done, _, _ = env.step(action)
        next_state = chw_transform(next_state)

        if reward == -1:
            incorrect_guesses += 1
            if incorrect_guesses >= 11:
                done = True

        state = next_state
        total_reward += reward

    print(f"Total Reward in Test Episode: {total_reward}, Incorrect Guesses: {incorrect_guesses}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("ALE/Hangman-v5", frameskip=50)

state_shape = env.observation_space.shape
n_actions = env.action_space.n

agent = Agent(
    state_shape=state_shape,
    n_actions=n_actions,
    buffer_capacity=10000,
    batch_size=64,
    gamma=0.99,
    lr=0.001,
    epsilon_start=1.0,
    epsilon_min=0.1,
    epsilon_decay=0.995,
)

train_agent(env, agent, num_episodes=5, target_update_freq=10, save_path="dqn_hangman.pth") #for training
# record_video(env, agent, "dqn_hangman_video.mp4") #for saving video
# run_episode(env, agent, load_path="dqn_hangman.pth") #for running saved episode with weights
