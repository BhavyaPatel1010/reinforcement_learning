"""Tabular Q-learning and DQN implementations built for this project."""
from __future__ import annotations

import random
from collections import deque

import numpy as np


class QLearningAgent:
    def __init__(self, action_size: int, bins: int = 8, alpha: float = 0.12,
                 gamma: float = 0.97, epsilon: float = 1.0, epsilon_min: float = 0.04,
                 epsilon_decay: float = 0.992):
        self.action_size, self.bins = action_size, bins
        self.alpha, self.gamma = alpha, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.q = {}

    def key(self, state):
        # A compact, interpretable tabular state. Keeping only the variables
        # most relevant to replenishment prevents an impractically sparse
        # 9-dimensional Q-table.
        s = np.asarray(state)
        inventory_position = int(np.clip((s[0] + s[1]) * self.bins, 0, self.bins * 2 - 1))
        demand_average = int(np.clip(s[5] / 2.5 * self.bins, 0, self.bins - 1))
        promotion = int(s[7] >= 0.5)
        return inventory_position, demand_average, promotion

    def values(self, state):
        key = self.key(state)
        if key not in self.q:
            self.q[key] = np.zeros(self.action_size, dtype=np.float32)
        return self.q[key]

    def act(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return int(np.argmax(self.values(state)))

    def learn(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma * float(np.max(self.values(next_state)))
        values = self.values(state)
        values[action] += self.alpha * (target - values[action])

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class DQNAgent:
    def __init__(self, state_size, action_size, device="cpu", lr=1e-3, gamma=0.97,
                 epsilon=1.0, epsilon_min=0.04, epsilon_decay=0.995):
        import torch
        import torch.nn as nn
        self.torch, self.device = torch, torch.device(device)
        self.action_size, self.gamma = action_size, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay
        self.memory = deque(maxlen=30_000)
        self.net = nn.Sequential(nn.Linear(state_size, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(), nn.Linear(96, action_size)).to(self.device)
        self.target = nn.Sequential(nn.Linear(state_size, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU(), nn.Linear(96, action_size)).to(self.device)
        self.target.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()
        self.steps = 0

    def act(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with self.torch.no_grad():
            x = self.torch.tensor(state, dtype=self.torch.float32, device=self.device).unsqueeze(0)
            return int(self.net(x).argmax(dim=1).item())

    def remember(self, *transition): self.memory.append(transition)

    def learn(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = map(np.asarray, zip(*batch))
        t = self.torch
        states = t.tensor(states, dtype=t.float32, device=self.device)
        next_states = t.tensor(next_states, dtype=t.float32, device=self.device)
        actions = t.tensor(actions, dtype=t.int64, device=self.device).unsqueeze(1)
        rewards = t.tensor(rewards, dtype=t.float32, device=self.device).unsqueeze(1)
        dones = t.tensor(dones, dtype=t.float32, device=self.device).unsqueeze(1)
        current = self.net(states).gather(1, actions)
        with t.no_grad():
            # Double DQN: separate action selection from target evaluation.
            next_actions = self.net(next_states).argmax(dim=1, keepdim=True)
            target = rewards + self.gamma * (1 - dones) * self.target(next_states).gather(1, next_actions)
        loss = self.loss(current, target)
        self.optimizer.zero_grad(); loss.backward()
        t.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.optimizer.step()
        self.steps += 1
        if self.steps % 250 == 0:
            self.target.load_state_dict(self.net.state_dict())
        return float(loss.item())

    def decay(self): self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
