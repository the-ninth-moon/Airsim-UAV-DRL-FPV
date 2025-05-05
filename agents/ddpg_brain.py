import os
from collections import deque
import random

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Union

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound=10):
        super(ActorNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound

class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        x = self.activation(self.fc1(x))
        x= self.activation(self.fc2(x))
        return self.out(x)

class DDPGAgent:
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int,
                 action_dim: int,
                 action_bound: int = 10,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-3,
                 gamma: float = 0.98,
                 sigma:float = 0.01,
                 tau: float = 0.005,
                 batch_size: int = 64,
                 minimal_size:int = 2,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        DDPG智能体

        Args:
            state_dim (int): 状态维度
            hidden_dim (int): 隐藏层维度
            action_dim (int): 动作维度
            actor_lr (float): Actor学习率
            critic_lr (float): Critic学习率
            gamma (float): 折扣因子
            tau (float): 软更新系数
            batch_size (int): 每批训练的样本数量
            device (str): 计算设备
        """
        self.device = torch.device(device)

        # 网络初始化
        self.actor = ActorNetwork(state_dim, hidden_dim, action_dim,action_bound).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_actor = ActorNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # 将目标网络初始化为与训练网络相同
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 超参数
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.minimal_size = minimal_size

        # 经验回放缓存
        self.replay_buffer = ReplayMemory()

        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.action_dim = action_dim
    def add_memory(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
    def get_memory_size(self):
        return len(self.replay_buffer.buffer)
    def choose_action(self, state: np.ndarray) -> np.ndarray:
        self.actor.eval()
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).cpu().detach().numpy().flatten()
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def learn(self):
        if self.get_memory_size() > self.minimal_size:
            b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample(min(self.batch_size, self.get_memory_size()))
            transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
            self.update(transition_dict)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # print(states.shape)
        # print(actions.shape)
        # print(rewards.shape)
        # print(next_states.shape)
        # print(dones.shape)

        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # print("actior loss",actor_loss)

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, path: str = './model/ddpg.pth'):
        """
        保存模型

        Args:
            path (str): 保存路径（包括文件名）
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str = './model/ddpg.pth'):
        """
        加载模型

        Args:
            path (str): 加载路径（包括文件名）
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")

        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
