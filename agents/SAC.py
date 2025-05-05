import os
import random
import numpy as np
import torch
from copy import deepcopy
from torch import nn, optim
from torch.distributions import Normal
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from agents.prioritized_memory import Memory


class Actor(nn.Module):
    def __init__(self, cfg):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states, cfg.actor_hidden_dim)
        self.fc2 = nn.Linear(cfg.actor_hidden_dim, cfg.actor_hidden_dim)
        self.fc_mu = nn.Linear(cfg.actor_hidden_dim, cfg.n_actions)
        self.fc_std = nn.Linear(cfg.actor_hidden_dim, cfg.n_actions)
        self.action_bound = torch.tensor(cfg.action_bound, dtype=torch.float32, device=cfg.device)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        mu = self.fc_mu(x)
        std = F.softmax(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob -= torch.log(1 - torch.tanh(action).pow(2) + 1e-6)
        action = action * self.action_bound
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(cfg.n_states + cfg.n_actions, cfg.critic_hidden_dim)
        self.fc2 = nn.Linear(cfg.critic_hidden_dim, cfg.critic_hidden_dim)
        self.fc3 = nn.Linear(cfg.critic_hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        q = F.sigmoid(self.fc1(cat))
        q = F.sigmoid(self.fc2(q))
        q = self.fc3(q)
        return q


class SAC:
    def __init__(self,action_dim,state_dim,action_bound):
        # SAC 相关的配置参数
        self.min_sample_size = 16
        self.train_eps = 500
        self.test_eps = 10
        self.batch_size = 512
        self.memory_capacity = 1000
        self.lr_a = 1e-4
        self.lr_c = 1e-4
        self.lr_alpha = 5e-4
        self.gamma = 0.98
        self.tau = 0.005
        self.seed = random.randint(0, 100)
        self.actor_hidden_dim = 256
        self.critic_hidden_dim = 256
        self.n_states = state_dim
        self.n_actions = action_dim
        self.action_bound = action_bound
        self.target_entropy = -action_dim
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # 初始化其他必要的对象
        self.memory = [Memory() for _ in range(30)]
        self.all_memory = Memory()

        self.actor = Actor(self).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr_a)

        self.critic1 = torch.jit.script(Critic(self).to(self.device))
        self.critic1_target = deepcopy(self.critic1)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=self.lr_c)

        self.critic2 = torch.jit.script(Critic(self).to(self.device))
        self.critic2_target = deepcopy(self.critic2)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=self.lr_c)

        self.log_alpha = torch.tensor(np.log(0.01), requires_grad=True, device=self.device, dtype=torch.float32)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr_alpha)
        self.scaler = GradScaler()

        # self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optim, step_size=3, gamma=0.98)
        # self.critic1_scheduler = torch.optim.lr_scheduler.StepLR(self.critic1_optim, step_size=3, gamma=0.98)
        # self.critic2_scheduler = torch.optim.lr_scheduler.StepLR(self.critic2_optim, step_size=3, gamma=0.98)

    def show(self):
        print('-' * 30 + '参数列表' + '-' * 30)
        for k, v in vars(self).items():
            print(k, '=', v)
        print('-' * 60)

    def append_sample(self,next_gate, state, action, reward, next_state, done):
        with torch.no_grad():
            next_state2 = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            target_q = self.calc_target_q(reward, next_state2, done) # torch.Size([1, 1])
            state2 = torch.tensor(state, dtype=torch.float32, device=self.device) #1 13
            action2 = torch.tensor(np.array([action]), dtype=torch.float32, device=self.device)
            # print(state2.shape,action2.shape)
            old_q = min(self.critic1(state2, action2),self.critic2(state2, action2))
            # print(old_q,target_q)
            error = abs(old_q-target_q).detach().cpu().numpy().flatten()[0]
            # print("error",error)
            with open("error.txt","a") as f:
                f.write(f"error:{error},state:{state}\n")
            self.memory[next_gate].add(error, (state, action, reward, next_state, done))
            self.all_memory.add(error, (state, action, reward, next_state, done))
    @torch.no_grad()
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor(state)
        return action.detach().cpu().numpy().flatten()

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def calc_target_q(self, reward, next_state, done):
        next_action, next_log_prob = self.actor(next_state)
        # 对 log_prob 进行求和，得到每个样本的总 log_prob
        next_log_prob = next_log_prob.sum(dim=1, keepdim=True)  # [batch_size, 1]
        next_q1 = self.critic1_target(next_state, next_action)
        next_q2 = self.critic2_target(next_state, next_action)
        next_q = torch.min(next_q1, next_q2) - self.log_alpha.exp() * next_log_prob
        target_q = reward + self.gamma * (1 - done) * next_q
        return target_q

    def update(self,index):
        if self.memory[index].size() < self.min_sample_size:
            print(f"not enough memory to sample {self.memory[index].size()}")
            return 0, 0, 0, 0
        print("training:",index)
        mini_batch, idxs, is_weights = self.memory[index].sample(min(self.batch_size,self.memory[index].size()))

        state = torch.tensor(np.array(mini_batch[0]),dtype=torch.float32).to(self.device)
        action = torch.tensor(np.array(mini_batch[1]),dtype=torch.float32).to(self.device)
        reward = torch.tensor(np.array(mini_batch[2]),dtype=torch.float32).to(self.device)
        next_state =torch.tensor(np.array(mini_batch[3]),dtype=torch.float32).to(self.device)
        done = torch.tensor(np.array(mini_batch[4]),dtype=torch.int).to(self.device)
        reward, done = reward.view(-1, 1), done.view(-1, 1)
        # bool to binary
        is_weights = torch.tensor(is_weights,dtype=torch.float32).to(self.device).unsqueeze(0)
        state = state.squeeze(1)
        next_state = next_state.squeeze(1)

        # print("state:",state.shape) #torch.Size([16, 13])
        # print("action:",action.shape)#torch.Size([16, 2])
        # print("reward:",reward.shape)#torch.Size([16, 1])
        # print("next_state:",next_state.shape)#torch.Size([16, 13])
        # print("dones:",done.shape)#torch.Size([16, 1])

        with torch.amp.autocast("cuda",enabled=True):
            target_q = self.calc_target_q(reward, next_state, done)
            q1 = self.critic1(state, action)
            q2 = self.critic2(state, action)
            # print("target q:",target_q.shape) #torch.Size([16, 1])
            # print("q1:",q1.shape)#torch.Size([16, 1])
            # print("q2:",q2.shape)#torch.Size([16, 1])
            critic1_loss = torch.mean(F.mse_loss(q1, target_q.detach())*is_weights)
            critic2_loss = torch.mean(F.mse_loss(q2, target_q.detach())*is_weights)


        self.critic1_optim.zero_grad()
        self.scaler.scale(critic1_loss).backward()
        self.scaler.step(self.critic1_optim)

        self.critic2_optim.zero_grad()
        self.scaler.scale(critic2_loss).backward()
        self.scaler.step(self.critic2_optim)

        with torch.amp.autocast("cuda",enabled=True):
            new_action, log_prob = self.actor(state)
            log_prob = log_prob.sum(dim=1, keepdim=True)  # [batch_size, 1]
            q = torch.min(self.critic1(state, new_action), self.critic2(state, new_action)) #torch.Size([16, 1])
            actor_loss = ((self.log_alpha.exp() * log_prob - q)*is_weights).mean()

        self.actor_optim.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optim)

        with torch.amp.autocast("cuda",enabled=True):
            alpha_loss = torch.mean(self.log_alpha * ((-log_prob - self.target_entropy)*is_weights).detach())

        self.alpha_optim.zero_grad()
        self.scaler.scale(alpha_loss).backward()
        self.scaler.step(self.alpha_optim)

        self.scaler.update()
        self.soft_update(self.critic1_target, self.critic1)
        self.soft_update(self.critic2_target, self.critic2)

        errors1 = torch.abs(q1-target_q).detach().cpu().numpy().flatten()
        errors2 = torch.abs(q2-target_q).detach().cpu().numpy().flatten()
        errors = np.minimum(errors1, errors2)
        # update priority
        for i in range(min(self.batch_size,self.memory[index].size())):
            idx = idxs[i]
            self.memory[index].update(idx, errors[i])

        # self.critic1_optim.step()
        # self.critic2_optim.step()
        # self.actor_optim.step()
        #
        # self.actor_scheduler.step()
        # self.critic1_scheduler.step()
        # self.critic2_scheduler.step()
    def mini_update(self,index):
        if self.all_memory.size() < self.min_sample_size:
            print(f"not enough memory to sample {self.all_memory.size()}")
            return 0, 0, 0, 0
        mini_batch, idxs, is_weights = self.all_memory.sample(min(32,self.all_memory.size()))

        state = torch.tensor(np.array(mini_batch[0]),dtype=torch.float32).to(self.device)
        action = torch.tensor(np.array(mini_batch[1]),dtype=torch.float32).to(self.device)
        reward = torch.tensor(np.array(mini_batch[2]),dtype=torch.float32).to(self.device)
        next_state =torch.tensor(np.array(mini_batch[3]),dtype=torch.float32).to(self.device)
        done = torch.tensor(np.array(mini_batch[4]),dtype=torch.int).to(self.device)
        reward, done = reward.view(-1, 1), done.view(-1, 1)
        # bool to binary
        is_weights = torch.tensor(is_weights,dtype=torch.float32).to(self.device).unsqueeze(0)
        state = state.squeeze(1)
        next_state = next_state.squeeze(1)

        # print("state:",state.shape) #torch.Size([16, 13])
        # print("action:",action.shape)#torch.Size([16, 2])
        # print("reward:",reward.shape)#torch.Size([16, 1])
        # print("next_state:",next_state.shape)#torch.Size([16, 13])
        # print("dones:",done.shape)#torch.Size([16, 1])

        with torch.amp.autocast("cuda",enabled=True):
            target_q = self.calc_target_q(reward, next_state, done)
            q1 = self.critic1(state, action)
            q2 = self.critic2(state, action)
            # print("target q:",target_q.shape) #torch.Size([16, 1])
            # print("q1:",q1.shape)#torch.Size([16, 1])
            # print("q2:",q2.shape)#torch.Size([16, 1])
            critic1_loss = torch.mean(F.mse_loss(q1, target_q.detach())*is_weights)
            critic2_loss = torch.mean(F.mse_loss(q2, target_q.detach())*is_weights)


        self.critic1_optim.zero_grad()
        self.scaler.scale(critic1_loss).backward()
        self.scaler.step(self.critic1_optim)

        self.critic2_optim.zero_grad()
        self.scaler.scale(critic2_loss).backward()
        self.scaler.step(self.critic2_optim)

        with torch.amp.autocast("cuda",enabled=True):
            new_action, log_prob = self.actor(state)
            log_prob = log_prob.sum(dim=1, keepdim=True)  # [batch_size, 1]
            q = torch.min(self.critic1(state, new_action), self.critic2(state, new_action)) #torch.Size([16, 1])
            actor_loss = ((self.log_alpha.exp() * log_prob - q)*is_weights).mean()

        self.actor_optim.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optim)

        with torch.amp.autocast("cuda",enabled=True):
            alpha_loss = torch.mean(self.log_alpha * ((-log_prob - self.target_entropy)*is_weights).detach())

        self.alpha_optim.zero_grad()
        self.scaler.scale(alpha_loss).backward()
        self.scaler.step(self.alpha_optim)

        self.scaler.update()
        self.soft_update(self.critic1_target, self.critic1)
        self.soft_update(self.critic2_target, self.critic2)

        errors1 = torch.abs(q1-target_q).detach().cpu().numpy().flatten()
        errors2 = torch.abs(q2-target_q).detach().cpu().numpy().flatten()
        errors = np.minimum(errors1, errors2)
        # update priority
        for i in range(min(128,self.all_memory.size())):
            idx = idxs[i]
            self.all_memory.update(idx, errors[i])
    def save(self, checkpoint_path):
        # 创建一个字典来保存所有需要保存的对象
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic1_optim': self.critic1_optim.state_dict(),
            'critic2_optim': self.critic2_optim.state_dict(),
            'alpha_optim': self.alpha_optim.state_dict(),
            'scaler': self.scaler.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
        }
        # 保存字典到一个文件
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


    def load(self, checkpoint_path):
        # 加载整个字典
        checkpoint = torch.load(checkpoint_path)
        # 加载模型和优化器的状态
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        self.critic1_optim.load_state_dict(checkpoint['critic1_optim'])
        self.critic2_optim.load_state_dict(checkpoint['critic2_optim'])
        self.alpha_optim.load_state_dict(checkpoint['alpha_optim'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])

        print(f"Checkpoint loaded from {checkpoint_path}")

if __name__ == '__main__':
    statedim = 10  # 假设输入向量的维度是100
    x = torch.rand((statedim))  # 输入形状为 (1, 1, statedim)

    model = SAC(2,10,5)  # 输出3个类别

    # print("x:",x.shape) #32 10

    # 模型输出
    output = model.select_action(x)
    print("Model Output Shape: ", output.shape)  # 输出形状应该是 (batch_size, n_actions)
