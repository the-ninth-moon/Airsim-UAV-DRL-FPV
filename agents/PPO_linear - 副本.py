import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from networks.pointnet import PointNetModel
# from networks.simplenet import SimpleModel
from networks.simplestnet import SimplestMLP, PolicyNetContinuous, ValueStateNet

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        # 初始化缓冲区，保存训练过程中采集的各类数据
        self.actions = []  # 存储执行的动作
        self.states = []  # 存储环境的状态
        self.logprobs = []  # 存储每个动作的对数概率
        self.rewards = []  # 存储每个动作的奖励
        self.is_terminals = []  # 存储是否为终止状态的标志

    def clear(self):
        # 清空所有缓冲区数据
        del self.actions[:]  # 清空动作列表
        del self.states[:]  # 清空状态列表
        del self.logprobs[:]  # 清空logprobs列表
        del self.rewards[:]  # 清空奖励列表
        del self.is_terminals[:]  # 清空终止状态标志列表


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space  # 是否为连续动作空间

        if has_continuous_action_space:
            self.action_dim = action_dim  # 连续动作的维度
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)  # 动作的方差，初始化为一个常数
            # self.action_var: torch.Size([2])
        # actor部分：决定智能体的动作
        self.actor = PolicyNetContinuous(state_dim, action_dim)

        # critic部分：估计状态值
        self.critic = ValueStateNet(state_dim, 1)  # 使用PointNetModel作为值网络，输出1个值

        self.actor.eval()  # 将actor设置为评估模式（不进行梯度更新）
        self.critic.eval()  # 将critic设置为评估模式（不进行梯度更新）

    def set_action_std(self, new_action_std):
        # 设置动作空间的标准差
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)  # 更新方差
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        # 前向传播函数，未实现，因为策略和价值网络有不同的行为
        raise NotImplementedError

    def act(self, state):
        # 选择一个动作
        if self.has_continuous_action_space:
            action_mean = self.actor(state)  # 获取动作均值
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)  # 计算协方差矩阵
            dist = MultivariateNormal(action_mean, cov_mat)  # 使用多元正态分布建模连续动作空间
        else:
            action_probs = self.actor(state)  # 获取每个动作的概率
            dist = Categorical(action_probs)  # 使用Categorical分布建模离散动作空间

        action = dist.sample()  # 从分布中采样一个动作
        action_logprob = dist.log_prob(action)  # 计算该动作的对数概率

        # 返回动作及其对数概率（用于策略优化）
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        # 评估一个动作，计算它的对数概率、状态值和熵
        if self.has_continuous_action_space:
            # print("95 state:",state.shape) #[batchsize,1,10]
            action_mean = self.actor(state)  # 获取动作均值
            action_var = self.action_var.expand_as(action_mean)  # 获取动作方差
            cov_mat = torch.diag_embed(action_var).to(device)  # 计算协方差矩阵
            dist = MultivariateNormal(action_mean, cov_mat)  # 使用多元正态分布

            # 如果是单个动作（动作维度为1），reshape为合适的形状
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)  # 获取动作概率
            dist = Categorical(action_probs)  # 使用Categorical分布处理离散动作

        # 计算动作的对数概率和分布的熵
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # 计算状态的价值
        state_values = self.critic(state)

        # 返回动作的对数概率、状态值和熵
        return action_logprobs, state_values, dist_entropy

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n )
class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

class PPO_linear:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, action_std_init=0.6,lambdas=0.95):
        # state_dim: 状态空间的维度
        # action_dim: 动作空间的维度
        # lr_actor: actor网络的学习率
        # lr_critic: critic网络的学习率
        # gamma: 折扣因子（用于计算累积奖励）
        # K_epochs: PPO的训练迭代次数（每次更新时重复的次数）
        # eps_clip: PPO的裁剪范围，防止策略更新过大
        # has_continuous_action_space: 是否为连续动作空间
        # action_std_init: 初始化动作标准差（仅用于连续动作空间）

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init  # 初始化动作标准差（控制连续动作空间的动作幅度）

        self.gamma = gamma  # 奖励的折扣因子
        self.eps_clip = eps_clip  # PPO的裁剪范围
        self.K_epochs = K_epochs  # 训练时每次更新的迭代次数
        self.lambdas = lambdas

        self.buffer = RolloutBuffer()  # 初始化RolloutBuffer，保存经历的轨迹数据
        self.normalization = Normalization(state_dim)

        # 初始化Actor-Critic模型，分别用于策略（actor）和价值（critic）估计
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.AdamW([  # 使用Adam优化器优化actor和critic
            {'params': self.policy.actor.parameters(), 'lr': lr_actor,'eps':1e-5},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic,'eps':1e-5}
        ])

        # 初始化旧的策略网络，用于计算PPO的目标函数和优势函数
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())  # 复制当前策略到旧策略

        self.MseLoss = nn.MSELoss()  # 用于计算均方误差（用于critic的值估计）

    def set_action_std(self, new_action_std):
        # 设置新的动作标准差（仅适用于连续动作空间）
        if self.has_continuous_action_space:
            self.action_std = new_action_std  # 更新动作标准差
            self.policy.set_action_std(new_action_std)  # 更新策略网络的标准差
            self.policy_old.set_action_std(new_action_std)  # 更新旧策略网络的标准差
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")  # 对离散动作空间调用该方法时的警告
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        # 逐步衰减动作标准差（用于控制策略在训练过程中的探索程度）
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate  # 按照衰减率更新动作标准差
            self.action_std = round(self.action_std, 4)  # 保证标准差为四位小数
            if (self.action_std <= min_action_std):  # 如果标准差小于最小值，则设置为最小值
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)  # 更新策略网络的标准差
        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")  # 对离散动作空间调用该方法时的警告
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        # state: 当前环境的状态，通常是一个向量或张量
        state = self.normalization(state)
        if self.has_continuous_action_space:
            # 如果是连续动作空间
            with torch.no_grad():
                # 使用旧的策略网络（policy_old）来选择动作，确保不计算梯度
                state = torch.FloatTensor(np.array([state])).to(device)  # 转换状态为Tensor并转移到device
                # print("191 state:",state.shape) #torch.Size([1,1, 10])
                action, action_logprob = self.policy_old.act(state)  # 使用策略网络选择动作和计算log概率
            # 将状态、动作和log概率存入buffer中
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            # 返回动作并将其从GPU移至CPU
            return action.detach().cpu().numpy().flatten()
        else:
            # 如果是离散动作空间
            with torch.no_grad():
                state = torch.FloatTensor(np.array([state])).to(device)  # 转换状态为Tensor并转移到device
                action, action_logprob = self.policy_old.act(state)  # 使用策略网络选择动作和计算log概率

            # 将状态、动作和log概率存入buffer中
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            # 返回选择的离散动作
            return action.item()
    def compute_advantage(self,gamma, lmbda, td_delta):
        td_delta = td_delta.cpu().detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.as_tensor(np.array(advantage_list), dtype=torch.float32)
    def update(self,nums):
        # 计算回报的蒙特卡洛估计
        rewards = []
        discounted_reward = 0
        # 遍历回报和终止标志，从后往前计算折扣奖励
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0  # 终止时，折扣奖励为0
            discounted_reward = reward + (self.gamma * discounted_reward)  # 使用折扣因子更新奖励
            rewards.insert(0, discounted_reward)  # 插入计算的折扣奖励到rewards列表的前面

        # 对奖励进行标准化处理
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)  # 转换为Tensor
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)  # 标准化（减去均值，除以标准差）

        # 将列表转为Tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_states=old_states.unsqueeze(1)
        # print("old_states:", old_states.shape) #torch.Size([batchsize, 1,10])
        # print("old_actions:", old_actions.shape) # torch.Size([batchsize, 2])
        # print("old_logprobs:", old_logprobs.shape)# torch.Size([batchsize])

        # 对策略进行K次优化
        for _ in range(nums):
            # 评估旧的策略（通过计算log概率、状态价值和熵）
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # print("logprobs: ", logprobs.shape) # torch.Size([batchsize])
            # print("state_values: ", state_values.shape)# torch.Size([batchsize,1])
            # print("dist_entropy: ", dist_entropy.shape)# torch.Size([batchsize])
            # print("reward: ",rewards.shape)# torch.Size([batchsize])

            # 将状态值张量的维度调整为与奖励张量相同
            state_values = torch.squeeze(state_values)
            # print("state_values2:",state_values.shape) #torch.Size([batchsize])

            # 计算比率 (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())  # 比率用于计算损失函数

            # print("ratio:",ratios.shape) #ratio: torch.Size([batchsize])
            # 计算PPO的目标函数中的Surrogate Loss
            advantages = rewards - state_values.detach()  # 优势函数 = 奖励 - 状态值
            # 计算均值和标准差
            advantage_mean = advantages.mean()
            advantage_std = advantages.std()
            # 进行归一化操作
            advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)  # 加一个小的常数以避免除零
            # print("advantages:", advantages.shape)advantages: torch.Size([batchsize])
            surr1 = ratios * advantages  # 第一种损失（不带裁剪）
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # 第二种损失（带裁剪）

            # 最终的PPO损失（包含Surrogate Loss、价值损失和熵惩罚）
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # 进行梯度更新
            self.optimizer.zero_grad()  # 清除梯度
            loss.mean().backward()  # 计算梯度
            self.optimizer.step()  # 更新模型参数

        # 将新的策略权重拷贝到旧策略中
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空buffer，准备下一次更新
        self.buffer.clear()
        torch.cuda.empty_cache()  # 清除GPU缓存


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))




