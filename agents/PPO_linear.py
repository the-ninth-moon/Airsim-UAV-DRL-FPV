import torch
import torch.nn as nn
import numpy as np
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

# from environment.sim_env import action
from networks.simplestnet import PolicyNetContinuous, ValueStateNet

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
        self.next_states = []  # 存储环境的状态
        self.logprobs = []  # 存储每个动作的对数概率
        self.rewards = []  # 存储每个动作的奖励
        self.is_terminals = []  # 存储是否为终止状态的标志

    def clear(self):
        # 清空所有缓冲区数据
        del self.actions[:]  # 清空动作列表
        del self.states[:]  # 清空状态列表
        del self.next_states[:]  # 存储环境的状态
        del self.logprobs[:]  # 清空logprobs列表
        del self.rewards[:]  # 清空奖励列表
        del self.is_terminals[:]  # 清空终止状态标志列表


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space  # 是否为连续动作空间
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
        # self.normalization = Normalization(state_dim)

        # 初始化Actor-Critic模型，分别用于策略（actor）和价值（critic）估计
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.AdamW([  # 使用Adam优化器优化actor和critic
            {'params': self.policy.actor.parameters(), 'lr': lr_actor,'eps':1e-5},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic,'eps':1e-5}
        ])

        # 初始化旧的策略网络，用于计算PPO的目标函数和优势函数
        # self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # self.policy_old.load_state_dict(self.policy.state_dict())  # 复制当前策略到旧策略

        self.MseLoss = nn.MSELoss()  # 用于计算均方误差（用于critic的值估计）
    def add_memory(self, state, action, action_logprob,reward, next_state, done):
        # state = self.normalization(state)
        # next_state = self.normalization(next_state)
        state = torch.FloatTensor(np.array([state])).to(device)  # 转换状态为Tensor并转移到device
        next_state = torch.FloatTensor(np.array([next_state])).to(device)  # 转换状态为Tensor并转移到device
        reward = torch.tensor(np.array(reward),dtype=torch.float32).to(device)
        done = torch.tensor(np.array(done),dtype=torch.float32).to(device)
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.next_states.append(next_state)
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)
    def set_action_std(self, new_action_std):
        # 设置新的动作标准差（仅适用于连续动作空间）
        if self.has_continuous_action_space:
            self.action_std = new_action_std  # 更新动作标准差
            self.policy.set_action_std(new_action_std)  # 更新策略网络的标准差
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
        # state = self.normalization(state)
        # 如果是连续动作空间
        with torch.no_grad():
            # 使用旧的策略网络（policy_old）来选择动作，确保不计算梯度

            state = torch.FloatTensor(np.array([state])).to(device)  # 转换状态为Tensor并转移到device
            # print("191 state:",state.shape) #torch.Size([1,1, 10])
            action, action_logprob = self.policy.act(state)  # 使用策略网络选择动作和计算log概率
        return action,action_logprob
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
        # 将列表转为Tensor
        states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_log_probs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        rewards = torch.squeeze(torch.stack(self.buffer.rewards, dim=0)).detach().to(device)
        next_states = torch.squeeze(torch.stack(self.buffer.next_states, dim=0)).detach().to(device)
        dones = torch.squeeze(torch.stack(self.buffer.is_terminals, dim=0)).detach().to(device)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # print("state:",states.shape)
        # print("next_state:",next_states.shape)
        # print("rewards:",rewards.shape)
        # print("dones:",dones.shape)
        # print("actions:",actions.shape)

        vs = self.policy.critic(next_states)
        # print("vs:",vs.shape) #vs: torch.Size([10, 1])
        td_target = rewards + self.gamma * self.policy.critic(next_states) * (1 -dones)
        # print("td_target:",td_target.shape) #td_target: torch.Size([10, 10])
        td_delta = td_target - self.policy.critic(states)
        advantage = self.compute_advantage(self.gamma, self.lambdas,td_delta.cpu()).to(device)
        advantage = (advantage-advantage.mean())/(advantage.std()+1e-8)
        advantage = advantage.squeeze(1)
        # print("advantage:",advantage.shape)

        for _ in range(nums):
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            ratio = torch.exp(logprobs - old_log_probs.detach())  # 比率用于计算损失函数
            # print("ratio:",ratio.shape,logprobs.shape,old_log_probs.shape)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage  # 截断
            # print("surr",surr1.shape,surr2.shape,dist_entropy.shape)
            actor_loss = torch.mean(-torch.min(surr1, surr2))- 0.01 * dist_entropy # PPO损失函数
            critic_loss = torch.mean(self.MseLoss(self.policy.critic(states), td_target.detach()))
            # print("tdtarget",td_target.shape)
            # 进行梯度更新
            self.optimizer.zero_grad()  # 清除梯度
            actor_loss.mean().backward()  # 计算梯度
            critic_loss.mean().backward()
            self.optimizer.step()  # 更新模型参数
        # 清空buffer，准备下一次更新
        self.buffer.clear()
        torch.cuda.empty_cache()  # 清除GPU缓存

    def bc_clone(self,batch_size=64, lr=1e-3, num_epochs=10):
        """
        使用行为克隆训练actor网络
        :param actor: Actor网络
        :param bc_clone_list: 存储 (state, action) 对的列表
        :param batch_size: 每次训练的batch大小
        :param lr: 学习率
        :param num_epochs: 训练的轮数
        """

        # 加载保存的state和action数据
        data = np.load('state_action_data.npz')
        # 定义优化器
        optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=lr)

        # 将actor网络设置为训练模式
        self.policy.actor.train()

        # 将数据转换为tensor格式
        # 获取state和action数组
        states = data['states']
        actions = data['actions']
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)

        # print(states)

        print("states:",states.shape)
        print("actions:",actions.shape)

        # 开始训练
        for epoch in range(num_epochs):
            # 打乱数据
            perm = torch.randperm(states.size(0))
            states = states[perm]
            actions = actions[perm]

            # 按batch_size划分数据
            for i in range(0, states.size(0), batch_size):
                state_batch = states[i:i + batch_size]
                action_batch = actions[i:i + batch_size]

                # 前向传播：actor根据state生成动作
                predicted_actions = self.policy.actor(state_batch)

                # 计算损失：最小化预测动作和实际动作之间的L2损失
                loss = self.MseLoss(predicted_actions, action_batch)

                # 反向传播并更新参数
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 每个epoch输出一次损失
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))




if __name__ == '__main__':


    has_continuous_action_space = True  # 是否为连续动作空间

    # 设置训练的参数
    max_ep_len = 500  # 每个episode最大步长
    max_training_timesteps = 20000  # 最大训练步长数

    print_freq = 150  # 每隔多少timesteps打印一次平均奖励
    log_freq = max_ep_len * 2  # 每隔多少timesteps记录一次日志
    save_model_freq = 200  # 每隔多少timesteps保存一次模型
    action_std = 0.8  # 初始的动作分布的标准差（多元正态分布）
    action_std_decay_rate = 0.03  # 动作标准差衰减率（action_std -= action_std_decay_rate）
    min_action_std = 0.1  # 最小动作标准差（当标准差小于此值时停止衰减）
    action_std_decay_freq = 2000  # 衰减频率
    update_timestep = 200 * 1  # 每多少timesteps更新一次策略
    K_epochs = 30  # 每次更新PPO策略时优化K轮
    eps_clip = 0.2  # PPO中的裁剪参数
    gamma = 0.95  # 折扣因子（对于连续动作空间较高）
    lr_actor = 3e-4  # actor网络的学习率
    lr_critic = 3e-4  # critic网络的学习率
    random_seed = 47  # 如果需要随机种子，设置为非零值
    # 获取状态空间和动作空间的维度
    state_dim = 6
    action_dim = 3  # 对于连续空间，动作空间的维度
    run_num_pretrained = 1 # 修改此值以防止覆盖权重文件，仅仅用于作为保存模型的参数
    ppo_agent = PPO_linear(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                           has_continuous_action_space, action_std)

    # 设置模型保存目录
    directory = "models"

    # checkpoint_path = directory + "PPO_{}_{}_{}.pth".format("DRL", random_seed, run_num_pretrained)
    ppo_agent.bc_clone(batch_size=64, lr=1e-3, num_epochs=50)

    ppo_agent.save(checkpoint_path='./ppo_bc_clone.pth')
