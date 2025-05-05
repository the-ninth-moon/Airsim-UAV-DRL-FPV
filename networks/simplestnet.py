import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta


class SimplestMLP(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(SimplestMLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)  # 第一层全连接层
        self.fc2 = nn.Linear(256, 256)  # 第二层全连接层
        self.fc3 = nn.Linear(256, 128)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        # self.fc4 = nn.Linear(128, 64)
        self.out = nn.Linear(128, n_actions)  # 输出层

    def forward(self, input_data):
        # input_data: 输入的简单向量，形状为 (batch_size, 1, input_dim)

        # 将 (batch_size, 1, input_dim) 形状的输入变成 (batch_size, input_dim)
        input_data = input_data.squeeze(1)  # 删除第二维度（1），变成 (batch_size, input_dim)

        # 前向传播
        x = F.sigmoid(self.fc1(input_data))  # 第一层
        x = F.sigmoid(self.fc2(x))  # 第二层
        x = F.sigmoid(self.fc3(x))
        output = self.out(x)  # 输出层

        return output

class ValueStateNet(torch.nn.Module):
    def __init__(self, input_dim, n_actions):
        super(ValueStateNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # 第一层全连接层
        self.fc2 = nn.Linear(256, 256)  # 第二层全连接层
        self.fc3 = nn.Linear(256, 128)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        # self.fc4 = nn.Linear(128, 64)
        self.out = nn.Linear(128, n_actions)  # 输出层

    def forward(self, input_data):
        # input_data: 输入的简单向量，形状为 (batch_size, 1, input_dim)
        # 将 (batch_size, 1, input_dim) 形状的输入变成 (batch_size, input_dim)
        input_data = input_data.squeeze(1)  # 删除第二维度（1），变成 (batch_size, input_dim)
        # 前向传播
        x = F.sigmoid(self.fc1(input_data))  # 第一层
        x = F.sigmoid(self.fc2(x))  # 第二层
        x = F.sigmoid(self.fc3(x))
        output = self.out(x)  # 输出层
        return output

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, input_dim, n_actions):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # 第一层全连接层
        self.fc2 = nn.Linear(256, 256)  # 第二层全连接层
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        self.out = nn.Linear(256, n_actions)

    def forward(self, input_data):
        input_data = input_data.squeeze(1)  # 删除第二维度（1），变成 (batch_size, input_dim)
        # 前向传播
        x = F.tanh(self.fc1(input_data))  # 第一层
        x = F.tanh(self.fc2(x))  # 第二层
        return F.tanh(self.out(x))


if __name__ == '__main__':
    statedim = 10  # 假设输入向量的维度是100
    x = torch.rand((1, 1, statedim))  # 输入形状为 (1, 1, statedim)

    model = SimplestMLP(input_dim=statedim, n_actions=3)  # 输出3个类别
    model.eval()

    # 模型输出
    output = model(x)
    print("Model Output Shape: ", output.shape)  # 输出形状应该是 (batch_size, n_actions)
