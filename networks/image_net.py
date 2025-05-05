import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueStateNet(torch.nn.Module):
    def __init__(self, input_channels=1):
        super(ValueStateNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),  # 输出: (32, 35, 63)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 输出: (64, 16, 30)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 输出: (64, 14, 28)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6))  # 输出: (64, 6, 6)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 6 * 6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出状态价值
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, input_data):
        # 输入形状: (batch_size, input_channels, height, width)
        x = self.conv_layers(input_data)
        x = x.view(x.size(0), -1)  # 展平
        return self.fc_layers(x)

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, input_channels=1, n_actions=2):
        super(PolicyNetContinuous, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 6 * 6, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.mean_layer = nn.Linear(256, n_actions)
        self.log_std = nn.Parameter(torch.zeros(n_actions))  # 可训练的对数标准差参数
        self._init_weights()

    def _init_weights(self):
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight)
        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        nn.init.xavier_normal_(self.mean_layer.weight)

    def forward(self, input_data):
        # 输入形状: (batch_size, input_channels, height, width)
        x = self.conv_layers(input_data)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        mean = self.mean_layer(x)
        # log_std = self.log_std.expand_as(mean)  # 扩展为相同形状
        return mean

import torch

# 假设 ValueStateNet 和 PolicyNetContinuous 已经定义在当前文件中

def main():
    # 构造一个测试输入，单个样本、单通道，尺寸 (144, 256)
    dummy_state = torch.randn(1, 1, 144, 256)

    # 实例化网络（注意：如果需要修改输入通道数或动作数，可在此调整参数）
    value_net = ValueStateNet(input_channels=1)
    policy_net = PolicyNetContinuous(input_channels=1, n_actions=2)

    # 前向传播得到网络输出
    value_output = value_net(dummy_state)
    policy_mean, policy_log_std = policy_net(dummy_state)

    # 打印输出结果
    print("ValueStateNet 输出形状:", value_output.shape)
    print("ValueStateNet 输出:", value_output)
    print("-" * 50)
    print("PolicyNetContinuous mean 输出形状:", policy_mean.shape)
    print("PolicyNetContinuous mean 输出:", policy_mean)
    print("PolicyNetContinuous log_std 输出形状:", policy_log_std.shape)
    print("PolicyNetContinuous log_std 输出:", policy_log_std)

if __name__ == "__main__":
    main()
