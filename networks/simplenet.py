import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 用于处理简单的向量输入 (修改后的版本)
class SimpleFeatureExtractor(nn.Module):
    def __init__(self, input_dim, emb_dims=1024, use_bn=False):
        super(SimpleFeatureExtractor, self).__init__()

        self.input_dim = input_dim  # 输入维度
        self.emb_dims = emb_dims  # 特征嵌入维度
        self.use_bn = use_bn  # 是否使用批量归一化

        # 创建网络结构（线性层和激活层）
        self.layers = self.create_structure()

    def create_structure(self):
        # 定义特征提取网络的各层
        self.fc1 = nn.Linear(self.input_dim, 512)  # 第1个全连接层
        self.fc2 = nn.Linear(512, 256)  # 第2个全连接层
        self.fc3 = nn.Linear(256, self.emb_dims)  # 第3个全连接层，用于输出嵌入特征

        self.relu = nn.ReLU()  # ReLU激活函数

        # 如果使用批量归一化，则为每个全连接层创建批量归一化层
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(self.emb_dims)

        # 根据是否使用批量归一化构建网络层
        if self.use_bn:
            layers = [self.fc1, self.bn1, self.relu,
                      self.fc2, self.bn2, self.relu,
                      self.fc3, self.bn3, self.relu]
        else:
            layers = [self.fc1, self.relu,
                      self.fc2, self.relu,
                      self.fc3, self.relu]

        return layers

    def forward(self, input_data):
        # input_data: 输入的简单向量，形状为 (batch_size, 1, input_dim)

        # 将 (batch_size, 1, input_dim) 形状的输入变成 (batch_size, input_dim)
        input_data = input_data.squeeze(1)  # 删除第二维度（1），变成 (batch_size, input_dim)

        output = input_data
        for idx, layer in enumerate(self.layers):
            output = layer(output)  # 逐层进行前向传播

        return output


class SimpleModel(nn.Module):
    def __init__(self, input_dim, emb_dims, n_actions, use_bn=False):
        super(SimpleModel, self).__init__()

        # 使用SimpleFeatureExtractor模型提取特征
        self.feature_model = SimpleFeatureExtractor(
            input_dim=input_dim, emb_dims=emb_dims, use_bn=use_bn)

        self.num_classes = n_actions  # 输出类别数或动作数

        # 线性层和批量归一化层，用于对提取的特征进行进一步处理
        self.linear1 = nn.Linear(self.feature_model.emb_dims, 512)
        self.bn1 = nn.BatchNorm1d(512)  # 对第一层线性输出进行批量归一化
        self.dropout1 = nn.Dropout(p=0.7)  # Dropout层，防止过拟合

        self.linear2 = nn.Linear(512, 256)  # 第二个线性层
        self.bn2 = nn.BatchNorm1d(256)  # 对第二层线性输出进行批量归一化
        self.dropout2 = nn.Dropout(p=0.7)  # 第二个Dropout层

        self.linear3 = nn.Linear(256, self.num_classes)  # 输出层，输出类别数或动作数

        # 设置设备为GPU（如果可用）或CPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # 将模型移到指定的设备

    def forward(self, input_data):
        # input_data: 输入的简单向量，形状为 (batch_size, 1, input_dim)

        # 确保输入数据与模型在同一设备上
        input_data = input_data.to(self.device)

        # 使用特征提取模型提取特征
        output = self.feature_model(input_data)

        # 通过全连接层进行特征映射，带有激活函数、批量归一化和Dropout
        output = F.relu(self.bn1(self.linear1(output)))  # 第一个全连接层，ReLU激活
        output = self.dropout1(output)  # 第一个Dropout层

        output = F.relu(self.bn2(self.linear2(output)))  # 第二个全连接层，ReLU激活
        output = self.dropout2(output)  # 第二个Dropout层

        output = self.linear3(output)  # 最后一层全连接层，输出动作/类别

        return output

    def count_parameters(self):
        # 计算模型的可训练参数总数
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the code with a simple input vector of shape (1, 1, statedim)
    statedim = 100  # 假设输入向量的维度是100
    x = torch.rand((1, 1, statedim))  # 输入形状为 (1, 1, statedim)

    model = SimpleModel(input_dim=statedim, emb_dims=1024, n_actions=3)  # 输出3个类别
    model.eval()

    # # 输出网络结构和参数数量
    # print("Network Architecture: ")
    # print(model)
    # print("Total Parameters: ", model.count_parameters())

    # 模型输出
    output = model(x)
    print("Model Output Shape: ", output.shape)  # 输出形状应该是 (batch_size, n_actions)
