import torch
import torch.nn as nn
import torch as T
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

# 用于处理雷达返回的点云数据
class PointNetFeatures(torch.nn.Module):
    def __init__(self, emb_dims=1024, input_shape="bnc", use_bn=False, global_feat=True):
        # emb_dims: 设定PointNet特征嵌入的维度（默认1024）。
        # input_shape: 输入点云的形状（b: 批次大小，n: 点的数量，c: 通道数）。
        # use_bn: 是否使用批量归一化。
        # global_feat: 是否只返回全局特征，若为False，返回每个点的局部特征。

        super(PointNetFeatures, self).__init__()

        # 检查输入数据形状是否合法
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' ")

        self.input_shape = input_shape  # 保存输入形状
        self.emb_dims = emb_dims  # 保存嵌入维度
        self.use_bn = use_bn  # 保存是否使用批量归一化
        self.global_feat = global_feat  # 保存是否只使用全局特征

        # 如果不使用全局特征，初始化最大池化操作
        if not self.global_feat:
            self.pooling = Pooling('max')

        # 创建网络结构（卷积层和激活层）
        self.layers = self.create_structure()

    def create_structure(self):
        # 定义PointNet网络的各层
        self.conv1 = torch.nn.Conv1d(3, 64, 1)  # 第1个卷积层：输入3个通道（x, y, z），输出64个通道
        self.conv2 = torch.nn.Conv1d(64, 64, 1)  # 第2个卷积层：输入64个通道，输出64个通道
        self.conv3 = torch.nn.Conv1d(64, 64, 1)  # 第3个卷积层：输入64个通道，输出64个通道
        self.conv4 = torch.nn.Conv1d(64, 128, 1)  # 第4个卷积层：输入64个通道，输出128个通道
        self.conv5 = torch.nn.Conv1d(128, self.emb_dims, 1)  # 第5个卷积层：输出embedding维度

        self.relu = torch.nn.ReLU()  # ReLU激活函数

        # 如果使用批量归一化，则为每个卷积层创建批量归一化层
        if self.use_bn:
            self.bn1 = torch.nn.BatchNorm1d(64)
            self.bn2 = torch.nn.BatchNorm1d(64)
            self.bn3 = torch.nn.BatchNorm1d(64)
            self.bn4 = torch.nn.BatchNorm1d(128)
            self.bn5 = torch.nn.BatchNorm1d(self.emb_dims)

        # 根据是否使用批量归一化构建网络层
        if self.use_bn:
            layers = [self.conv1, self.bn1, self.relu,
                      self.conv2, self.bn2, self.relu,
                      self.conv3, self.bn3, self.relu,
                      self.conv4, self.bn4, self.relu,
                      self.conv5, self.bn5, self.relu]
        else:
            layers = [self.conv1, self.relu,
                      self.conv2, self.relu,
                      self.conv3, self.relu,
                      self.conv4, self.relu,
                      self.conv5, self.relu]

        return layers

    def forward(self, input_data):
        # input_data: 输入的点云数据，形状为input_shape
        # output: 输出的PointNet特征，形状为 (Batch x emb_dims)

        # 如果输入数据是 'bnc' 形式，转置为 'bcn' 形式
        if self.input_shape == "bnc":
            num_points = input_data.shape[1]  # 点的数量
            input_data = input_data.permute(0, 2, 1)  # 转换维度（batch, channels, num_points）
        else:
            num_points = input_data.shape[2]  # 点的数量

        # 检查输入数据的通道数是否为3
        if input_data.shape[1] != 3:
            raise RuntimeError(
                "shape of x must be of [Batch x 3 x NumInPoints]")

        output = input_data
        for idx, layer in enumerate(self.layers):
            output = layer(output)  # 逐层进行前向传播

            # 如果不使用全局特征（global_feat=False），则保存每个点的特征
            if idx == 1 and not self.global_feat:
                point_feature = output

        # 如果需要全局特征（global_feat=True），返回全局特征
        if self.global_feat:
            return output
        else:
            # 如果不需要全局特征，进行最大池化，并返回每个点的局部特征
            output = self.pooling(output)
            output = output.view(-1, self.emb_dims, 1).repeat(1, 1, num_points)  # 复制嵌入特征
            return torch.cat([output, point_feature], 1)  # 将全局特征和局部特征拼接


class Pooling(torch.nn.Module):
    def __init__(self, pool_type='max'):
        # 初始化Pooling层
        # pool_type: 池化类型，支持 'max' (最大池化) 或 'avg'/'average' (平均池化)
        self.pool_type = pool_type
        super(Pooling, self).__init__()

    def forward(self, input):
        # input: 输入张量，通常是形状为 (batch_size, channels, num_points)
        if self.pool_type == 'max':
            # 使用最大池化，torch.max 返回一个元组，[0]表示最大值
            return torch.max(input, 2)[0].contiguous()  # 沿着第2维进行最大池化，并返回结果
        elif self.pool_type == 'avg' or self.pool_type == 'average':
            # 使用平均池化
            return torch.mean(input, 2).contiguous()  # 沿着第2维进行平均池化，并返回结果


class PointNetModel(nn.Module):
    def __init__(self, emb_dims, n_actions, input_shape="bnc", use_bn=False, global_feat=True):
        # emb_dims: 特征嵌入维度（PointNet的输出维度）
        # n_actions: 输出的类别数或动作数
        # input_shape: 输入数据的形状，通常为 "bnc"（batch, channels, num_points）
        # use_bn: 是否使用批量归一化
        # global_feat: 是否只提取全局特征

        super(PointNetModel, self).__init__()

        # 使用PointNetFeatures模型提取点云特征
        self.feature_model = PointNetFeatures(
            emb_dims=emb_dims, input_shape=input_shape, use_bn=use_bn, global_feat=global_feat)

        self.num_classes = n_actions  # 输出类别数或动作数

        # 线性层和批量归一化层，用于对提取的特征进行进一步处理
        self.linear1 = torch.nn.Linear(self.feature_model.emb_dims, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)  # 对第一层线性输出进行批量归一化
        self.dropout1 = torch.nn.Dropout(p=0.7)  # Dropout层，防止过拟合

        self.linear2 = torch.nn.Linear(512, 256)  # 第二个线性层
        self.bn2 = torch.nn.BatchNorm1d(256)  # 对第二层线性输出进行批量归一化
        self.dropout2 = torch.nn.Dropout(p=0.7)  # 第二个Dropout层

        self.linear3 = torch.nn.Linear(256, self.num_classes)  # 输出层，输出类别数或动作数

        # 池化操作（最大池化）
        self.pooling = Pooling('max')

        # 设置设备为GPU（如果可用）或CPU
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)  # 将模型移到指定的设备

    def forward(self, input_data):
        # input_data: 输入的点云数据，形状为 (batch_size, channels, num_points)

        # 使用特征提取模型得到的特征进行池化操作
        output = self.pooling(self.feature_model(input_data))

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
    # Test the code.
    x = torch.rand((1, 3000, 3))

    pn = PointNetModel(1024, 3)
    pn.eval()
    #classes = pn(x)
    
    print("Network Architecture: ")
    print(pn)
    print(pn.count_parameters())
    #print('Input Shape: {}\nClassification Output Shape: {}'.format(x.shape, classes.shape))
