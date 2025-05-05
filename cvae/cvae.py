import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2  # 导入 cv2
from torch.nn import functional as F
from torchvision.utils import save_image
from PIL import Image  # 仍然需要 PIL 用于保存图像

# 超参数配置
class Config:
    latent_dim = 64         # 潜在空间维度
    condition_dim = 4       # 根据文件名中参数数量调整
    batch_size = 64          # 提高batch_size加速训练
    lr = 3e-4                # 更小的学习率
    epochs = 201
    save_interval = 20
    test_interval = 20       # 每隔多少epoch进行测试和生成图像
    num_test_images = 20     # 测试集图片数量
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_size = (240, 320)    # 图像尺寸 (H, W) # 修改后的图像尺寸
    channels = 1             # 输入图像通道数 (根据实际数据调整) # 修改为1通道
    output_dir = 'cvae_output' # 输出文件夹

    os.makedirs(output_dir, exist_ok=True)

# 数据集类
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, split='train', test_size=Config.num_test_images):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

        # 验证文件名参数数量
        # if self.img_files:
        #     test_name = self.img_files[0].replace('.jpg', '').split()
        #     assert len(test_name) == Config.condition_dim, \
        #         f"文件名参数数量({len(test_name)})与配置({Config.condition_dim})不符"

        # 划分训练集和测试集
        if split == 'test':
            self.img_files = self.img_files[:test_size]
        else: # split == 'train'
            self.img_files = self.img_files[test_size:]


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img_np = cv2.imread(img_path) # 使用 cv2 读取图像，得到 numpy array
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) # 转换为灰度图

        # 提取条件参数
        filename = self.img_files[idx].replace('.jpg', '')
        conditions = torch.tensor([float(x) for x in filename.split()[:4]], dtype=torch.float32)
        # print(conditions)

        # 数据增强
        if self.transform:
            img_tensor = self.transform(Image.fromarray(img_np)) # 先将 numpy array 转为 PIL Image，再进行 transform
        else:
            img_tensor = transforms.ToTensor()(Image.fromarray(img_np)) # 默认转换为 Tensor，并确保通道数为1


        return img_tensor, conditions

# 改进的Encoder (修改网络参数以适应新的图像尺寸)
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 图像特征提取
        self.conv_stack = nn.Sequential(
            nn.Conv2d(Config.channels, 64, 4, 2, 1),  # 输入通道数修改为 Config.channels, 240x320 -> 120x160
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1), # 120x160 -> 60x80
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1), # 60x80 -> 30x40
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1), # 30x40 -> 15x20  # Add one more conv layer
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        # 条件融合
        self.condition_fc = nn.Linear(Config.condition_dim, 15*20) # 修改：适应 15x20

        # 潜在空间预测
        self.fc = nn.Sequential(
            nn.Linear(512*15*20 + 15*20, 1024), # 修改：适应 512*15*20 + 15*20
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.fc_mu = nn.Linear(1024, Config.latent_dim)
        self.fc_logvar = nn.Linear(1024, Config.latent_dim)

    def forward(self, x, conditions):
        # 提取图像特征
        img_feat = self.conv_stack(x)
        b, c, h, w = img_feat.size()
        img_feat = img_feat.view(b, -1)  # [b, c*h*w]

        # 处理条件
        cond_feat = self.condition_fc(conditions)  # [b, 15*20] # 修改：适应 15x20

        # 特征融合
        combined = torch.cat([img_feat, cond_feat], dim=1)
        hidden = self.fc(combined)

        return self.fc_mu(hidden), self.fc_logvar(hidden)

# 改进的Decoder (修改网络参数以适应新的图像尺寸)
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # 初始全连接
        self.fc = nn.Sequential(
            nn.Linear(Config.latent_dim + Config.condition_dim, 512*15*20), # 修改：适应 512*15*20
            nn.BatchNorm1d(512*15*20), # 修改：适应 512*15*20
            nn.LeakyReLU(0.2),
        )

        # 上采样层
        self.deconv_stack = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 15x20 -> 30x40
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 30x40 -> 60x80
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 60x80 -> 120x160
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, Config.channels, 4, 2, 1),  # 120x160 -> 240x320 # 输出通道数修改为 Config.channels
            nn.Sigmoid()
        )

    def forward(self, z, conditions):
        # 拼接潜在向量和条件
        combined = torch.cat([z, conditions], dim=1)

        # 初始投影
        x = self.fc(combined)
        x = x.view(-1, 512, 15, 20) # 修改：适应 15x20

        # 上采样
        return self.deconv_stack(x)

# CVAE模型 (保持不变)
class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x, conditions):
        mu, logvar = self.encoder(x, conditions)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, conditions)
        return recon, mu, logvar

    def get_z(self, x, conditions):
        mu, logvar = self.encoder(x, conditions)
        z = self.reparameterize(mu, logvar)
        return z

# 改进的损失函数 (保持不变)
def loss_function(recon_x, x, mu, logvar):
    # 重建损失
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# 保存图像网格
def save_sample_images(real_images, reconstructed_images, generated_images, epoch, output_dir):
    num_images = min(Config.num_test_images, 5) # 保存的图像数量，最多5个，防止测试集少于5个
    real_grid = save_image(real_images[:num_images], os.path.join(output_dir, f'epoch_{epoch}_real.png'), nrow=num_images, normalize=True, value_range=(0, 1)) # 灰度图调整 value_range
    recon_grid = save_image(reconstructed_images[:num_images], os.path.join(output_dir, f'epoch_{epoch}_recon.png'), nrow=num_images, normalize=True, value_range=(0, 1)) # 灰度图调整 value_range
    gen_grid = save_image(generated_images[:num_images], os.path.join(output_dir, f'epoch_{epoch}_gen.png'), nrow=num_images, normalize=True, value_range=(0, 1)) # 灰度图调整 value_range

    # 可视化拼接 (可选，如果想在一个图中显示)
    real_pil = Image.open(os.path.join(output_dir, f'epoch_{epoch}_real.png'))
    recon_pil = Image.open(os.path.join(output_dir, f'epoch_{epoch}_recon.png'))
    gen_pil = Image.open(os.path.join(output_dir, f'epoch_{epoch}_gen.png'))

    widths, heights = zip(real_pil.size, recon_pil.size, gen_pil.size)
    total_width = sum(widths)
    max_height = max(heights)

    combined_img = Image.new('RGB', (total_width, max_height)) # 仍然创建RGB，因为PIL默认保存为RGB即使是灰度图
    x_offset = 0
    for img in (real_pil, recon_pil, gen_pil):
        combined_img.paste(img, (x_offset,0))
        x_offset += img.size[0]

    combined_img.save(os.path.join(output_dir, f'epoch_{epoch}_combined.png'))


# 测试函数 (生成和保存图像)
def test_and_generate(model, test_loader, epoch, output_dir):
    model.eval()
    real_images_list = []
    reconstructed_images_list = []
    generated_images_list = []


    with torch.no_grad():
        for real_imgs, real_conds in test_loader:
            start_time = time.time()
            real_imgs = real_imgs.to(Config.device)
            real_conds = real_conds.to(Config.device)
            # 重构图像
            recon_imgs, mu, logvar = model(real_imgs, real_conds)
            end_time = time.time()
            print(f"重建花费：{end_time - start_time:.3f}s")
            # 生成图像 (使用相同的条件，但从潜在空间采样)
            z = torch.randn(real_imgs.size(0), Config.latent_dim).to(Config.device) # 从标准正态分布采样z
            gen_imgs = model.decoder(z, real_conds)

            real_images_list.append(real_imgs.cpu())
            reconstructed_images_list.append(recon_imgs.cpu())
            generated_images_list.append(gen_imgs.cpu())

    real_images = torch.cat(real_images_list, dim=0)
    reconstructed_images = torch.cat(reconstructed_images_list, dim=0)
    generated_images = torch.cat(generated_images_list, dim=0)

    save_sample_images(real_images, reconstructed_images, generated_images, epoch, output_dir)
    model.train() # 恢复训练模式

    print(f"Epoch {epoch}: Test images, reconstructions, and generations saved to {output_dir}")

# 训练函数
def train():
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(Config.img_size),
        transforms.RandomHorizontalFlip(),  # 数据增强
        transforms.ToTensor(),
    ])

    # 准备数据集和数据加载器
    train_dataset = ImageDataset('vae_train_imgs', transform=transform, split='train')
    test_dataset = ImageDataset('vae_train_imgs', transform=transform, split='test') # 创建测试集

    train_loader = DataLoader(train_dataset,
                              batch_size=Config.batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, # 使用测试集
                             batch_size=Config.num_test_images, # 测试集一次性全部加载，或者根据实际情况调整
                             shuffle=False, # 测试集不需要shuffle
                             num_workers=2,
                             pin_memory=True)

    print(f"test len:{len(test_dataset)}")

    # 初始化模型
    model = CVAE().to(Config.device)
    model.load_state_dict(torch.load(f'{Config.output_dir}/best_cvae.pth')) # 加载模型参数
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    # 训练循环
    best_loss = float('inf')
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0

        for imgs, conds in train_loader: # 使用 train_loader
            imgs = imgs.to(Config.device)
            conds = conds.to(Config.device)

            optimizer.zero_grad()
            # print(type(imgs),imgs.shape)
            recon, mu, logvar = model(imgs, conds)
            loss = loss_function(recon, imgs, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 计算平均损失
        avg_loss = total_loss / len(train_loader.dataset) # 使用 train_loader.dataset
        scheduler.step(avg_loss)

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f'{Config.output_dir}/best_cvae.pth')

        # 定期保存检查点
        if epoch % Config.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, f'{Config.output_dir}/cvae_checkpoint_{epoch}.pth')

        # 定期测试和生成图像
        if epoch % Config.test_interval == 0:
            test_and_generate(model, test_loader, epoch, Config.output_dir) # 使用 test_loader

        print(f'Epoch [{epoch+1}/{Config.epochs}] Loss: {avg_loss:.4f} LR: {optimizer.param_groups[0]["lr"]:.1e}')

# 运行训练
if __name__ == '__main__':
    train()