import os
import time

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from cvae import CVAE, Config  # 假设模型定义在 model.py 文件中

def perform_cvae_forward(image_path, conditions, model_path='cvae_output240/best_cvae.pth'):
    """
    加载预训练的CVAE模型，并对指定图像和条件进行前向传播。

    Args:
        image_path (str): 图像文件的路径。
        conditions (list or numpy.ndarray): 与图像相关的条件参数列表。
        model_path (str, optional): 预训练模型文件的路径。
                                     默认为 'cvae_output/best_cvae.pth'。

    Returns:
        tuple: 包含重构图像 (recon_imgs), 均值 (mu), 和对数方差 (logvar) 的元组。
               返回的都是 PyTorch tensors，并且在 CPU 上。
    """
    # 检查 CUDA 是否可用，并设置设备
    device = Config.device

    # 数据预处理 - 与训练时保持一致
    transform = transforms.Compose([
        transforms.Resize(Config.img_size),
        transforms.ToTensor(), #  不需要 RandomHorizontalFlip() 等数据增强
    ])

    # 加载图像并进行预处理
    try:
        image_np = cv2.imread(image_path) # 使用 cv2 读取图像
        if image_np is None:
            raise FileNotFoundError(f"无法读取图像文件: {image_path}")
        image_np_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) # 转换为灰度图
        print(image_np_gray.shape)
        image_pil = Image.fromarray(image_np_gray) # 从 numpy array 创建 PIL Image
        print(f"image:{type(image_pil)},{image_pil.size}")
    except FileNotFoundError:
        raise FileNotFoundError(f"图像文件未找到: {image_path}")
    image_tensor = transform(image_pil).unsqueeze(0) # 添加 batch 维度
    image_tensor = image_tensor.to(device)
    image_tensor = image_tensor.float()

    # 准备条件张量
    conditions_tensor = torch.tensor([float(c) for c in conditions]).unsqueeze(0).to(device) # 添加 batch 维度

    # 加载模型
    model = CVAE().to(Config.device)
    model.load_state_dict(torch.load('cvae_output/best_cvae.pth'))
    model.eval() # 设置为评估模式

    # 进行前向传播，禁用梯度计算
    with torch.no_grad():
        recon_imgs, mu, logvar = model(image_tensor, conditions_tensor)
    print(f"recong img:{type(recon_imgs)},{recon_imgs.size()}")
    start_time = time.time()
    # 进行前向传播，禁用梯度计算
    with torch.no_grad():
        z = model.get_z(image_tensor, conditions_tensor)
        print("zzz",z.flatten().shape)
    end_time = time.time()


    print(f"time costs: {end_time - start_time}")
    # 使用 cv2 显示重构图像 (需要转换回 numpy array 并调整通道)
    recon_img_np = recon_imgs.cpu().numpy().squeeze() # 移除 batch 和 channel 维度
    recon_img_np = (recon_img_np * 255).astype(np.uint8) # 反归一化到 0-255 并转换为 uint8
    cv2.imshow("recongimg", recon_img_np) # 直接显示灰度图
    cv2.waitKey(0)
    return recon_imgs.cpu(), mu.cpu(), logvar.cpu() # 返回 CPU 张量


if __name__ == '__main__':
    # 示例使用
    # 假设你有一张图片 'sample_image.jpg' 和对应的 14 个条件参数
    sample_image_path = 'vae_test_imgs/5 1 320 233 0.74 0.28 -0.58 0.02 -0.01 0.77 0.64 -0.00 0.00 0.00 1.00.jpg' # 请替换为你的实际图片路径
    # sample_conditions = [0.5] * 15 # 请替换为你的实际条件参数
    sample_conditions = torch.tensor([float(x) for x in '5 1 320 233 0.74 0.28 -0.58 0.02 -0.01 0.77 0.64 -0.00 0.00 0.00 1.00'.split()], dtype=torch.float32)
    print(f"sample:{sample_conditions.shape}")
    # 创建一个虚拟的灰度图像用于测试，如果sample_image.jpg不存在
    if not os.path.exists(sample_image_path):
        dummy_img_np = np.zeros(Config.img_size, dtype=np.uint8) + 128 # 创建一个灰度图像的 numpy array
        cv2.imwrite(sample_image_path, dummy_img_np) # 使用 cv2 保存
        print(f"创建了一个虚拟测试图像: {sample_image_path}")
    try:

        recon_imgs, mu, logvar = perform_cvae_forward(sample_image_path, sample_conditions)
        print(f"前向传播完成!")
        print("Reconstructed Images shape:", recon_imgs.shape)
        print("Mu shape:", mu.shape)
        print("Logvar shape:", logvar.shape)

        # 你可以进一步处理 recon_imgs, mu, logvar，例如保存重构图像
        from torchvision.utils import save_image
        save_image(recon_imgs, 'reconstructed_sample.png')
        print("重构图像已保存为 reconstructed_sample.png")

    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")
        import traceback
        traceback.print_exc() # 打印更详细的错误信息