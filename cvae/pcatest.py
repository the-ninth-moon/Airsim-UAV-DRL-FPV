import os
import time
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # 导入 matplotlib

def load_and_preprocess_images(image_folder, target_size=(32, 24), test_size=0.2, random_state=42):
    # ... (load_and_preprocess_images 函数保持不变，与之前的代码相同)
    image_paths = []
    filenames = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_paths.append(os.path.join(image_folder, filename))
            filenames.append(filename)

    train_paths, test_paths, train_filenames_list, test_filenames_list = train_test_split(
        image_paths, filenames, test_size=test_size, random_state=random_state
    )

    train_vectors = []
    test_vectors = []

    for img_path in train_paths:
        try:
            img = Image.open(img_path).convert('L')
            img_resized = img.resize(target_size, Image.BILINEAR)
            img_array = np.array(img_resized)
            img_vector = img_array.flatten()
            train_vectors.append(img_vector)
        except Exception as e:
            filename = os.path.basename(img_path)
            print(f"Error processing training image {filename}: {e}")

    for img_path in test_paths:
        try:
            img = Image.open(img_path).convert('L')
            img_resized = img.resize(target_size, Image.BILINEAR)
            img_array = np.array(img_resized)
            img_vector = img_array.flatten()
            test_vectors.append(img_vector)
        except Exception as e:
            filename = os.path.basename(img_path)
            print(f"Error processing testing image {filename}: {e}")

    return np.array(train_vectors), np.array(test_vectors), train_filenames_list, test_filenames_list


def train_pca(image_vectors, n_components=64):
    # ... (train_pca 函数保持不变，与之前的代码相同)
    pca = PCA(n_components=n_components)
    pca.fit(image_vectors)
    return pca

def compress_image_with_pca(image_path, pca_model, target_size=(32, 24)):
    """
    加载单张图片，降采样，使用训练好的 PCA 模型压缩到指定维度，并返回原始降采样后的图像数组。

    Args:
        image_path: 图片路径
        pca_model: 训练好的 PCA 模型
        target_size: 降采样后的目标尺寸

    Returns:
        compressed_vector: 压缩后的 64 维向量
        original_resized_array: 原始降采样后的图像数组 (用于对比)
    """
    try:
        img = Image.open(image_path).convert('L')
        img_resized = img.resize(target_size, Image.BILINEAR)
        original_resized_array = np.array(img_resized) # 获取降采样后的图像数组
        img_vector = original_resized_array.flatten()
        compressed_vector = pca_model.transform(img_vector.reshape(1, -1))
        return compressed_vector.flatten(), original_resized_array # 返回压缩向量和原始图像数组
    except Exception as e:
        filename = os.path.basename(image_path)
        print(f"Error processing image {filename}: {e}")
        return None, None # 出错时返回 None, None


def reconstruct_image_from_pca_vector(compressed_vector, pca_model, output_size):
    """
    使用 PCA 模型将压缩向量恢复成图像数组。

    Args:
        compressed_vector: 压缩后的 64 维向量
        pca_model: 训练好的 PCA 模型
        output_size: 恢复后的图像尺寸 (宽度, 高度)，即降采样后的尺寸

    Returns:
        reconstructed_array: 恢复后的图像数组
    """
    reconstructed_vector = pca_model.inverse_transform(compressed_vector.reshape(1, -1))
    reconstructed_array = reconstructed_vector.reshape(output_size[1], output_size[0]) # 注意 reshape 成 (height, width)
    return reconstructed_array


if __name__ == '__main__':
    image_folder = 'vae_train_imgs'
    target_dimension = 128
    downsample_size = (32, 24) # 降采样尺寸
    test_data_ratio = 0.2
    num_comparison_images = 3 # 要对比显示的测试集图片数量

    start_time = time.time()
    train_vectors, test_vectors, train_filenames, test_filenames = load_and_preprocess_images(
        image_folder, downsample_size, test_size=test_data_ratio
    )
    end_time = time.time()
    load_time = end_time - start_time
    print(f"加载和预处理图片并划分数据集耗时: {load_time:.4f} 秒")
    print(f"训练集图片数量: {len(train_filenames)}")
    print(f"测试集图片数量: {len(test_filenames)}")

    if train_vectors.size == 0:
        print("没有加载到任何训练图片，请检查图片文件夹路径和图片格式。")
    else:
        start_time = time.time()
        pca_model = train_pca(train_vectors, target_dimension)
        end_time = time.time()
        pca_train_time = end_time - start_time
        print(f"PCA 训练耗时 (仅使用训练集): {pca_train_time:.4f} 秒")

        # 遍历测试集图片进行压缩和重建，并显示对比
        plt.figure(figsize=(10, 6)) # 设置 figure 大小，可以根据需要调整
        for i in range(min(num_comparison_images, len(test_filenames))): # 避免超出测试集图片数量
            test_image_path = os.path.join(image_folder, test_filenames[i])
            compressed_vector, original_resized_array = compress_image_with_pca(test_image_path, pca_model, downsample_size)

            if compressed_vector is not None and original_resized_array is not None:
                reconstructed_array = reconstruct_image_from_pca_vector(compressed_vector, pca_model, downsample_size)

                original_image_pil = Image.fromarray(original_resized_array.astype(np.uint8)) # 转换为 PIL Image，注意数据类型转换
                reconstructed_image_pil = Image.fromarray(reconstructed_array.astype(np.uint8))

                # 显示原始图像
                plt.subplot(num_comparison_images, 2, 2 * i + 1) # 创建子图，num_comparison_images 行，2 列
                plt.imshow(original_image_pil, cmap='gray') # 使用 'gray' colormap 显示灰度图
                plt.title(f"Original (Test Set - {test_filenames[i]})")
                plt.axis('off') # 关闭坐标轴

                # 显示重建图像
                plt.subplot(num_comparison_images, 2, 2 * i + 2)
                plt.imshow(reconstructed_image_pil, cmap='gray')
                plt.title(f"Reconstructed (PCA)")
                plt.axis('off')

                if i == 0: # 仅在第一行添加总标题，避免重复
                    plt.suptitle("Original vs. PCA Reconstructed Images (Test Set)", fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整子图布局，rect 参数调整标题位置
        plt.show() # 显示图像对比结果

        # 单张图片压缩速度测试 (使用测试集第一张图片)
        if test_filenames:
            test_image_path = os.path.join(image_folder, test_filenames[0])
            start_time = time.time()
            compressed_vector, _ = compress_image_with_pca(test_image_path, pca_model, downsample_size) # 只需要压缩向量，忽略原始数组
            end_time = time.time()
            compress_time = end_time - start_time
            print(f"\n单张测试集图片压缩耗时: {compress_time:.4f} 秒")
            print(f"压缩后的向量维度: {compressed_vector.shape}")

            if compress_time <= 0.1:
                print("单张图片压缩速度满足实时性要求 (小于 0.1 秒)。")
            else:
                print("单张图片压缩速度可能不满足实时性要求 (大于 0.1 秒)。")

            print("\n前 10 个压缩后的向量值 (测试集图片, 仅供参考):")
            print(compressed_vector[:10])
        else:
            print("测试集为空，无法进行单张图片压缩和速度测试。")