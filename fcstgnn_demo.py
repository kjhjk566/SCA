"""
FCSTGNN时空编码器使用示例
演示如何在实际场景中使用集成的时空特征捕捉模块
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('/home/kuangjunhua/research/new_method')

from module.TemporalEncoder import TemporalEncoder
from module.fcstgnn_config import DEFAULT_CONFIG, SMALL_CONFIG, LARGE_CONFIG

def demonstrate_encoder_comparison():
    """演示Transformer和FCSTGNN编码器的差异"""
    print("=== 编码器性能比较演示 ===")
    
    # 设置参数
    batch_size = 8
    sequence_length = 50
    hidden_dim = 128
    
    # 创建模拟时间序列数据（包含周期性和趋势）
    t = np.linspace(0, 4*np.pi, sequence_length)
    data = []
    for i in range(batch_size):
        # 每个样本有不同的频率和相位
        freq = 1 + 0.5 * i
        phase = np.pi * i / 4
        trend = 0.1 * i * t
        noise = 0.1 * np.random.randn(sequence_length)
        
        signal = np.sin(freq * t + phase) + 0.5 * np.cos(2 * freq * t) + trend + noise
        data.append(signal)
    
    x = torch.tensor(np.array(data), dtype=torch.float32)
    print(f"输入数据形状: {x.shape}")
    
    # 创建两种编码器
    transformer_encoder = TemporalEncoder(
        input_length=sequence_length,
        hidden_dim=hidden_dim,
        encoder_type='transformer'
    )
    
    fcstgnn_encoder = TemporalEncoder(
        input_length=sequence_length,
        hidden_dim=hidden_dim,
        encoder_type='fcstgnn'
    )
    
    # 编码对比
    with torch.no_grad():
        transformer_features = transformer_encoder(x)
        fcstgnn_features = fcstgnn_encoder(x)
    
    print(f"Transformer特征: {transformer_features.shape}")
    print(f"FCSTGNN特征: {fcstgnn_features.shape}")
    
    # 计算特征差异
    feature_diff = torch.norm(transformer_features - fcstgnn_features, dim=1)
    print(f"特征差异范围: {feature_diff.min().item():.4f} - {feature_diff.max().item():.4f}")
    print(f"平均特征差异: {feature_diff.mean().item():.4f}")
    
    return transformer_features, fcstgnn_features

def demonstrate_different_configs():
    """演示不同配置的性能影响"""
    print("\n=== 不同配置性能演示 ===")
    
    batch_size = 4
    sequence_length = 30
    hidden_dim = 64
    
    # 创建测试数据
    x = torch.randn(batch_size, sequence_length)
    
    configs = {
        'SMALL': SMALL_CONFIG,
        'DEFAULT': DEFAULT_CONFIG,
        'LARGE': LARGE_CONFIG
    }
    
    for name, config in configs.items():
        print(f"\n--- {name} 配置 ---")
        print(f"滑动窗口: {config.moving_window}")
        print(f"LSTM隐藏维度: {config.lstm_hidden_dim}")
        print(f"卷积输出通道: {config.conv_out}")
        
        # 使用FCSTGNN编码器
        encoder = TemporalEncoder(
            input_length=sequence_length,
            hidden_dim=hidden_dim,
            encoder_type='fcstgnn'
        )
        
        # 测量编码时间
        import time
        start_time = time.time()
        with torch.no_grad():
            features = encoder(x)
        end_time = time.time()
        
        print(f"输出特征: {features.shape}")
        print(f"编码时间: {(end_time - start_time)*1000:.2f}ms")
        print(f"特征统计: min={features.min().item():.4f}, max={features.max().item():.4f}")

def demonstrate_anomaly_detection():
    """演示时空编码器在异常检测中的应用"""
    print("\n=== 异常检测应用演示 ===")
    
    # 创建正常和异常数据
    sequence_length = 40
    batch_size = 6
    
    # 正常数据：规律的正弦波
    normal_data = []
    for i in range(batch_size // 2):
        t = np.linspace(0, 2*np.pi, sequence_length)
        signal = np.sin(t) + 0.1 * np.random.randn(sequence_length)
        normal_data.append(signal)
    
    # 异常数据：包含突变点的信号
    anomaly_data = []
    for i in range(batch_size // 2):
        t = np.linspace(0, 2*np.pi, sequence_length)
        signal = np.sin(t) + 0.1 * np.random.randn(sequence_length)
        # 添加异常：中间部分突然跳跃
        signal[sequence_length//2:sequence_length//2+5] += 2.0
        anomaly_data.append(signal)
    
    # 合并数据
    all_data = normal_data + anomaly_data
    x = torch.tensor(np.array(all_data), dtype=torch.float32)
    labels = [0] * (batch_size // 2) + [1] * (batch_size // 2)  # 0=正常, 1=异常
    
    # 使用FCSTGNN编码器
    encoder = TemporalEncoder(
        input_length=sequence_length,
        hidden_dim=64,
        encoder_type='fcstgnn'
    )
    
    with torch.no_grad():
        features = encoder(x)
    
    # 简单的异常检测：计算特征的欧几里德距离
    center = features[:batch_size//2].mean(dim=0)  # 正常样本的中心
    distances = torch.norm(features - center, dim=1)
    
    print("样本异常度分析:")
    for i, (dist, label) in enumerate(zip(distances, labels)):
        status = "异常" if label == 1 else "正常"
        print(f"样本 {i+1}: 距离={dist.item():.4f}, 真实标签={status}")
    
    # 简单的异常检测性能
    threshold = distances[:batch_size//2].max().item()  # 用正常样本最大距离作为阈值
    predictions = (distances > threshold).int().tolist()
    accuracy = sum([p == l for p, l in zip(predictions, labels)]) / len(labels)
    print(f"异常检测准确率: {accuracy:.2%} (阈值: {threshold:.4f})")

def demonstrate_multi_scale_features():
    """演示多尺度特征捕捉能力"""
    print("\n=== 多尺度特征捕捉演示 ===")
    
    # 创建包含多个尺度模式的信号
    sequence_length = 60
    t = np.linspace(0, 6*np.pi, sequence_length)
    
    # 多尺度信号：短周期 + 长周期 + 趋势
    short_period = np.sin(4 * t)      # 短周期
    long_period = 0.5 * np.sin(0.5 * t)  # 长周期
    trend = 0.1 * t                   # 线性趋势
    noise = 0.1 * np.random.randn(sequence_length)
    
    signal = short_period + long_period + trend + noise
    
    # 单个样本测试
    x = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    print(f"多尺度信号形状: {x.shape}")
    
    # FCSTGNN编码
    encoder = TemporalEncoder(
        input_length=sequence_length,
        hidden_dim=32,
        encoder_type='fcstgnn'
    )
    
    with torch.no_grad():
        features = encoder(x)
    
    print(f"提取的时空特征: {features.shape}")
    print(f"特征统计: mean={features.mean().item():.4f}, std={features.std().item():.4f}")
    
    # 可视化信号（如果可能）
    try:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(t, signal)
        plt.title('Multi-scale Input Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        
        plt.subplot(1, 2, 2)
        features_np = features.squeeze().numpy()
        plt.bar(range(len(features_np)), features_np)
        plt.title('FCSTGNN Features')
        plt.xlabel('Feature Index')
        plt.ylabel('Feature Value')
        
        plt.tight_layout()
        plt.savefig('/home/kuangjunhua/research/new_method/fcstgnn_demo.png')
        print("可视化结果保存至: fcstgnn_demo.png")
    except Exception as e:
        print(f"可视化失败 (可能缺少matplotlib): {e}")

def main():
    """运行所有演示"""
    print("🚀 FCSTGNN时空编码器演示程序")
    print("=" * 50)
    
    try:
        # 1. 编码器比较
        demonstrate_encoder_comparison()
        
        # 2. 配置比较
        demonstrate_different_configs()
        
        # 3. 异常检测应用
        demonstrate_anomaly_detection()
        
        # 4. 多尺度特征
        demonstrate_multi_scale_features()
        
        print("\n" + "=" * 50)
        print("🎉 所有演示完成！FCSTGNN时空编码器功能正常。")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
