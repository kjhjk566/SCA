"""
简单的FCSTGNN模块测试
"""

import sys
import os

# 添加路径
sys.path.append('/home/kuangjunhua/research/new_method')

try:
    print("开始导入模块...")
    
    # 检查PyTorch
    import torch
    print(f"✓ PyTorch {torch.__version__} 导入成功")
    
    # 检查自定义模块
    from module.fcstgnn_config import DEFAULT_CONFIG
    print("✓ fcstgnn_config 导入成功")
    
    from module.SpatioTemporalBlock import SpatioTemporalBlock
    print("✓ SpatioTemporalBlock 导入成功")
    
    from module.TemporalEncoder import TemporalEncoder
    print("✓ TemporalEncoder 导入成功")
    
    # 简单功能测试
    print("\n开始功能测试...")
    
    # 测试配置
    config = DEFAULT_CONFIG
    print(f"✓ 配置加载成功: {config.pooling_choice}")
    
    # 创建简单的时空模块
    st_block = SpatioTemporalBlock(
        input_dim=1,
        conv_out=4,
        lstm_hidden_dim=8,
        lstm_out_dim=4,
        conv_kernel=3,
        hidden_dim=8,
        time_length=5,
        num_node=2,
        num_windows=2,
        moving_window=[2, 3],
        stride=[1, 1],
        decay=0.9,
        pooling_choice='mean',
        dropout=0.1
    )
    print("✓ SpatioTemporalBlock 创建成功")
    
    # 创建测试数据
    x = torch.randn(2, 5, 2, 1)  # [batch, time, nodes, features]
    print(f"✓ 测试数据创建成功: {x.shape}")
    
    # 前向传播
    output = st_block(x)
    print(f"✓ 前向传播成功: {output.shape}")
    
    print("\n🎉 所有测试通过！FCSTGNN模块集成成功！")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
