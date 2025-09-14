#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DataFrame模式的MetricsVisualizationTool使用示例

这个示例展示了如何使用更新后的MetricsVisualizationTool来直接处理DataFrame数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加工具路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from case_pic import MetricsVisualizationTool

def create_sample_dataframe():
    """
    创建一个示例DataFrame，模拟监控指标数据
    第一列是时间戳，后面的列是各种指标
    """
    # 创建时间序列 (24小时数据，每分钟一个点)
    start_time = datetime.now() - timedelta(hours=24)
    timestamps = [start_time + timedelta(minutes=i) for i in range(24 * 60)]
    
    # 模拟不同的指标数据
    np.random.seed(42)  # 设置随机种子以获得可重现的结果
    
    data = {
        'timestamp': timestamps,
        'recommendationservice_cpu_usage': np.random.normal(50, 10, len(timestamps)),
        'recommendationservice_memory_usage': np.random.normal(70, 15, len(timestamps)),
        'recommendationservice_response_time': np.random.normal(200, 50, len(timestamps)),
        'recommendationservice_error_rate': np.random.exponential(2, len(timestamps)),
        'cartservice_cpu_usage': np.random.normal(30, 8, len(timestamps)),
        'cartservice_memory_usage': np.random.normal(40, 12, len(timestamps)),
        'frontend_cpu_usage': np.random.normal(25, 5, len(timestamps)),
        'frontend_memory_usage': np.random.normal(35, 8, len(timestamps)),
        'database_connections': np.random.poisson(50, len(timestamps)),
        'network_throughput': np.random.normal(1000, 200, len(timestamps))
    }
    
    # 在故障时间段添加异常值
    fault_start_idx = len(timestamps) - 120  # 故障开始于2小时前
    fault_end_idx = len(timestamps) - 60     # 故障结束于1小时前
    
    # 在故障期间增加CPU和内存使用率，增加响应时间和错误率
    data['recommendationservice_cpu_usage'][fault_start_idx:fault_end_idx] += 30
    data['recommendationservice_memory_usage'][fault_start_idx:fault_end_idx] += 20
    data['recommendationservice_response_time'][fault_start_idx:fault_end_idx] += 300
    data['recommendationservice_error_rate'][fault_start_idx:fault_end_idx] *= 5
    
    df = pd.DataFrame(data)
    
    # 确保数值在合理范围内
    df = df.clip(lower=0)
    
    return df

def example_1_basic_usage():
    """
    示例1: 基础DataFrame模式使用
    """
    print("=== 示例1: 基础DataFrame模式使用 ===")
    
    # 创建示例数据
    df = create_sample_dataframe()
    print(f"创建的示例数据形状: {df.shape}")
    print(f"数据列: {list(df.columns)}")
    
    # 设置故障时间（对应我们在数据中添加异常的时间）
    fault_start = (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
    fault_end = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
    
    # 创建可视化工具，只显示recommendationservice相关指标
    tool = MetricsVisualizationTool(
        output_dir="./example_charts/basic_usage",
        fault_start_time=fault_start,
        fault_end_time=fault_end,
        metric_name=['recommendationservice'],  # 只显示包含'recommendationservice'的指标
        data_df=df  # 直接传入DataFrame
    )
    
    # 生成图表
    tool.create_charts()
    tool.generate_summary_report()
    
    print(f"图表已保存到: {tool.output_dir}")

def example_2_runtime_dataframe():
    """
    示例2: 运行时传入DataFrame
    """
    print("\n=== 示例2: 运行时传入DataFrame ===")
    
    # 创建示例数据
    df = create_sample_dataframe()
    
    # 设置显示时间范围（只显示最后6小时的数据）
    display_start = (datetime.now() - timedelta(hours=6)).strftime('%Y-%m-%d %H:%M:%S')
    display_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 创建可视化工具，不在初始化时提供DataFrame
    tool = MetricsVisualizationTool(
        output_dir="./example_charts/runtime_df",
        metric_name=['cpu', 'memory'],  # 只显示CPU和内存相关指标
        display_start_time=display_start,
        display_end_time=display_end
    )
    
    # 运行时传入DataFrame
    tool.create_charts(df=df)
    tool.generate_summary_report(df=df)
    
    print(f"图表已保存到: {tool.output_dir}")

def example_3_multiple_filters():
    """
    示例3: 多种指标筛选
    """
    print("\n=== 示例3: 多种指标筛选 ===")
    
    df = create_sample_dataframe()
    
    # 创建多个工具实例，使用不同的筛选条件
    filters = [
        (['recommendationservice'], 'recommendation_service'),
        (['cpu'], 'cpu_metrics'),
        (['memory'], 'memory_metrics'),
        (['error', 'response'], 'performance_metrics')
    ]
    
    for filter_list, folder_name in filters:
        tool = MetricsVisualizationTool(
            output_dir=f"./example_charts/{folder_name}",
            metric_name=filter_list,
            data_df=df
        )
        tool.create_charts()
        print(f"{folder_name} 图表已保存")

def main():
    """
    运行所有示例
    """
    print("DataFrame模式的MetricsVisualizationTool使用示例")
    print("=" * 50)
    
    # 确保输出目录存在
    os.makedirs("./example_charts", exist_ok=True)
    
    try:
        example_1_basic_usage()
        example_2_runtime_dataframe()
        example_3_multiple_filters()
        
        print("\n所有示例运行完成!")
        print("请查看 ./example_charts 目录中生成的图表")
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
