import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_failed_cases(case_data, pred_services, true_services, instance_names, window_size=10, save_dir='visualization_results'):
    """
    绘制预测失败案例的指标对比图
    
    参数:
    case_data: 原始指标数据
    pred_services: 预测的根因服务列表
    true_services: 真实的根因服务列表
    instance_names: 所有实例名称列表
    window_size: 时间窗口大小
    save_dir: 图片保存目录
    """
    # 确保保存目录存在
    ensure_dir(save_dir)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置图表样式
    sns.set_style("whitegrid")
    
    # 为每个预测失败的服务创建子图
    n_plots = len(pred_services) + len(true_services)
    fig, axes = plt.subplots(n_plots, 1, figsize=(15, 5*n_plots))
    if n_plots == 1:
        axes = [axes]
    
    # 绘制预测的服务指标
    for i, service in enumerate(pred_services):
        ax = axes[i]
        # 检查是否是服务级别的故障
        if '-' not in service:
            # 找到该服务的所有实例的所有指标
            service_metrics = [col for col in case_data.columns if f"&{service}-" in col]
        else:
            # 实例级别的故障，只显示该实例的指标
            service_metrics = [col for col in case_data.columns if col.endswith(f"&{service}")]
        
        for metric in service_metrics:
            metric_data = case_data[metric]
            metric_name = metric.split('&')[0]  # 获取指标名称
            ax.plot(metric_data, label=metric_name, alpha=0.7)
        
        ax.set_title(f'预测的根因服务: {service}')
        ax.set_xlabel('时间窗口')
        ax.set_ylabel('指标值')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 绘制真实的服务指标
    for i, service in enumerate(true_services):
        ax = axes[len(pred_services) + i]
        # 检查是否是服务级别的故障
        if '-' not in service:
            # 找到该服务的所有实例的所有指标
            service_metrics = [col for col in case_data.columns if f"&{service}-" in col]
        else:
            # 实例级别的故障，只显示该实例的指标
            service_metrics = [col for col in case_data.columns if col.endswith(f"&{service}")]
        
        for metric in service_metrics:
            metric_data = case_data[metric]
            metric_name = metric.split('&')[0]  # 获取指标名称
            ax.plot(metric_data, label=metric_name, alpha=0.7)
        
        ax.set_title(f'真实的根因服务: {service}')
        ax.set_xlabel('时间窗口')
        ax.set_ylabel('指标值')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"failed_cases_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"图片已保存至: {save_path}")

def plot_metrics_comparison(case_data, pred_services, true_services, instance_names, window_size=10, save_dir='visualization_results'):
    """
    绘制预测服务和真实服务的指标对比图（并排显示）
    
    参数:
    case_data: 原始指标数据
    pred_services: 预测的根因服务列表
    true_services: 真实的根因服务列表
    instance_names: 所有实例名称列表
    window_size: 时间窗口大小
    save_dir: 图片保存目录
    """
    # 确保保存目录存在
    ensure_dir(save_dir)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置图表样式
    sns.set_style("whitegrid")
    
    # 创建左右两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # 绘制预测的服务指标
    for service in pred_services:
        # 检查是否是服务级别的故障
        if '-' not in service:
            # 找到该服务的所有实例的所有指标
            service_metrics = [col for col in case_data.columns if f"&{service}-" in col]
        else:
            # 实例级别的故障，只显示该实例的指标
            service_metrics = [col for col in case_data.columns if col.endswith(f"&{service}")]
            
        for metric in service_metrics:
            metric_data = case_data[metric]
            metric_name = metric.split('&')[0]  # 获取指标名称
            ax1.plot(metric_data, label=f"{metric_name}", alpha=0.7)
    
    ax1.set_title('预测的根因服务指标')
    ax1.set_xlabel('时间窗口')
    ax1.set_ylabel('指标值')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 绘制真实的服务指标
    for service in true_services:
        # 检查是否是服务级别的故障
        if '-' not in service:
            # 找到该服务的所有实例的所有指标
            service_metrics = [col for col in case_data.columns if f"&{service}-" in col]
        else:
            # 实例级别的故障，只显示该实例的指标
            service_metrics = [col for col in case_data.columns if col.endswith(f"&{service}")]
            
        for metric in service_metrics:
            metric_data = case_data[metric]
            metric_name = metric.split('&')[0]  # 获取指标名称
            ax2.plot(metric_data, label=f"{metric_name}", alpha=0.7)
    
    ax2.set_title('真实的根因服务指标')
    ax2.set_xlabel('时间窗口')
    ax2.set_ylabel('指标值')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comparison_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"图片已保存至: {save_path}") 