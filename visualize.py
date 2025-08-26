import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

def visualize_instance_reconstruction(model, data_loader, config, device, instance_name, save_path=None):
    """
    可视化指定实例的重构效果（处理完整的data_loader）
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        config: 配置对象
        device: 设备
        instance_name: 要查看的实例名称
        save_path: 保存路径（可选）
    """
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    
    print("正在处理所有批次数据...")
    
    # 处理所有batch的数据
    with torch.no_grad():
        for batch_idx, (x_window, instance_names, y_window) in enumerate(data_loader):
            x_window = x_window.to(device)
            y_window = y_window.to(device)
            
            # 获取重构结果
            predictions, _ = model.get_prediction(x_window)
            
            # 转换为CPU numpy数组并收集
            batch_y_true = y_window.cpu().numpy()  # [batch_size, num_metrics]
            batch_y_pred = predictions.cpu().numpy()  # [batch_size, num_metrics]
            
            all_y_true.append(batch_y_true)
            all_y_pred.append(batch_y_pred)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"已处理 {batch_idx + 1} 个批次")
    
    # 拼接所有批次的数据
    all_y_true = np.concatenate(all_y_true, axis=0)  # [total_samples, num_metrics]
    all_y_pred = np.concatenate(all_y_pred, axis=0)  # [total_samples, num_metrics]
    
    print(f"总共处理了 {all_y_true.shape[0]} 个时间步，{all_y_true.shape[1]} 个指标")
    
    # 找到指定实例的指标索引范围
    start_idx = 0
    instance_metrics_indices = []
    instance_metric_names = []
    
    for inst_name in config.instance_metric_count_dict.keys():
        num_metrics = config.instance_metric_count_dict[inst_name]
        
        if inst_name == instance_name:
            instance_metrics_indices = list(range(start_idx, start_idx + num_metrics))
            # 生成指标名称
            for i in range(num_metrics):
                instance_metric_names.append(f"{inst_name}_metric_{i}")
            break
        
        start_idx += num_metrics
    
    # 检查是否找到了指定的实例
    if not instance_metrics_indices:
        print(f"未找到实例 '{instance_name}'")
        print(f"可用的实例有: {list(config.instance_metric_count_dict.keys())}")
        return
    
    # 提取该实例的完整时间序列数据
    instance_y_true = all_y_true[:, instance_metrics_indices]  # [total_samples, num_instance_metrics]
    instance_y_pred = all_y_pred[:, instance_metrics_indices]  # [total_samples, num_instance_metrics]
    
    num_metrics = len(instance_metrics_indices)
    num_timesteps = instance_y_true.shape[0]
    
    print(f"实例 '{instance_name}' 有 {num_metrics} 个指标，{num_timesteps} 个时间步")
    
    # 创建子图
    cols = min(4, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
    #fig.suptitle(f'实例 "{instance_name}" 的完整时间序列重构效果对比', fontsize=16, fontweight='bold')
    
    # 处理单个子图的情况
    if num_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    elif cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # 绘制每个指标的完整时间序列对比图
    time_points = range(num_timesteps)
    
    for i in range(num_metrics):
        ax = axes[i]
        
        # 绘制完整时间序列
        metric_true = instance_y_true[:, i]
        metric_pred = instance_y_pred[:, i]
        
        ax.plot(time_points, metric_true, 'b-', label='True', linewidth=1.5, alpha=0.8)
        ax.plot(time_points, metric_pred, 'r--', label='Reconstructed', linewidth=1.5, alpha=0.8)
        
        # 计算指标
        correlation = np.corrcoef(metric_true, metric_pred)[0, 1] if len(metric_true) > 1 else 0
        mse = np.mean((metric_true - metric_pred)**2)
        mae = np.mean(np.abs(metric_true - metric_pred))
        
        # 设置标题和标签
        ax.set_title(f'{instance_metric_names[i]}\nCorr: {correlation:.3f}, MSE: {mse:.4f}, MAE: {mae:.4f}', 
                    fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        
        # 如果时间序列太长，可以选择性显示采样点
        if num_timesteps > 1000:
            ax.set_title(ax.get_title() + f'\n(显示 {num_timesteps} 个时间步)', fontsize=9)
    
    # 隐藏多余的子图
    for i in range(num_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # 保存或显示图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()
    
    # # 打印详细统计信息
    # print(f"\n实例 '{instance_name}' 完整时间序列重构效果统计:")
    # print("=" * 70)
    # print(f"时间步数: {num_timesteps}")
    # print(f"指标数量: {num_metrics}")
    # print("-" * 70)
    
    # overall_correlation = []
    # overall_mse = []
    # overall_mae = []
    
    # for i in range(num_metrics):
    #     metric_true = instance_y_true[:, i]
    #     metric_pred = instance_y_pred[:, i]
        
    #     corr = np.corrcoef(metric_true, metric_pred)[0, 1] if len(metric_true) > 1 else 0
    #     mse = np.mean((metric_true - metric_pred)**2)
    #     mae = np.mean(np.abs(metric_true - metric_pred))
        
    #     overall_correlation.append(corr)
    #     overall_mse.append(mse)
    #     overall_mae.append(mae)
        
    #     print(f"指标 {i+1:2d}: Corr={corr:6.3f}, MSE={mse:8.4f}, MAE={mae:8.4f}")
    
    # print("-" * 70)
    # print(f"平均相关性: {np.mean(overall_correlation):.3f}")
    # print(f"平均MSE:   {np.mean(overall_mse):.4f}")
    # print(f"平均MAE:   {np.mean(overall_mae):.4f}")
    
    # return {
    #     'instance_name': instance_name,
    #     'y_true': instance_y_true,
    #     'y_pred': instance_y_pred,
    #     'correlation': overall_correlation,
    #     'mse': overall_mse,
    #     'mae': overall_mae,
    #     'avg_correlation': np.mean(overall_correlation),
    #     'avg_mse': np.mean(overall_mse),
    #     'avg_mae': np.mean(overall_mae)
    # }

def visualize_multiple_instances(model, data_loader, config, device, instance_names=None, save_dir='./visualizations/'):
    """
    可视化多个实例的重构效果
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        config: 配置对象
        device: 设备
        instance_names: 要可视化的实例名称列表，如果为None则可视化所有实例
        save_dir: 保存目录
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    if instance_names is None:
        instance_names = list(config.instance_metric_count_dict.keys())
    
    results = []
    
    for instance_name in instance_names:
        print(f"\n{'='*60}")
        print(f"正在处理实例: {instance_name}")
        print(f"{'='*60}")
        
        save_path = os.path.join(save_dir, f'reconstruction_{instance_name}.png')
        
        result = visualize_instance_reconstruction(
            model=model,
            data_loader=data_loader,
            config=config,
            device=device,
            instance_name=instance_name,
            save_path=save_path
        )
        
        if result:
            results.append(result)
    
    # 生成汇总报告
    if results:
        print(f"\n{'='*80}")
        print("所有实例重构效果汇总")
        print(f"{'='*80}")
        print(f"{'实例名称':<20} {'平均相关性':<12} {'平均MSE':<12} {'平均MAE':<12}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['instance_name']:<20} {result['avg_correlation']:<12.3f} "
                  f"{result['avg_mse']:<12.4f} {result['avg_mae']:<12.4f}")
    
    return results

def save_reconstruction_data(model, data_loader, config, device, instance_name, save_path):
    """
    保存重构数据到文件（用于后续分析）
    """
    result = visualize_instance_reconstruction(
        model=model,
        data_loader=data_loader,
        config=config,
        device=device,
        instance_name=instance_name,
        save_path=None  # 不保存图像
    )
    
    if result:
        np.savez(save_path,
                y_true=result['y_true'],
                y_pred=result['y_pred'],
                correlation=result['correlation'],
                mse=result['mse'],
                mae=result['mae'])
        print(f"重构数据已保存到: {save_path}")
    
    return result