#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import glob
from typing import List, Dict, Tuple
import math

# 设置字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

class MetricsVisualizationTool:
    """
    监控指标可视化工具 - 简化版
    """
    
    def __init__(self, data_dir: str = "./output", output_dir: str = "./charts_simple", 
                 fault_start_time: str = None, fault_end_time: str = None, metric_name: List[str] = None,
                 display_start_time: str = None, display_end_time: str = None):
        """
        初始化工具
        
        Args:
            data_dir: 数据目录
            output_dir: 图表输出目录
            fault_start_time: 故障开始时间 (格式: 'YYYY-MM-DD HH:MM:SS')
            fault_end_time: 故障结束时间 (格式: 'YYYY-MM-DD HH:MM:SS')
            metric_name: 指标名称过滤列表，如果为None则显示所有指标，如果传入列表则只显示包含列表中任一子串的指标
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.max_metrics_per_file = 5  # 每个文件最多显示的指标数
        self.charts_per_row = 5        # 每行显示的图表数
        self.max_rows_per_figure = 6   # 每张大图最多的行数
        self.metric_name = metric_name  # 指标名称过滤列表
        self.display_start_time = pd.to_datetime(display_start_time) if display_start_time else None
        self.display_end_time = pd.to_datetime(display_end_time) if display_end_time else None
        
   

        
        # 故障时间段
        self.fault_start_time = None
        self.fault_end_time = None
        if fault_start_time and fault_end_time:
            try:
                # 输入的时间是北京时间，需要转换为与数据一致的格式
                # 先解析为朴素时间，然后当作UTC时间处理，再转换为北京时间
                fault_start_naive = pd.to_datetime(fault_start_time)
                fault_end_naive = pd.to_datetime(fault_end_time)
                
                # 减去8小时，然后按UTC处理，再转换为北京时间（相当于直接使用原时间）
                self.fault_start_time = fault_start_naive
                self.fault_end_time = fault_end_naive
                
                print(f"Fault period (Beijing Time): {self.fault_start_time} to {self.fault_end_time}")
            except Exception as e:
                print(f"Error parsing fault time: {e}")
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def should_include_file(self, filename: str) -> bool:
        """
        判断是否应该包含某个文件
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否应该包含此文件
        """
        if self.metric_name is None:
            return True
        
        # 提取文件对应的指标名称
        metric_name = self.extract_metric_name_from_filename(filename)
        
        # 检查是否有任何一个过滤词是指标名称的子串
        for filter_name in self.metric_name:
            if filter_name.lower() in metric_name.lower():
                return True
        
        return False
    
    def should_include_metric(self, metric_name: str) -> bool:
        """
        判断是否应该包含某个指标
        
        Args:
            metric_name: 指标名称
            
        Returns:
            bool: 是否应该包含此指标
        """
        if self.metric_name is None:
            return True
        
        # 检查是否有任何一个过滤词是指标名称的子串
        for filter_name in self.metric_name:
            if filter_name.lower() in metric_name.lower():
                return True
        
        return False
    
    # 主要修改 load_csv_files 方法中的时间转换部分
    def load_csv_files(self, csv_file_path: str = None) -> List[Tuple[str, pd.DataFrame]]:
        """
        加载单个CSV文件，第一列是时间，后面每列是一个指标
        
        Args:
            csv_file_path: CSV文件路径，如果为None则使用data_dir作为文件路径
            
        Returns:
            List[Tuple[str, pd.DataFrame]]: (指标名, DataFrame)的列表
        """
        csv_files = []
        
        # 如果没有传入文件路径，使用data_dir作为文件路径
        if csv_file_path is None:
            csv_file_path = self.data_dir
        
        try:
            print(f"Loading CSV file: {csv_file_path}")
            
            df = pd.read_csv(csv_file_path)
            
            if df.empty:
                print(f"Warning: CSV file is empty: {csv_file_path}")
                return csv_files
            
            # 假设第一列是时间列
            time_column = df.columns[0]
            print(f"Time column detected: {time_column}")
            
            # 转换时间列为datetime
            # 转换时间列为datetime
            if 'time' in time_column.lower():
                # 如果是Unix时间戳（秒）
                df['datetime'] = pd.to_datetime(df[time_column], unit='s') + pd.Timedelta(hours=8)
            else:
                # 尝试直接解析时间格式
                df['datetime'] = pd.to_datetime(df[time_column]) + pd.Timedelta(hours=8)
            print(f"Parsed datetime column: {df['datetime'].head()}")
            
            # 获取所有指标列（除了时间列）
            metric_columns = [col for col in df.columns if col != time_column and col != 'datetime']
            
            print(f"Found {len(metric_columns)} metrics: {metric_columns[:5]}..." if len(metric_columns) > 5 else f"Found {len(metric_columns)} metrics: {metric_columns}")
            
            # 为每个指标列创建单独的DataFrame
            for metric_col in metric_columns:
                # 检查是否应该包含此指标
                if not self.should_include_metric(metric_col):
                    continue
                
                # 创建只包含时间和该指标的DataFrame
                metric_df = df[['datetime', metric_col]].copy()
                metric_df = metric_df.rename(columns={metric_col: 'value'})
                
                # 添加原始列名信息用于标题显示
                metric_df.attrs['original_column'] = metric_col
                metric_df.attrs['metric_name'] = metric_col
                
                # 过滤时间范围（如果设置了显示时间范围）
                if self.display_start_time is not None and self.display_end_time is not None:
                    mask = (metric_df['datetime'] >= self.display_start_time) & (metric_df['datetime'] <= self.display_end_time)
                    metric_df = metric_df[mask]
                
                csv_files.append((metric_col, metric_df))
            
            print(f"Successfully loaded {len(csv_files)} metrics from {csv_file_path}")
            
        except Exception as e:
            print(f"Failed to load {csv_file_path}: {e}")
        
        return csv_files

    
    def extract_metric_name_from_filename(self, filename: str) -> str:
        """
        从文件名中提取插件ID或机器指标名称作为标题
        
        Args:
            filename: 文件名
            
        Returns:
            str: 插件ID或机器指标名称
        """
        # 移除.csv后缀
        name = filename.replace('.csv', '')
        
        # 分割文件名
        parts = name.split('_')
        
        # 检查是否为机器指标
        if 'machine_metric' in name and len(parts) >= 4:
            # 机器指标格式: alsc-pos-order_machine_metric_jvm_bufferpool_direct_max
            # 提取从第三部分开始的所有部分作为指标名称
            metric_name = '_'.join(parts[3:])
            return metric_name
        elif len(parts) >= 4:
            # 插件指标格式: app_plugin_id_description
            # 例如: alsc-pos-order_1030_MM_1909_hsf服务异常-预发
            # 提取插件ID：第二、三、四部分组合 (1030_MM_1909)
            plugin_id = '_'.join(parts[1:4])
            return plugin_id
        elif len(parts) >= 3:
            # 如果只有3部分，取后两部分
            plugin_id = '_'.join(parts[1:3])
            return plugin_id
        elif len(parts) >= 2:
            # 如果只有2部分，取第二部分
            return parts[1]
        
        # 如果格式不匹配，返回原文件名（限制长度）
        return filename[:30]
    
    def select_top_metrics(self, df: pd.DataFrame) -> List[str]:
        """
        选择要显示的指标列
        
        Args:
            df: DataFrame
            
        Returns:
            List[str]: 选中的指标列名
        """
        # 对于新格式，每个DataFrame只包含一个指标（value列）
        if 'value' in df.columns:
            return ['value']
        
        # 原有逻辑作为备用
        metric_columns = [col for col in df.columns 
                         if col not in ['timestamp', 'datetime', 'time'] and pd.api.types.is_numeric_dtype(df[col])]
        
        if len(metric_columns) <= self.max_metrics_per_file:
            return metric_columns
        
        # 计算每个指标的方差，选择方差最大的前N个
        variances = {}
        for col in metric_columns:
            try:
                values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(values) > 0:
                    variances[col] = values.var()
                else:
                    variances[col] = 0
            except:
                variances[col] = 0
        
        # 按方差排序，选择前N个
        sorted_metrics = sorted(variances.items(), key=lambda x: x[1], reverse=True)
        selected_metrics = [metric[0] for metric in sorted_metrics[:self.max_metrics_per_file]]
        
        print(f"Selected metrics: {len(selected_metrics)}")
        return selected_metrics
    
    def create_subplot(self, ax, df: pd.DataFrame, selected_metrics: List[str], title: str):
        """
        创建单个指标的子图
        
        Args:
            ax: matplotlib轴对象
            df: DataFrame
            selected_metrics: 选中的指标列表
            title: 图标题
        """
        try:
            if not selected_metrics:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=8, pad=2)
                return
            
            # 如果只有一个指标，直接绘制
            if len(selected_metrics) == 1:
                metric = selected_metrics[0]
                valid_data = df[['datetime', metric]].dropna()
                
                if len(valid_data) == 0:
                    ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title, fontsize=8, pad=2)
                    return
                
                ax.plot(valid_data['datetime'], valid_data[metric], linewidth=1, alpha=0.8, color='blue')
            else:
                # 多个指标：绘制标准化后的数据
                valid_data = df[['datetime'] + selected_metrics].dropna()
                
                if len(valid_data) == 0:
                    ax.text(0.5, 0.5, 'No Valid Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title, fontsize=8, pad=2)
                    return
                
                colors = ['blue', 'red', 'green', 'orange', 'purple']
                for i, metric in enumerate(selected_metrics):
                    try:
                        values = valid_data[metric]
                        if values.std() > 0:
                            normalized_values = (values - values.mean()) / values.std()
                        else:
                            normalized_values = values
                        
                        color = colors[i % len(colors)]
                        ax.plot(valid_data['datetime'], normalized_values, 
                               label=f'M{i+1}', linewidth=1, alpha=0.8, color=color)
                    except Exception as e:
                        print(f"Error plotting metric {metric}: {e}")
                
                if len(selected_metrics) > 1:
                    ax.legend(fontsize=5, loc='upper right')
            
            # 添加故障时间段的背景色
            if self.fault_start_time and self.fault_end_time:
                # 获取数据的时间范围
                if not df.empty and 'datetime' in df.columns:
                    data_start = df['datetime'].min()
                    data_end = df['datetime'].max()
                    
                    #print(f"Debug - Data time range: {data_start} to {data_end}")
                    #print(f"Debug - Fault time range: {self.fault_start_time} to {self.fault_end_time}")
                    
                    # 确保故障时间段与数据时间范围有交集
                    fault_start_display = max(self.fault_start_time, data_start)
                    fault_end_display = min(self.fault_end_time, data_end)
                    
                    #print(f"Debug - Display range: {fault_start_display} to {fault_end_display}")
                    
                    if fault_start_display <= fault_end_display:
                        ax.axvspan(fault_start_display, fault_end_display, 
                                 alpha=0.3, color='red', zorder=0)
                        #print(f"Debug - Added fault background for {title}")
                    else:
                        print(f"Debug - No overlap for {title}")
            
            # 设置标题
            ax.set_title(title, fontsize=8, pad=2)
            
            # 格式化x轴时间为北京时间
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            
            # 旋转x轴标签
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=6)
            
            # 设置y轴
            ax.tick_params(axis='y', labelsize=6)
            
            # 添加网格
            ax.grid(True, alpha=0.3, linewidth=0.5)
            
        except Exception as e:
            print(f"Error creating subplot for {title}: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)[:20]}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=8, pad=2)
    
    def create_charts(self, csv_file_path: str = None):
        """
        创建所有图表
        
        Args:
            csv_file_path: CSV文件路径，如果为None则使用data_dir
        """
        csv_files = self.load_csv_files(csv_file_path)
        
        if not csv_files:
            print("No valid metrics found in CSV file")
            return
        
        print(f"Total metrics found: {len(csv_files)}")
        
        # 计算需要的图表数量
        total_charts = len(csv_files)
        charts_per_figure = self.charts_per_row * self.max_rows_per_figure
        num_figures = math.ceil(total_charts / charts_per_figure)
        
        print(f"Creating {num_figures} figures, max {charts_per_figure} charts per figure")
        
        for fig_idx in range(num_figures):
            start_idx = fig_idx * charts_per_figure
            end_idx = min(start_idx + charts_per_figure, total_charts)
            current_files = csv_files[start_idx:end_idx]
            
            print(f"Creating figure {fig_idx + 1}, metrics {start_idx + 1} to {end_idx}")
            
            self.create_single_figure(current_files, fig_idx + 1)
    
    def create_single_figure(self, csv_files: List[Tuple[str, pd.DataFrame]], fig_num: int):
        """
        创建单张大图
        
        Args:
            csv_files: 当前大图要包含的文件列表
            fig_num: 图编号
        """
        num_charts = len(csv_files)
        num_rows = math.ceil(num_charts / self.charts_per_row)
        
        # 创建图形
        fig_width = 20  # 增加图形宽度
        fig_height = max(12, num_rows * 2.5)  # 根据行数调整高度
        
        fig, axes = plt.subplots(num_rows, self.charts_per_row, 
                                figsize=(fig_width, fig_height))
        
        # 确保axes是二维数组
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        elif self.charts_per_row == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle(f'Metrics Analysis Dashboard - Page {fig_num}', fontsize=16, y=0.98)
        
        # 绘制每个文件的图表
        for idx, (metric_column_name, df) in enumerate(csv_files):
            row = idx // self.charts_per_row
            col = idx % self.charts_per_row
            ax = axes[row, col]
            
            # 使用列名作为图表标题
            chart_title = metric_column_name
            if hasattr(df, 'attrs') and 'original_column' in df.attrs:
                # 使用完整的列名（指标名&实例名）
                chart_title = df.attrs['original_column']
            
            # 选择要显示的指标
            selected_metrics = self.select_top_metrics(df)
            
            # 创建子图
            self.create_subplot(ax, df, selected_metrics, chart_title)
        
        # 隐藏空的子图
        for idx in range(num_charts, num_rows * self.charts_per_row):
            row = idx // self.charts_per_row
            col = idx % self.charts_per_row
            axes[row, col].set_visible(False)
        
        # 调整布局
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # 保存图片
        output_path = os.path.join(self.output_dir, f'metrics_dashboard_page_{fig_num}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved chart: {output_path}")
    
    def generate_summary_report(self, csv_file_path: str = None):
        """
        生成分析报告
        
        Args:
            csv_file_path: CSV文件路径，如果为None则使用data_dir
        """
        csv_files = self.load_csv_files(csv_file_path)
        
        report_path = os.path.join(self.output_dir, 'metrics_summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Metrics Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Directory: {self.data_dir}\n")
            f.write(f"Total Metrics: {len(csv_files)}\n\n")
            
            f.write("Metric Details:\n")
            f.write("-" * 30 + "\n")
            
            for metric_column_name, df in csv_files:
                selected_metrics = self.select_top_metrics(df)
                
                f.write(f"\nMetric: {metric_column_name}\n")
                if hasattr(df, 'attrs') and 'metric_name' in df.attrs:
                    f.write(f"Metric Type: {df.attrs['metric_name']}\n")
                f.write(f"Data Shape: {df.shape}\n")
                if not df.empty and 'datetime' in df.columns:
                    f.write(f"Time Range: {df['datetime'].min()} to {df['datetime'].max()}\n")
                f.write(f"Selected Metrics: {len(selected_metrics)}\n")
        
        print(f"Summary report saved: {report_path}")

def main():
    """
    主函数
    """
    print("Starting metrics visualization...")
    
    # 故障时间段设置（如果有的话，可以根据需要调整）
    fault_start = '2025-05-20 02:10:13'  # "2025-01-22 08:17:00"
    fault_end = '2025-05-20 02:20:13'    # "2025-01-22 08:30:00"
    display_start_time='2025-05-20 00:58:00'
    display_end_time='2025-05-20 02:59:59'
    
    # CSV文件路径 - 修改为您的具体CSV文件路径
    csv_file_path = "/home/kuangjunhua/research/data/aiops25/20/all_metric.csv"  # 请修改为您的CSV文件路径
    instance =  'recommendationservice'
    # 创建可视化工具
    tool = MetricsVisualizationTool(
        data_dir=csv_file_path,  # 现在这个参数作为默认文件路径
        output_dir=f"/home/kuangjunhua/research/data/aiops25/20/chart/{instance}",  # 图表输出目录
        fault_start_time=fault_start,
        fault_end_time=fault_end,
        metric_name=[instance],  # 设置为None显示所有指标，或传入列表如 ["cpu", "memory"] 进行过滤
        display_start_time=display_start_time,
        display_end_time=display_end_time
    )
    
    # 生成图表 - 传入CSV文件路径
    tool.create_charts(csv_file_path)
    
    # 生成报告
    tool.generate_summary_report(csv_file_path)
    
    print("Visualization completed!")
    print(f"Charts saved in: {tool.output_dir}")
    if fault_start and fault_end:
        print(f"Fault period highlighted: {fault_start} to {fault_end}")

if __name__ == "__main__":
    main()
