import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def plot_metrics(df, metric_names, start_timestamp, end_timestamp, title=None):
    """
    绘制指定时间范围内的指标折线图
    
    参数:
    df: pandas.DataFrame, 包含时间戳和指标数据的数据框
    metric_names: list, 要绘制的指标名称列表
    start_timestamp: int/float, 开始时间戳
    end_timestamp: int/float, 结束时间戳
    title: str, 图表标题（可选）
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建图表
    plt.figure(figsize=(12, 6))
    
    # 设置样式
    sns.set_style("whitegrid")
    
    # 筛选时间范围内的数据
    mask = (df['timestamp'] >= start_timestamp) & (df['timestamp'] <= end_timestamp)
    plot_df = df.loc[mask]
    
    # 将时间戳转换为可读时间
    plot_df['time'] = pd.to_datetime(plot_df['timestamp'], unit='s')
    
    # 绘制每个指标的折线
    for metric in metric_names:
        if metric in plot_df.columns:
            plt.plot(plot_df['time'], plot_df[metric], label=metric, linewidth=2)
    
    # 设置图表属性
    plt.xlabel('时间')
    plt.ylabel('指标值')
    if title:
        plt.title(title)
    else:
        plt.title('指标时间序列图')
    
    # 设置x轴时间格式
    plt.gcf().autofmt_xdate()
    
    # 添加图例
    plt.legend(loc='best')
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图表
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 示例数据
    data = {
        'timestamp': [1620000000, 1620003600, 1620007200, 1620010800],
        'cpu_usage': [30, 45, 60, 75],
        'memory_usage': [50, 55, 65, 70]
    }
    df = pd.DataFrame(data)
    
    # 调用函数
    plot_metrics(
        df=df,
        metric_names=['cpu_usage', 'memory_usage'],
        start_timestamp=1620000000,
        end_timestamp=1620010800,
        title='系统资源使用率'
    ) 