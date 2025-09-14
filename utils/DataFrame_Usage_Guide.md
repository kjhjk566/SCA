# MetricsVisualizationTool DataFrame模式使用指南

## 概述

`MetricsVisualizationTool` 已经更新以支持直接使用 DataFrame 作为数据源，而不仅仅是CSV文件。这使得工具更加灵活，可以轻松集成到现有的数据处理管道中。

## 主要更新

### 1. 新的初始化参数

```python
MetricsVisualizationTool(
    data_dir: str = "./output",           # CSV文件目录（原有）
    output_dir: str = "./charts_simple",  # 输出目录（原有）
    fault_start_time: str = None,         # 故障开始时间（原有）
    fault_end_time: str = None,           # 故障结束时间（原有）
    metric_name: List[str] = None,        # 指标筛选（原有）
    display_start_time: str = None,       # 显示开始时间（原有）
    display_end_time: str = None,         # 显示结束时间（原有）
    data_df: pd.DataFrame = None          # 新增：DataFrame数据源
)
```

### 2. 新的方法

- `load_dataframe()`: 处理DataFrame格式的数据
- `create_charts()`: 支持传入DataFrame参数
- `generate_summary_report()`: 支持传入DataFrame参数

## 使用方式

### 方式1: 初始化时提供DataFrame

```python
import pandas as pd
from case_pic import MetricsVisualizationTool

# 准备你的DataFrame（第一列必须是时间戳，后面的列是指标）
df = pd.DataFrame({
    'timestamp': [...],  # 时间戳列
    'service_cpu_usage': [...],
    'service_memory_usage': [...],
    'service_response_time': [...]
})

# 创建工具实例
tool = MetricsVisualizationTool(
    output_dir="./charts",
    fault_start_time="2025-01-22 08:17:00",
    fault_end_time="2025-01-22 08:30:00",
    metric_name=['service'],  # 筛选包含'service'的指标
    data_df=df  # 传入DataFrame
)

# 生成图表
tool.create_charts()
tool.generate_summary_report()
```

### 方式2: 运行时传入DataFrame

```python
# 创建工具实例（不提供DataFrame）
tool = MetricsVisualizationTool(
    output_dir="./charts",
    fault_start_time="2025-01-22 08:17:00",
    fault_end_time="2025-01-22 08:30:00",
    metric_name=['cpu', 'memory']
)

# 运行时传入DataFrame
tool.create_charts(df=your_dataframe)
tool.generate_summary_report(df=your_dataframe)
```

### 方式3: 兼容原有CSV文件模式

```python
# 原有的CSV文件模式仍然完全支持
tool = MetricsVisualizationTool(
    data_dir="./data/metrics.csv",
    output_dir="./charts",
    fault_start_time="2025-01-22 08:17:00",
    fault_end_time="2025-01-22 08:30:00"
)

tool.create_charts("./data/metrics.csv")
tool.generate_summary_report("./data/metrics.csv")
```

## DataFrame数据格式要求

你的DataFrame必须满足以下要求：

1. **第一列**: 时间戳列（可以是Unix时间戳或日期时间字符串）
2. **后续列**: 每一列代表一个指标
3. **列名**: 指标名称，用于图表标题和筛选

### 示例DataFrame结构

```
| timestamp     | service1_cpu | service1_memory | service2_cpu | service2_memory |
|---------------|--------------|-----------------|--------------|-----------------|
| 1747677060    | 45.2         | 67.8           | 23.1         | 34.5           |
| 1747677120    | 47.1         | 68.2           | 24.3         | 35.1           |
| ...           | ...          | ...            | ...          | ...            |
```

## 保留的所有功能

以下所有原有功能在DataFrame模式下都完全保留：

### 指标筛选
```python
# 只显示包含特定关键词的指标
metric_name=['recommendationservice', 'cpu', 'memory']
```

### 故障时间段高亮
```python
# 在图表中用红色背景标识故障时间段
fault_start_time='2025-01-22 08:17:00'
fault_end_time='2025-01-22 08:30:00'
```

### 时间范围筛选
```python
# 只显示特定时间范围的数据
display_start_time='2025-01-22 06:00:00'
display_end_time='2025-01-22 10:00:00'
```

### 自动布局和分页
- 每行显示5个图表
- 每页最多6行（30个图表）
- 自动分页处理大量指标

### 时间处理
- 自动识别Unix时间戳和日期时间格式
- 自动转换为北京时间（UTC+8）
- 智能时间轴格式化

## 完整示例

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from case_pic import MetricsVisualizationTool

# 创建示例数据
timestamps = [datetime.now() - timedelta(hours=i) for i in range(24, 0, -1)]
df = pd.DataFrame({
    'timestamp': timestamps,
    'recommendationservice_cpu': np.random.normal(50, 10, 24),
    'recommendationservice_memory': np.random.normal(70, 15, 24),
    'cartservice_cpu': np.random.normal(30, 8, 24),
    'cartservice_memory': np.random.normal(40, 12, 24)
})

# 使用DataFrame模式
tool = MetricsVisualizationTool(
    output_dir="./monitoring_charts",
    fault_start_time=(datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
    fault_end_time=(datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
    metric_name=['recommendationservice'],  # 只显示recommendation服务的指标
    data_df=df
)

# 生成图表和报告
tool.create_charts()
tool.generate_summary_report()

print(f"图表已保存到: {tool.output_dir}")
```

## 优势

1. **更灵活**: 可以直接处理内存中的DataFrame，无需先保存为CSV文件
2. **更高效**: 避免了文件I/O操作
3. **更易集成**: 可以轻松集成到现有的数据分析流程中
4. **向后兼容**: 完全保持原有CSV文件模式的功能
5. **功能完整**: 所有原有功能（指标筛选、故障高亮、时间筛选等）都得到保留

## 运行示例

运行提供的示例文件来查看工具的使用效果：

```bash
cd /home/kuangjunhua/research/SCA/utils
python dataframe_visualization_example.py
```
