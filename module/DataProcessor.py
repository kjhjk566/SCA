import pandas as pd
import torch
from torch_geometric.data import Batch
from module.graph.SubgraphManager import SubgraphManager
from module.graph.StaticCallGraphBuilder import StaticCallGraphBuilder
from collections import defaultdict


class DataProcessor:
    def __init__(self, df, config, window_size=20, stride=1, window_stride=3,small_window_size=5):
        """
        
        :param config: 包含 all_enum 的配置对象
        :param window_size: 取多长时间作为一个窗口
        :param stride: 滑动窗口步长
        """
        
        self.config = config
        self.window_size = window_size
        self.stride = stride
        self.window_stride = window_stride
        self.small_window_size = small_window_size

        self.df = df
        # 归一化特征列
        feature_data = self.df.iloc[:, 1:]
        min_vals = feature_data.min()
        max_vals = feature_data.max()
        normalized_features = (feature_data - min_vals) / (max_vals - min_vals + 1e-8)  # 防止除0
        self.df.iloc[:, 1:] = normalized_features
        # 重新排列列
        #print("before reorder_columns:",self.df.shape)
        self.reorder_columns()
        
        self.feature_columns = self.df.columns[1:]  # 除去timestamp列
        
        # 保存指标名称顺序到config中
        self.save_metric_names_to_config()

        
    def reorder_columns(self):
        """
        按照 config.all_enum 中定义的指标顺序，重新排列 DataFrame 中的列
        适配原始列名是 'kpi名称&实例名称' 的格式。
        """
        feature_columns = self.df.columns[1:]  # 除去timestamp列
        instance_name_map = defaultdict(list)  # k: instance name, v: list of full feature names
        for col in feature_columns:
            if "&" in col:
                _, instance = col.split("&", 1)
                instance_name_map[instance].append(col)

        ordered_feature_cols = []
        for instance in self.config.all_enum.keys():
            if instance in instance_name_map:
                ordered_feature_cols.extend(instance_name_map[instance])  # 添加所有属于该实例的指标列
            else:
                raise ValueError(f"Instance {instance} not found in raw data columns!")

        self.df = pd.concat([self.df.iloc[:, [0]], self.df[ordered_feature_cols]], axis=1)
        # 记录每个实例的指标数量
        self.instance_metric_count_dict = {inst: len(cols) for inst, cols in instance_name_map.items()}
        
        # 保存重新排序后的指标列名
        self.ordered_feature_columns = ordered_feature_cols
    
    def save_metric_names_to_config(self):
        """
        将指标名称信息保存到config中，用于可视化
        """
        # 保存完整的指标列名列表（按顺序）
        self.config.ordered_feature_names = self.ordered_feature_columns
        
        # 按实例分组保存指标名称
        self.config.instance_metric_names = {}
        start_idx = 0
        
        # 关键修复：必须按照 config.all_enum.keys() 的顺序遍历，与 reorder_columns 保持一致
        for instance in self.config.all_enum.keys():
            if instance in self.instance_metric_count_dict:
                metric_count = self.instance_metric_count_dict[instance]
                end_idx = start_idx + metric_count
                instance_metrics = self.ordered_feature_columns[start_idx:end_idx]
                
                self.config.instance_metric_names[instance] = {
                    'full_names': instance_metrics,  # 完整的列名（包含实例名）
                    'start_idx': start_idx,          # 在全局指标数组中的起始索引
                    'end_idx': end_idx,              # 在全局指标数组中的结束索引
                    'metric_count': metric_count     # 指标数量
                }
                
                start_idx = end_idx
        
        # 保存实例指标数量映射
        self.config.instance_metric_count_dict = self.instance_metric_count_dict
        
        # print("指标名称信息已保存到config中:")
        # for instance, info in self.config.instance_metric_names.items():
        #     print(f"  {instance}: {len(info['full_names'])} 个指标 (索引: {info['start_idx']}-{info['end_idx']})")
        #     for name in info['full_names']:
        #         print(f"    - {name}")
    def generate_patches_tensor(self,time_series, patch_size, stride=None):
        """
        使用 PyTorch 将多变量时间序列张量划分为 patch。

        参数:
        - time_series: torch.Tensor, shape (N, L)，N个传感器，L个时间步
        - patch_size: int，patch的时间步长
        - stride: int，滑动窗口步长，默认等于patch_size（无重叠）

        返回:
        - patches: torch.Tensor, shape (num_patches, N, patch_size)
        """
        N, L = time_series.shape
        num_patches = (L - patch_size) // stride + 1

        # 使用 unfold 展开成滑动窗口，shape: (N, num_patches, patch_size)
        patches = time_series.unfold(dimension=1, size=patch_size, step=stride)
        return patches

        
    def init_subgraph_manager(self):
        """初始化 SubgraphManager，分配指标到对应实例"""
        self.subgraph_manager = SubgraphManager(self.feature_columns, self.config)

    def generate_windows(self):
        """生成滑动时间窗口的数据对 (past, future)"""
        feature_data = torch.tensor(self.df.iloc[:, 1:].values, dtype=torch.float)  # [T, num_features]

        windows = []
        for start_idx in range(0, feature_data.size(0) - self.window_size, self.stride):
            x_window = feature_data[start_idx: start_idx + self.window_size]       # 输入窗口
           
            x_window = x_window.permute(1,0)            
            x_window = self.generate_patches_tensor(x_window, patch_size=self.small_window_size, stride=self.window_stride)  # [num_patches, num_features, patch_size]
            y_window = feature_data[start_idx + self.window_size]                  # 预测目标（单步预测）
            windows.append((x_window, y_window))
        return windows
    def generate_windows_gnet(self):
        """生成滑动时间窗口的数据对 (past, future)，输出形状 (B, C_in, N, T)"""
        feature_data = torch.tensor(self.df.iloc[:, 1:].values, dtype=torch.float)  # [T, num_features]

        windows = []
        for start_idx in range(0, feature_data.size(0) - self.window_size, self.stride):
            # [window_size, num_features]
            x_window = feature_data[start_idx: start_idx + self.window_size]

            # 转换为 (num_features, window_size)
            x_window = x_window.T  # [N, T]

            # 增加通道维度 C_in=1，并扩展 batch 维度
            # 最终形状: (1, 1, N, T)
            x_window = x_window.unsqueeze(0)  

            # 单步预测目标: [num_features]
            y_window = feature_data[start_idx + self.window_size]

            windows.append((x_window, y_window))
        return windows
    def generate_windows_normal(self):
        """生成滑动时间窗口的数据对 (past, future)，输出形状 (B, C_in, N, T)"""
        feature_data = torch.tensor(self.df.iloc[:, 1:].values, dtype=torch.float)  # [T, num_features]

        windows = []
        for start_idx in range(0, feature_data.size(0) - self.window_size, self.stride):
            # [window_size, num_features]
            x_window = feature_data[start_idx: start_idx + self.window_size]

            # 转换为 (num_features, window_size)
            x_window = x_window.T  # [N, T]

            # 增加通道维度 C_in=1，并扩展 batch 维度
          

            # 单步预测目标: [num_features]
            y_window = feature_data[start_idx + self.window_size]

            windows.append((x_window, y_window))
        return windows


from torch.utils.data import Dataset

class TimeWindowDataset(Dataset):
    def __init__(self,data_processor, config):
        self.data_processor = data_processor
        self.config = config
        

        self.windows = self.data_processor.generate_windows_normal()  # 生成滑动窗口数据对


    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x_window, y_window = self.windows[idx]
        
        instance_names = list(map(str, self.config.all_enum.keys()))
        return x_window, instance_names, y_window,