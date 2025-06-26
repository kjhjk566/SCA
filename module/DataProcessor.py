import pandas as pd
import torch
from torch_geometric.data import Batch
from module.graph.SubgraphManager import SubgraphManager
from module.graph.StaticCallGraphBuilder import StaticCallGraphBuilder
from collections import defaultdict


class DataProcessor:
    def __init__(self, df, config, window_size=10, stride=1):
        """
        
        :param config: 包含 all_enum 的配置对象
        :param window_size: 取多长时间作为一个窗口
        :param stride: 滑动窗口步长
        """
        
        self.config = config
        self.window_size = window_size
        self.stride = stride

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
        # print("DataFrame after reordering columns:")
        # print(self.df.head())
        #将self.df.head()保存为csv文件
        #self.df.to_csv('reordered_data.csv', index=False)
        #print("after reorder_columns:",self.df.shape)
        self.feature_columns = self.df.columns[1:]  # 除去timestamp列

        self.subgraph_manager = SubgraphManager(self.feature_columns, self.config)
        self.instance_metric_mapping = self.subgraph_manager.instance_to_metric_count_dict  # 存储实例到指标数量的映射
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
    def init_subgraph_manager(self):
        """初始化 SubgraphManager，分配指标到对应实例"""
        self.subgraph_manager = SubgraphManager(self.feature_columns, self.config)

    def generate_windows(self):
        """生成滑动时间窗口的数据对 (past, future)"""
        feature_data = torch.tensor(self.df.iloc[:, 1:].values, dtype=torch.float)  # [T, num_features]

        windows = []
        for start_idx in range(0, feature_data.size(0) - self.window_size, self.stride):
            x_window = feature_data[start_idx: start_idx + self.window_size]       # 输入窗口
            y_window = feature_data[start_idx + self.window_size]                  # 预测目标（单步预测）
            windows.append((x_window, y_window))
        return windows

    def create_batch(self, x_window):
        """
        根据一个输入窗口，构建子图Batch
        :param x_window: [window_size, num_features]
        :return: PyG Batch对象
        """
        self.subgraph_manager.build_initial_subgraphs(x_window)  # 构建新的子图
        self.instance_metric_mapping = self.subgraph_manager.instance_to_metric_count_dict  # 更新节点映射
        subgraph_list = [self.subgraph_manager.get_instance_graph(inst) for inst in self.subgraph_manager.list_all_instances()]
        # for i, subgraph in enumerate(subgraph_list):
        #     if subgraph is None:
        #         continue
            

        batch = Batch.from_data_list(subgraph_list)
        return batch

from torch.utils.data import Dataset

class TimeWindowDataset(Dataset):
    def __init__(self,data_processor, config):
        self.data_processor = data_processor
        self.config = config
        

        self.windows = self.data_processor.generate_windows()  # 生成滑动窗口数据对


    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x_window, y_window = self.windows[idx]
        batch = self.data_processor.create_batch(x_window)
        instance_names = list(map(str, self.config.all_enum.keys()))
        return batch, instance_names, y_window