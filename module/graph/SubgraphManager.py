import torch
from torch_geometric.data import Data

class SubgraphManager:
    def __init__(self, feature_columns, config):
        """
        :param feature_columns: List[str], like ['cpu&cartservice-0', 'memory&cartservice-0', ...]
        :param config: Config object, contains all_enum mapping
        """
        self.feature_columns = feature_columns
        self.all_enum = config.all_enum
        self.instance_to_indices = self._build_instance_to_indices()

        self.instance_subgraphs = {}  # 存储每个实例的子图Data对象
        
        self.instance_to_metric_count_dict = {instance: len(indices) for instance, indices in self.instance_to_indices.items()}  # 存储实例到指标数量的映射

    def _build_instance_to_indices(self):
        """根据feature列名，建立instance到feature列index的映射"""
        instance_to_indices = {}
        for idx, col_name in enumerate(self.feature_columns):
            if '&' in col_name:
                _, instance = col_name.split('&')
                if instance not in instance_to_indices:
                    instance_to_indices[instance] = []
                instance_to_indices[instance].append(idx)
        return instance_to_indices

    def build_initial_subgraphs(self, features_tensor):
        """
        :param features_tensor: Tensor of shape [time_len, num_features], 特征矩阵
        这里是将原始的数据[time_len, num_features]，转换为每个实例的子图,其中x:[num_features,time]
        """
        for instance, indices in self.instance_to_indices.items():
            if instance not in self.all_enum:
                continue  # 只处理合法实例
            x = features_tensor[:, indices]  # [time_len, num_features_per_instance]
            x = x.transpose(0, 1).contiguous()  # [num_nodes, time_len]
            self.instance_to_metric_count_dict[instance] = len(indices)
            edge_index = self._build_fully_connected_edges(len(indices))
            #print(f"instance:{instance}, x.shape:{x.shape}, edge_index.shape:{edge_index.shape}")
            data = Data(x=x, edge_index=edge_index)
            self.instance_subgraphs[instance] = data

    def _build_fully_connected_edges(self, num_nodes):
        """生成全连接的edge_index"""
        src = []
        dst = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return edge_index

    def get_instance_graph(self, instance_name):
        """返回某个实例的子图Data对象"""
        return self.instance_subgraphs.get(instance_name, None)

    def list_all_instances(self):
        """返回当前有哪些实例子图"""
        return list(self.instance_subgraphs.keys())