import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from module.TemporalEncoder import TemporalEncoder

class InstanceGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, nhead=4, dropout=0.1):
        """
        :param input_dim: 每个指标节点的特征维度
        :param hidden_dim: 中间隐层大小
        :param output_dim: 最终实例嵌入的维度
        :param num_layers: GCN层数
        :param nhead: Transformer的注意力头数
        :param dropout: Dropout比率
        """
        super(InstanceGraphEncoder, self).__init__()
        self.num_layers = num_layers
        
        # 使用Transformer进行时序编码
       

        # GCN层
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        """
        :param x: [num_nodes_total, input_dim]
        :param edge_index: [2, num_edges_total]
        :param batch: [num_nodes_total]，每个节点属于哪个子图的索引
        :return: [num_instances, output_dim]，每个实例的最终嵌入
        """
        # 使用Transformer进行时序编码
        #x = self.temporal_encoder(x)
        
        # GCN层处理
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
            
        x = self.convs[-1](x, edge_index)  # 最后一层不加激活

        # 以子图为单位做 pooling
        h_instance = global_mean_pool(x, batch)  # [num_instances, output_dim]

        return h_instance