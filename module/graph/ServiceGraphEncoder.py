import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class ServiceGraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, conv_type='gcn', heads=4):
        """
        :param input_dim: 输入实例向量的维度
        :param hidden_dim: 隐层维度
        :param output_dim: 输出实例向量的维度
        :param conv_type: 'gcn' or 'gat'
        :param heads: GAT的多头数量（仅在GAT时有效）
        """
        super(ServiceGraphEncoder, self).__init__()
        self.conv_type = conv_type.lower()

        if self.conv_type == 'gcn':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)
        elif self.conv_type == 'gat':
            self.conv1 = GATConv(input_dim, hidden_dim // heads, heads=heads)
            self.conv2 = GATConv(hidden_dim, output_dim // heads, heads=heads)
        else:
            raise ValueError(f"Unknown conv_type {conv_type}, should be 'gcn' or 'gat'")

    def forward(self, x, edge_index, edge_weight=None):
        """
        :param x: [num_instances, input_dim] 初始实例表示
        :param edge_index: [2, num_edges] 服务调用边
        :param edge_weight: [num_edges] 可选，边的权重
        :return: x': [num_instances, output_dim] 传播后的实例表示
        """
        if self.conv_type == 'gcn':
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = self.conv2(x, edge_index, edge_weight)
        elif self.conv_type == 'gat':
            x = F.elu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
        return x