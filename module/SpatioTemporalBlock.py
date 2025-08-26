"""
基于FCSTGNN的时空特征捕捉模块
参考: Fully-Connected Spatial-Temporal Graph Neural Network for Multivariate Time-Series Data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import OrderedDict


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 在GPU上计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(100.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Feature_extractor_1DCNN(nn.Module):
    """一维CNN特征提取器"""
    def __init__(self, input_channels, num_hidden, embedding_dimension, kernel_size=3, stride=1, dropout=0):
        super(Feature_extractor_1DCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(input_channels, num_hidden, kernel_size=kernel_size,
                      stride=stride, bias=False, padding=(kernel_size//2)),
            nn.BatchNorm1d(num_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(num_hidden, num_hidden*2, kernel_size=kernel_size, stride=1, bias=False, padding=2),
            nn.BatchNorm1d(num_hidden*2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(num_hidden*2, embedding_dimension, kernel_size=kernel_size, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(embedding_dimension),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

    def forward(self, x_in):
        x = torch.transpose(x_in, -1, -2)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x


class DotGraphConstruction(nn.Module):
    """点积图构建模块"""
    def __init__(self, input_dim):
        super().__init__()
        self.mapping = nn.Linear(input_dim, input_dim)

    def forward(self, node_features):
        node_features = self.mapping(node_features)
        bs, N, dimen = node_features.size()

        node_features_1 = torch.transpose(node_features, 1, 2)
        Adj = torch.bmm(node_features, node_features_1)

        eyes_like = torch.eye(N).repeat(bs, 1, 1).to(node_features.device)
        eyes_like_inf = eyes_like * 1e8
        Adj = F.leaky_relu(Adj - eyes_like_inf)
        Adj = F.softmax(Adj, dim=-1)
        Adj = Adj + eyes_like

        return Adj


class MPNN_mk_v2(nn.Module):
    """消息传递神经网络模块"""
    def __init__(self, input_dimension, output_dimension, k):
        super(MPNN_mk_v2, self).__init__()
        self.way_multi_field = 'sum'
        self.k = k
        theta = []
        for kk in range(self.k):
            theta.append(nn.Linear(input_dimension, output_dimension))
        self.theta = nn.ModuleList(theta)
        self.bn1 = nn.BatchNorm1d(output_dimension)

    def forward(self, X, A):
        GCN_output_ = []
        for kk in range(self.k):
            if kk == 0:
                A_ = A
            else:
                A_ = torch.bmm(A_, A)
            out_k = self.theta[kk](torch.bmm(A_, X))
            GCN_output_.append(out_k)

        if self.way_multi_field == 'cat':
            GCN_output_ = torch.cat(GCN_output_, -1)
        elif self.way_multi_field == 'sum':
            GCN_output_ = sum(GCN_output_)

        GCN_output_ = torch.transpose(GCN_output_, -1, -2)
        GCN_output_ = self.bn1(GCN_output_)
        GCN_output_ = torch.transpose(GCN_output_, -1, -2)

        return F.leaky_relu(GCN_output_)


def conv_graphst(input, time_window_size, stride):
    """时空卷积操作"""
   
    bs, time_length, num_sensors, feature_dim = input.size()
    x_ = torch.transpose(input, 1, 3)
    y_ = F.unfold(x_, (num_sensors, time_window_size), stride=stride)
    y_ = torch.reshape(y_, [bs, feature_dim, num_sensors, time_window_size, -1])
    y_ = torch.transpose(y_, 1, -1)
    return y_


def mask_matrix(num_node, time_length, decay_rate):
    #print("Creating mask matrix..., num_node:", num_node, "time_length:", time_length, "decay_rate:", decay_rate)
    """创建掩码矩阵"""
    Adj = torch.ones(num_node * time_length, num_node * time_length)
    for i in range(time_length):
        v = 0
        for r_i in range(i, time_length):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (r_i) * num_node
            idx_e_col = (r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] * (decay_rate ** v)
            v = v + 1
        v = 0
        for r_i in range(i + 1):
            idx_s_row = i * num_node
            idx_e_row = (i + 1) * num_node
            idx_s_col = (i - r_i) * num_node
            idx_e_col = (i - r_i + 1) * num_node
            Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] = Adj[idx_s_row:idx_e_row, idx_s_col:idx_e_col] * (decay_rate ** v)
            v = v + 1
    return Adj


class CausalGating(nn.Module):
    def __init__(self, num_sensors, time_window_size, p0=0.1):
        super().__init__()
        # Initialize theta parameters for gating with shape [num_sensors, num_sensors, time_window_size]
        self.num_sensors = num_sensors
        self.time_window_size = time_window_size
        self.theta = nn.Parameter(torch.randn(num_sensors, num_sensors, time_window_size) * 0.01 + math.log(p0 / (1 - p0)))

    def forward(self, device=None, dtype=None):
        # Sigmoid activation to get gating matrix of shape [num_sensors, num_sensors, time_window_size]
        gate = torch.sigmoid(self.theta)
        # Reshape to [T*N, T*N] for gating adjacency matrix
        # Construct the big gating matrix G for full adjacency with temporal dimension
        N = self.num_sensors
        T = self.time_window_size
        G = torch.zeros(T * N, T * N, device=device, dtype=dtype)
        for lag in range(T):
            diag = gate[:, :, lag]
            for t in range(T - lag):
                i = t + lag
                j = t
                G[i * N:(i + 1) * N, j * N:(j + 1) * N] = diag
        return G


class GraphConvPoolMPNN_block(nn.Module):
    """图卷积池化MPNN模块"""
    def __init__(self, input_dim, output_dim, num_sensors, time_length, time_window_size, stride, decay, pool_choice, enable_causal_gating: bool = False, gate_init_p: float = 0.1, semi_dynamic: bool = True, ema_beta: float = 0.8):
        super(GraphConvPoolMPNN_block, self).__init__()
        self.time_window_size = time_window_size
        self.stride = stride
        self.output_dim = output_dim

        self.graph_construction = DotGraphConstruction(input_dim)
        self.BN = nn.BatchNorm1d(input_dim)
        self.MPNN = MPNN_mk_v2(input_dim, output_dim, k=1)
        print('num_sensors:', num_sensors, 'time_window_size:', time_window_size)
        
        self.pre_relation = mask_matrix(num_sensors, time_window_size, decay)
        self.pool_choice = pool_choice

        # === 因果门控 ===
        self.enable_causal_gating = enable_causal_gating
        if self.enable_causal_gating:
            self.causal_gate = CausalGating(num_sensors=num_sensors,
                                            time_window_size=time_window_size,
                                            p0=gate_init_p)
        else:
            self.causal_gate = None
        # === 半动态 (强度随窗, 结构固定) 的 EMA 平滑 ===
        self.semi_dynamic = semi_dynamic
        self.ema_beta = ema_beta
        self.register_buffer('ema_adj', None, persistent=False)  # 将在第一次前向时初始化
    

    def forward(self, input):
        # 输入尺寸 (bs, time_length, num_nodes, input_dim)
        # 输出尺寸 (bs, output_node_t, output_node_s, output_dim)
        print(f"Input shape: {input.shape}")
        input_con = conv_graphst(input, self.time_window_size, self.stride)
        print(f"After conv_graphst: {input_con.shape}")
        
        # input_con 尺寸 (bs, num_windows, num_sensors, time_window_size, feature_dim)
        bs, num_windows, num_sensors, time_window_size, feature_dim = input_con.size()
        print(f"Extracted dimensions: bs={bs}, num_windows={num_windows}, num_sensors={num_sensors}, time_window_size={time_window_size}")
        input_con_ = torch.transpose(input_con, 2, 3)
        input_con_ = torch.reshape(input_con_, [bs * num_windows, time_window_size * num_sensors, feature_dim])
        print("input_con_.shape:", input_con_.shape)
        A_input = self.graph_construction(input_con_)
        print("A_input.shape:", A_input.shape)
        print("self.pre_relation.shape:", self.pre_relation.shape)
        A_input = A_input * self.pre_relation.to(A_input.device)

        # === 叠加因果门控: 按 (j->i, lag) 的可学习门控过滤全连接边 ===
        if self.enable_causal_gating and self.causal_gate is not None:
            G = self.causal_gate(device=A_input.device, dtype=A_input.dtype)  # [T*N, T*N]
            A_input = A_input * G.unsqueeze(0)

        # === 半动态图的 EMA 平滑 (仅对强度进行平滑, 结构由门控固定) ===
        if self.semi_dynamic:
            with torch.no_grad():
                A_mean = A_input.detach().mean(dim=0, keepdim=True)  # [1, T*N, T*N]
                if self.ema_adj is None:
                    self.ema_adj = A_mean.clone()
                else:
                    self.ema_adj.mul_(self.ema_beta).add_(A_mean, alpha=(1.0 - self.ema_beta))
            # 扩展回当前 batch 大小
            A_eff = self.ema_adj.expand(A_input.shape[0], -1, -1)
        else:
            A_eff = A_input

        input_con_ = torch.transpose(input_con_, -1, -2)
        input_con_ = self.BN(input_con_)
        input_con_ = torch.transpose(input_con_, -1, -2)
        X_output = self.MPNN(input_con_, A_eff)

        X_output = torch.reshape(X_output, [bs, num_windows, time_window_size, num_sensors, self.output_dim])

        if self.pool_choice == 'mean':
            X_output = torch.mean(X_output, 2)
        elif self.pool_choice == 'max':
            X_output, ind = torch.max(X_output, 2)
        else:
            raise ValueError('Unsupported pooling choice')

        return X_output


class SpatioTemporalBlock(nn.Module):
    """
    基于FCSTGNN的时空特征捕捉模块
    
    主要功能：
    1. 时序特征提取: 使用1D CNN提取时序模式
    2. 空间图构建: 基于节点特征构建动态图
    3. 时空关联建模: 通过FC-STGNN捕捉不同时间戳不同传感器之间的关联
    4. 位置编码: 为时序数据添加位置信息
    """
    
    def __init__(self, input_dim, conv_out, lstm_hidden_dim, lstm_out_dim, conv_kernel, 
                 hidden_dim, time_length, num_node, num_windows, moving_window, 
                 stride, decay, pooling_choice, dropout=0.1):
        super(SpatioTemporalBlock, self).__init__()
        
        # 存储参数用于后续使用
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # 非线性映射层 - 1D CNN特征提取
        self.nonlin_map = Feature_extractor_1DCNN(
            1, lstm_hidden_dim, lstm_out_dim, kernel_size=conv_kernel, dropout=dropout
        )
        
        # 特征投影层
        self.nonlin_map2 = nn.Sequential(
            #nn.Linear(lstm_out_dim * conv_out, 2 * hidden_dim),
            nn.Linear(256, 2 * hidden_dim),
            nn.BatchNorm1d(2 * hidden_dim)
        )

        # 位置编码
        self.positional_encoding = PositionalEncoding(2 * hidden_dim, 0.1, max_len=5000)

        # MPNN模块 - 多尺度时空特征捕捉
        self.MPNN1 = GraphConvPoolMPNN_block(
            2 * hidden_dim, hidden_dim, num_node, time_length, 
            time_window_size=moving_window[0], stride=stride[0], 
            decay=decay, pool_choice=pooling_choice,
            enable_causal_gating=True, gate_init_p=0.1,
            semi_dynamic=True, ema_beta=0.8
        )
        self.MPNN2 = GraphConvPoolMPNN_block(
            2 * hidden_dim, hidden_dim, num_node, time_length, 
            time_window_size=moving_window[1], stride=stride[1], 
            decay=decay, pool_choice=pooling_choice,
            enable_causal_gating=True, gate_init_p=0.1,
            semi_dynamic=True, ema_beta=0.8
        )

        # 全连接分类器 - 动态计算输入维度
        # 注意：实际输入维度需要根据MPNN输出动态计算
        self.fc_input_dim = None  # 延迟初始化
        self.fc = None

    def forward(self, X):
        """
        前向传播
        Args:
            X: 输入张量 [bs, tlen, num_node, dimension]
        
        Returns:
            features: 时空特征 [bs, hidden_dim]
        """
        bs, tlen, num_node, dimension = X.size()

        # 1. 图生成 - 时序特征提取
        A_input = torch.reshape(X, [bs * tlen * num_node, dimension, 1])
        #print("A_input shape:", A_input.shape)  # 输出形状检查
        A_input_ = self.nonlin_map(A_input)
        #print("A_input_ shape after nonlin_map:", A_input_.shape)  # 输出形状检查
        A_input_ = torch.reshape(A_input_, [bs * tlen * num_node, -1])
        #print("A_input_ shape after reshape:", A_input_.shape)  # 输出形状检查
        A_input_ = self.nonlin_map2(A_input_)
        #print("A_input_ shape after nonlin_map2:", A_input_.shape)  # 输出形状检查
        A_input_ = torch.reshape(A_input_, [bs, tlen, num_node, -1])
        #print("A_input_ final shape:", A_input_.shape)  # 输出形状检查

        # 2. 位置编码
        X_ = torch.reshape(A_input_, [bs, tlen, num_node, -1])
        X_ = torch.transpose(X_, 1, 2)
        X_ = torch.reshape(X_, [bs * num_node, tlen, -1])
        X_ = self.positional_encoding(X_)
        X_ = torch.reshape(X_, [bs, num_node, tlen, -1])
        X_ = torch.transpose(X_, 1, 2)
        A_input_ = X_
        #print("A_input_ after positional encoding:", A_input_.shape)  # 输出形状检查

        # 3. 多尺度MPNN处理
        MPNN_output1 = self.MPNN1(A_input_)
        #MPNN_output2 = self.MPNN2(A_input_)
        #print("MPNN_output1 shape:", MPNN_output1.shape)  # 输出形状检查
        #print("MPNN_output2 shape:", MPNN_output2.shape)

        # 4. 特征融合
        features = torch.mean(MPNN_output1, dim=1)  # 维度 [bs, num_nodes, hidden_dim]
        #features1 = torch.mean(MPNN_output1, dim=1)  # 维度 [bs, num_nodes, hidden_dim]

        #features2 = torch.mean(MPNN_output2, dim=1)  # 同上
        #features = torch.cat([features1, features2], dim=-1)  # [bs, num_nodes, hidden_dim * 2]
        #print("features shape after concat:", features.shape)  # 输出形状检查

        # 5. 动态初始化全连接层
        if self.fc is None:
            input_dim = features.size(-1)
            self.fc = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(input_dim, self.hidden_dim * 2)),
                ('relu1', nn.ReLU(inplace=True)),
                ('dropout1', nn.Dropout(self.dropout)),
                ('fc2', nn.Linear(self.hidden_dim * 2, self.hidden_dim)),
            ])).to(features.device)

        # 6. 最终特征映射
        features = self.fc(features)  # [bs, num_nodes, hidden_dim]
        #print("Final features shape:", features.shape)  # 输出形状检查

        return features

    def extract_spatiotemporal_features(self, X):
        """
        提取时空特征（不进行最终分类）
        """
        return self.forward(X)

    def freeze_gates(self):
        """冻结因果门控参数 theta, 用于 B 方案 (结构固定, 强度随窗)"""
        for b in [getattr(self, "MPNN1", None), getattr(self, "MPNN2", None)]:
            if b is not None and getattr(b, "causal_gate", None) is not None:
                for p in b.causal_gate.parameters():
                    p.requires_grad = False

    @torch.no_grad()
    def export_static_scores(self, delta: float = 0.8):
        """
        导出结构基准得分 S_struct (不含相似度项), 形状:
        - 'MPNN1': [N, N, T1], 'MPNN2': [N, N, T2]
        用户可在外部再乘上校准期望相似度 E[sim] 以得到完整 S。
        """
        out = {}
        if hasattr(self, "MPNN1") and getattr(self.MPNN1, "causal_gate", None) is not None:
            m = torch.sigmoid(self.MPNN1.causal_gate.theta)  # [N,N,T1]
            L = m.shape[-1]
            gamma = (torch.tensor(delta, device=m.device, dtype=m.dtype) ** torch.arange(L, device=m.device, dtype=m.dtype)).view(1,1,L)
            out["MPNN1"] = m * gamma
        if hasattr(self, "MPNN2") and getattr(self.MPNN2, "causal_gate", None) is not None:
            m = torch.sigmoid(self.MPNN2.causal_gate.theta)  # [N,N,T2]
            L = m.shape[-1]
            gamma = (torch.tensor(delta, device=m.device, dtype=m.dtype) ** torch.arange(L, device=m.device, dtype=m.dtype)).view(1,1,L)
            out["MPNN2"] = m * gamma
        return out
