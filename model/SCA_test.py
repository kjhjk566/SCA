import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from re import L
from tkinter import NO
from sympy import N
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.PodEmbedding import PodEmbedding  # 导入 PodEmbedding 模块
from module.SpatioTemporalBlock import SpatioTemporalBlock 
from module.NodeDecoder import NodeDecoder
from module.TemporalEncoder import TemporalEncoder  # 导入时序编码器
from model.net import gtnet


class SCA(nn.Module):
    def __init__(self,config, input_dim, hidden_dim, sca_hidden_dim,metric_num, temperature=0.5, lambda_reg=0.001,lambda_granger = 0.5,lambda_sparse = 1.0,device = None):
        super(SCA, self).__init__()
        # 初始化 PodEmbedding 模块
        self.transformer_encoder = TemporalEncoder(hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=2, nhead=2, dropout=0.1)
        self.pod_embedding = PodEmbedding(input_dim, hidden_dim)
        self.st_embedding = SpatioTemporalBlock(
            input_dim=input_dim,           # 每个节点的输入特征维度
            conv_out=8,             # 1D-CNN输出通道数
            lstm_hidden_dim=32,     # 1D-CNN隐藏通道数
            lstm_out_dim=64,        # 1D-CNN输出通道数
            conv_kernel=3,          # 1D-CNN卷积核大小
            hidden_dim=128,         # 时空特征隐藏维度
            time_length=5,         # 时间步长
            num_node=len(config.all_enum.keys()),            # 节点数量
            num_windows=6,          # 窗口数量（可选，部分代码未直接用到）
            moving_window=[3, 6],   # 多尺度窗口大小
            stride=[1, 2],          # 多尺度窗口滑动步长
            decay=0.9,              # 时空关联衰减系数
            pooling_choice='mean',  # 池化方式
            dropout=0.1             # Dropout概率
        )
        self.gtnet = gtnet(
            gcn_true=True, 
            buildA_true=True, 
            gcn_depth=2, #从2->10
            num_nodes=len(config.all_enum.keys()), 
            #num_nodes=metric_num,
            device=device, 
            dropout=0.3, 
            subgraph_size=20, 
            dilation_exponential=1, 
            conv_channels=32, 
            residual_channels=32, 
            skip_channels=64, 
            end_channels=64, 
            seq_length=20, 
            in_dim=64, 
            out_dim=12, 
            layers=10, #从3->30
            propalpha=0.05, 
            tanhalpha=3, 
            layer_norm_affline=True
            )
        self.config = config
        self.lambda_granger = lambda_granger
        self.lambda_sparse = lambda_sparse

        self.decoder = NodeDecoder(node_input_dim=64, hidden_dim=128, node_mapping=config.instance_metric_count_dict)  # 实例解码器


        # 其他 SCA 模块的初始化
        self.sca_layer = nn.Linear(hidden_dim, sca_hidden_dim)  # 示例层
        self.output_layer = nn.Linear(sca_hidden_dim, 1)  # 输出层

    def forward(self, x_window, y_window):
        """
        :param x_window: Tensor of shape [B, N, D, T]
            - B: batch size
            - N: number of instances (pods)
            - D: embedding dimension per metric
            - T: time steps (patch size)
        :param y_window: Tensor of shape [B, 1]，预测目标
        :return: output: Tensor of shape [B, 1], l1_reg: scalar regularization term
        """
        p = self.get_prediction(x_window)  # 获取预测结果
        # p = self.get_prediction(x_window)
        # 第二步：通过 SCA 的后续层处理
        loss = self.loss(p, y_window)  # 计算损失
        #print("loss:", loss.item())  # 输出损失值
        return loss
    def get_prediction(self, x_window):
        """
        :param metric_embeddings: Tensor of shape [B, M, D]
            - B: batch size
            - M: number of metrics per pod
            - D: embedding dimension per metric
        :return: output: Tensor of shape [B, 1]
        """
        # 第一步：通过 PodEmbedding 生成 pod-level 表示
        #print('x_window shape:', x_window.shape)
        metric_embedding = self.transformer_encoder(x_window)  # [B, N, D]
        #print("metric_embedding shape:", metric_embedding.shape)  # 输出形状检查

        pod_embedding= self.pod_embedding(x_window,self.config.instance_metric_count_dict)  # [B,N D]
        #print("pod_embedding:", pod_embedding.shape)  # 输出形状检查
        #st_in = pod_embedding.permute(0, 3, 1, 2) 
        #pod_embedding = pod_embedding.unsqueeze(1)
        print("pod_embedding:", pod_embedding.shape)  # 输出形状检查
        pod_embedding = pod_embedding.permute(0, 2, 1, 3)
        print("pod_embedding:", pod_embedding.shape)  # 输出形状检查
        

        st_out = self.gtnet.encode(pod_embedding)
        
        st_out = st_out.squeeze(-1)        # [16, 128, 541]
        st_out = st_out.permute(0, 2, 1)   # [16, 541, 128]
        #print("st_out shape:", st_out.shape)


        instance_names = list(self.config.instance_metric_count_dict.keys())
        predictions = self.decoder.forward(h_instances =st_out,metric_embeddings=metric_embedding, instance_names=instance_names)  # [B, total_metrics]
        #print("predictions:", predictions.shape)
       



        # return predictions, l1_reg
        return predictions

    def loss(self, predictions, y_window):
        """
        计算损失函数
        :param predictions: 模型预测结果
        :param y_window: 真实标签
        :return: 损失值
        """
        # 使用 MSELoss 计算损失
        loss_fn = nn.MSELoss()
        #Lg,Ls,L =  self.get_causal_gate_loss()
        loss = loss_fn(predictions, y_window)
        return loss

    # def get_causal_gate_loss(self):
    #     theta = self.st_embedding.MPNN1.causal_gate.theta
    #     m_val = torch.sigmoid(theta)
    #     extra_granger = 0 - (m_val * m_val).mean()
    #     extra_sparse = m_val.abs().sum()

    #     loss = self.lambda_granger * extra_granger + self.lambda_sparse * extra_sparse

    #     return extra_granger,extra_sparse,loss
    def get_causal_gate_loss(self):
        """
        extra_granger = -E[m^2]  （奖励有用边）
        extra_sparse  =  E[|m|]  （鼓励稀疏）
        - 去自环 (i==j)
        - 多层 MPNN（如 MPNN1/MPNN2）取平均
        """
        layers = []
        if hasattr(self.st_embedding, "MPNN1") and getattr(self.st_embedding.MPNN1, "causal_gate", None) is not None:
            layers.append(self.st_embedding.MPNN1.causal_gate.theta)
        if hasattr(self.st_embedding, "MPNN2") and getattr(self.st_embedding.MPNN2, "causal_gate", None) is not None:
            layers.append(self.st_embedding.MPNN2.causal_gate.theta)

        if not layers:
            zero = torch.tensor(0.0, device=next(self.parameters()).device)
            return zero, zero, zero  # extra_granger, extra_sparse, loss_reg

        gr_list, sp_list = [], []
        for theta in layers:
            m = torch.sigmoid(theta)                     # [N, N, L]
            N = m.shape[0]
            eye = torch.eye(N, device=m.device, dtype=m.dtype).unsqueeze(-1)
            m = m * (1.0 - eye)                          # 去自环
            gr_list.append(-(m * m).mean())              # -E[m^2]
            sp_list.append(m.abs().mean())               #  E[|m|]

        extra_granger = sum(gr_list) / len(gr_list)
        extra_sparse  = sum(sp_list) / len(sp_list)
        loss_reg = self.lambda_granger * extra_granger + self.lambda_sparse * extra_sparse
        return extra_granger, extra_sparse, loss_reg






    def get_ans(self, x_window,y_window):
        #self.get_prediction(x_window)
        # 计算指标级别和实例级别的误差
        metric_losses, instance_losses, total_loss = self.calculate_instance_loss(
            predictions=x_window, 
            y_true=y_window, 
            instance_metric_count_dict=self.config.instance_metric_count_dict,
            instance_names=list(self.config.all_enum.keys())
        )
        #返回误差最高的5个实例
        top5_indices = torch.topk(instance_losses, k=5, dim=1).indices
        top5_instances = [list(self.config.all_enum.keys())[i] for i in top5_indices[0]]

        return top5_instances


        # 在你的 SCA.py 文件中添加这个函数
    def calculate_metric_loss(self, x_window, y_window, loss_type='mse'):
        """
        计算指标级别的误差

        Args:
            predictions: [B, total_metrics] 所有指标的预测值
            y_true: [B, total_metrics] 所有指标的真实值

        Returns:
            metric_losses: [B, total_metrics] 每个指标的误差
        """
        predictions = self.get_prediction(x_window)
        y_true = y_window
        y_true = y_true.unsqueeze(0) 
        # 计算每个指标的误差
        if loss_type == 'mse':
            metric_losses = F.mse_loss(predictions, y_true, reduction='none')
        elif loss_type == 'mae':
            metric_losses = F.l1_loss(predictions, y_true, reduction='none')
        elif loss_type == 'rmse':
            metric_losses = torch.sqrt(F.mse_loss(predictions, y_true, reduction='none'))
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        return metric_losses