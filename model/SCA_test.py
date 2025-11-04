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

# --------- MultiLagCausalLearner & LagCrossNodeAttention ---------
import math

class MultiLagCausalLearner(nn.Module):
    """
    Learn a set of multi-lag causal strength matrices {A^(tau)} with low-rank parameterization
    and prior mask. Produces static per-batch A for efficiency. Shapes:
      - mask: [N, N] in {0,1}
      - returns A_stack: [K, N, N] with non-neg weights masked by prior.
    """
    def __init__(self, num_nodes: int, num_lags: int = 5, rank: int = 8, mask: torch.Tensor = None, init_scale: float = 0.1, device=None):
        super().__init__()
        self.N = num_nodes
        self.K = num_lags
        self.r = rank
        self.device = device
        if mask is None:
            mask = torch.ones(num_nodes, num_nodes, dtype=torch.float32)
        self.register_buffer("mask", mask.float())
        # Low-rank factors
        self.U = nn.Parameter(torch.randn(self.N, self.r) * init_scale)
        self.V = nn.Parameter(torch.randn(self.N, self.r) * init_scale)
        # Per-lag scale
        self.s_lag = nn.Parameter(torch.randn(self.K) * 0.01)
        # Optional structural bias per edge (used by attention as additive bias if needed)
        self.B_struct = nn.Parameter(torch.zeros(self.N, self.N))
        # cache for last A
        self._last_A = None

    def forward(self, H: torch.Tensor):
        """
        H is not directly used here (kept for interface parity & future dynamic variants).
        Returns:
            A_stack: [K, N, N] masked, non-negative.
        """
        UV = torch.matmul(self.U, self.V.t())  # [N,N]
        base = UV  # can add self.B_struct if preferred in future
        base = torch.sigmoid(base)  # [0,1]
        # broadcast per-lag scale via exp to keep positivity influence
        scales = torch.relu(self.s_lag).view(self.K, 1, 1) + 1e-6  # [K,1,1]
        A_stack = base.unsqueeze(0) * scales  # [K,N,N]
        # apply mask strictly
        A_stack = A_stack * self.mask.unsqueeze(0)
        # zero diagonal to avoid self-loops in same-time slice
        eye = torch.eye(self.N, device=A_stack.device).view(1, self.N, self.N)
        A_stack = A_stack * (1.0 - eye)
        self._last_A = A_stack.detach()
        return A_stack

    # --- Regularizers & utils for training integration ---
    def l1_learned(self):
        # encourage sparsity on masked domain
        if self._last_A is None:
            return torch.tensor(0.0, device=self.U.device)
        return (self._last_A.abs().sum() / (self.N * self.N * max(1, self.K)))

    def l1_residual_vs_prior(self):
        # if a dense prior is provided, penalize outside prior (already masked here)
        return torch.tensor(0.0, device=self.U.device)

    def dag_penalty(self):
        # For lagged graphs X(t-τ)->X(t), acyclicity across same-time is naturally avoided.
        # Return zero to keep compatibility with existing code.
        return torch.tensor(0.0, device=self.U.device)

    def get_last_graph(self):
        # [K,N,N]
        if self._last_A is None:
            # synthesize a current estimate if forward hasn't been called
            with torch.no_grad():
                UV = torch.matmul(self.U, self.V.t())
                base = torch.sigmoid(UV)
                scales = torch.relu(self.s_lag).view(self.K, 1, 1) + 1e-6
                A_stack = base.unsqueeze(0) * scales
                A_stack = A_stack * self.mask.unsqueeze(0)
                eye = torch.eye(self.N, device=self.U.device).view(1, self.N, self.N)
                A_stack = A_stack * (1.0 - eye)
            return A_stack
        return self._last_A


class LagCrossNodeAttention(nn.Module):
    """
    Lag-aware Cross-Node Attention (L-CNA)
    Inputs:
        H: [B, T, N, Din]         - time×node embeddings
        A_stack: [K, N, N]        - multi-lag causal strengths (non-negative)
        mask: [N, N] in {0,1}     - prior connectivity mask
    Returns:
        H_out: [B, T, N, Dout]    - updated representations per (t,i)
    Side effect:
        self.last_align_loss: scalar alignment loss KL( attention || softmax(log A) )
    """
    def __init__(self, d_in: int, d_out: int = 128, num_heads: int = 4, num_lags: int = 5, dropout: float = 0.0, device=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_in = d_in
        self.d_out = d_out
        self.h = num_heads
        self.K = num_lags
        self.dh = d_out // num_heads
        self.device = device

        self.Wq = nn.Linear(d_in, self.dh * self.h, bias=False)
        self.Wk = nn.Linear(d_in, self.dh * self.h, bias=False)
        self.Wv = nn.Linear(d_in, self.dh * self.h, bias=False)
        self.Wo = nn.Linear(self.dh * self.h, d_out, bias=False)

        # relative time encodings per lag
        self.rho = nn.Parameter(torch.randn(self.K, self.dh) * 0.01)
        # lag bias (encourage shorter lags by default)
        self.beta = nn.Parameter(torch.linspace(-0.25, -0.25 * self.K, steps=self.K))
        # gate & fusion
        self.gate_w = nn.Linear(d_in + d_out, 1)
        self.ffn = nn.Sequential(
            nn.Linear(d_in + d_out, 2 * d_out),
            nn.ReLU(inplace=True),
            nn.Linear(2 * d_out, d_out),
        )
        self.ln = nn.LayerNorm(d_out)

        self.dropout = nn.Dropout(dropout)
        self.last_align_loss = torch.tensor(0.0)

    def forward(self, H: torch.Tensor, A_stack: torch.Tensor, mask: torch.Tensor):
        """
        Vectorized computation across lags via time shifting.
        """
        B, T, N, Din = H.shape
        device = H.device
        K = min(self.K, T - 1)  # need at least one lagged step
        if K <= 0:
            # not enough time context; just project to d_out
            proj = self.Wo(self.Wv(H.view(B * T * N, Din)).view(B, T, N, self.h, self.dh).flatten(-2))
            self.last_align_loss = torch.tensor(0.0, device=device)
            return proj

        # projections
        Q = self.Wq(H).view(B, T, N, self.h, self.dh)         # [B,T,N,h,dh]
        Kx = self.Wk(H).view(B, T, N, self.h, self.dh)        # [B,T,N,h,dh]
        Vx = self.Wv(H).view(B, T, N, self.h, self.dh)        # [B,T,N,h,dh]

        # prepare target soft labels from A_stack over (tau,j)
        eps = 1e-8
        A = A_stack[:K].clamp_min(eps)                        # [K,N,N]
        # mask zero-entries remain zero; build normalized distribution over (tau,j)
        A_flat = A.view(K, N, N)
        A_norm = A_flat / (A_flat.sum(dim=(0,2), keepdim=True) + eps)  # normalize per receiver i across (tau,j)
        # shape to [K,N,N] still; will gather later

        # adjacency mask
        M = (mask > 0).to(H.dtype)                            # [N,N]
        neg_inf = -1e9

        # containers for outputs
        H_out = H.new_zeros(B, T, N, self.d_out)
        align_losses = []

        # compute for each lag in vectorized manner via time shift
        # We'll accumulate per-lag neighbor-softmax, then compute lag-softmax.
        per_tau_msgs = []   # list of [B, T-K, N, h, dh]
        per_tau_logits_mean = []  # for lag-softmax: [B, T-K, N]

        # valid time range for aggregation: t in [K, T-1]
        Tr = T - K

        for tau in range(1, K + 1):
            # align time dims: Q at t=tau..T-1 and K,V at t'=0..T-1-tau
            Q_tau = Q[:, tau:, :, :, :]                       # [B,Tr,N,h,dh]
            K_tau = Kx[:, :-tau, :, :, :] + self.rho[tau - 1] # [B,Tr,N,h,dh]
            V_tau = Vx[:, :-tau, :, :, :]                     # [B,Tr,N,h,dh]

            # attention logits: einsum over dh
            # logits[b,t,i,h,j] = <Q_tau[b,t,i,h,:], K_tau[b,t,j,h,:]>
            logits = torch.einsum('b t i h d, b t j h d -> b t i h j', Q_tau, K_tau) / math.sqrt(self.dh)

            # add structure/lag biases
            logits = logits + self.beta[tau - 1]              # scalar add
            # add causal bias: log A_ij^(tau)
            A_tau = (A[tau - 1] + eps).log()                  # [N,N]
            logits = logits + A_tau.view(1, 1, N, 1, N)

            # apply adjacency mask
            mask_j = (M.view(1, 1, 1, 1, N) > 0)
            logits = torch.where(mask_j, logits, logits.new_full(logits.shape, neg_inf))

            # neighbor softmax over j
            omega = torch.softmax(logits, dim=-1)             # [B,Tr,i,h,j]
            # mean logits over (h,j) for lag-competition
            lag_score = logits.masked_fill(~mask_j, 0.0).mean(dim=(-1, -2))  # [B,Tr,i]
            per_tau_logits_mean.append(lag_score)
            # message: sum_j omega * V
            msg = torch.einsum('b t i h j, b t j h d -> b t i h d', omega, V_tau)  # [B,Tr,i,h,dh]
            per_tau_msgs.append(msg)

        # stack over tau
        msgs = torch.stack(per_tau_msgs, dim=2)                   # [B,Tr,i, K, h, dh]
        lag_scores = torch.stack(per_tau_logits_mean, dim=-1)     # [B,Tr,i, K]
        # lag softmax over tau
        pi = torch.softmax(lag_scores, dim=-1)                    # [B,Tr,i,K]
        # combine msgs with pi
        pi_exp = pi.view(B, Tr, N, self.K, 1, 1)
        msg_all = (msgs * pi_exp).sum(dim=3)                      # [B,Tr,i,h,dh]
        msg_all = msg_all.contiguous().view(B, Tr, N, self.h * self.dh)
        msg_all = self.Wo(msg_all)                                # [B,Tr,N,d_out]
        msg_all = self.dropout(msg_all)

        # gate + fuse with local representation at aligned times (t=tau..T-1 uses base at those times)
        base = H[:, K:, :, :]                                     # [B,Tr,N,Din]
        z = torch.sigmoid(self.gate_w(torch.cat([base, msg_all], dim=-1)))  # [B,Tr,N,1]
        fused = self.ln(base + self.ffn(torch.cat([base, z * msg_all], dim=-1)))  # [B,Tr,N,d_out]

        # write output: keep first K steps as passthrough (or simple projection)
        if self.d_in == self.d_out:
            H_out[:, :K, :, :] = H[:, :K, :, :]
        else:
            # lightweight projection for first K steps
            proj_first = self.Wo(self.Wv(H[:, :K, :, :]).view(B, K, N, self.h, self.dh).flatten(-2))
            H_out[:, :K, :, :] = proj_first
        H_out[:, K:, :, :] = fused

        # --- alignment loss ---
        # attention distribution over (tau,j): a_ij^(tau) = pi_i^(tau) * E_h[omega_ij^(tau)]
        # Build E_h[omega] per tau: [B,Tr,i,j]
        with torch.no_grad():
            # target over (tau,j) from A_stack normalized per receiver i
            # A_norm[K,N,N] -> expand to [1,1,K,N,N] then gather as needed
            pass
        # Compute average attention distribution across heads
        # For efficiency, recompute neighbor weights mean using stored logits:
        # We'll approximate using softmax(lag_score) for pi and uniform omega mass aligned by A mask.
        # Simpler: compare pi to marginal over tau from A (summing over j), and skip omega alignment.
        A_tau_marginal = (A / (A.sum(dim=(0, 2), keepdim=True) + eps)).sum(dim=-1)  # [K,N]
        # broadcast to [B,Tr,N,K]
        A_pi_target = A_tau_marginal.t().view(1, 1, N, K)        # [1,1,N,K]
        align = torch.nn.functional.kl_div(
            (pi + eps).log(), A_pi_target.expand_as(pi), reduction='batchmean'
        )
        self.last_align_loss = align

        return H_out


class SCA(nn.Module):
    def __init__(self,config, input_dim, hidden_dim, sca_hidden_dim,metric_num, temperature=0.5, lambda_reg=0.001,lambda_granger = 0.5,lambda_sparse = 1.0,device = None,adj = None):
       
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
            predefined_A = adj,
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

        # ---- L-CNA & Multi-Lag Causal Learner wiring ----
        num_nodes = len(config.all_enum.keys())
        self.lag_K = 5
        self.num_heads = 4
        # infer pod-level embedding dim from previous settings
        self.pod_hidden_dim = hidden_dim
        self.d_model_out = 128

        # build prior mask from adj (fallback to dense if None)
        if adj is not None:
            prior_mask = torch.tensor(adj, dtype=torch.float32, device=device)
            prior_mask = (prior_mask > 0).float()
        else:
            prior_mask = torch.ones(num_nodes, num_nodes, dtype=torch.float32, device=device)
        # forbid self loops on same-time
        eye = torch.eye(num_nodes, device=device)
        prior_mask = prior_mask * (1.0 - eye)
        self.register_buffer("prior_mask", prior_mask)

        self.causal_learner = MultiLagCausalLearner(
            num_nodes=num_nodes,
            num_lags=self.lag_K,
            rank=8,
            mask=self.prior_mask,
            device=device
        )
        self.lcna = LagCrossNodeAttention(
            d_in=self.pod_hidden_dim,
            d_out=self.d_model_out,
            num_heads=self.num_heads,
            num_lags=self.lag_K,
            dropout=0.1,
            device=device
        )
        self.lambda_align = 0.1

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
        print('x_window shape:', x_window.shape)
        metric_embedding = self.transformer_encoder(x_window)  # [B, N, D]
        print("metric_embedding shape:", metric_embedding.shape)  # 输出形状检查

        pod_embedding= self.pod_embedding(x_window,self.config.instance_metric_count_dict)  # [B,N D]
        print("pod_embedding:", pod_embedding.shape)  # 输出形状检查
        #st_in = pod_embedding.permute(0, 3, 1, 2) 
        #pod_embedding = pod_embedding.unsqueeze(1)
        
        pod_embedding = pod_embedding.permute(0, 2, 1, 3)
        print("pod_embedding2:", pod_embedding.shape)  # 输出形状检查
        

        # ---- L-CNA forward with multi-lag causal learner ----
        # pod_embedding is [B, T, N, D], already permuted above
        A_list = self.causal_learner(pod_embedding)             # [K, N, N]
        st_time = self.lcna(pod_embedding, A_list, self.prior_mask)  # [B, T, N, d_model_out]
        st_out = st_time[:, -1, :, :]                           # take last time step as instance representation [B, N, d_model_out]

        instance_names = list(self.config.instance_metric_count_dict.keys())
        predictions = self.decoder.forward(h_instances=st_out, metric_embeddings=metric_embedding, instance_names=instance_names)  # [B, total_metrics]
        return predictions

    def loss(self, predictions, y_window):
        """
        组合任务损失 + 因果图正则（可开关）
        """
        device = predictions.device
        # 1) 主任务
        task_loss = F.mse_loss(predictions, y_window)

        # Alignment loss from L-CNA (if computed)
        align_loss = getattr(self.lcna, "last_align_loss", torch.tensor(0.0, device=device))

        reg_l1 = torch.tensor(0.0, device=device)
        reg_prior = torch.tensor(0.0, device=device)
        reg_dag = torch.tensor(0.0, device=device)
        reg_ent = torch.tensor(0.0, device=device)

        # Prefer new causal learner if present
        if hasattr(self, "causal_learner") and self.causal_learner is not None:
            cl = self.causal_learner
            if hasattr(cl, "l1_learned"):
                reg_l1 = cl.l1_learned()
            if hasattr(cl, "dag_penalty"):
                reg_dag = cl.dag_penalty()
            if hasattr(cl, "get_last_graph"):
                A_last = cl.get_last_graph()             # [L,N,N]
                eps = 1e-12
                P = A_last.clamp_min(eps)
                row_ent = -(P * P.log()).sum(dim=-1)     # [L,N]
                reg_ent = row_ent.mean()
        elif hasattr(self.gtnet, "causal_learner"):
            cl = self.gtnet.causal_learner
            if hasattr(cl, "l1_learned"):
                reg_l1 = cl.l1_learned()
            if hasattr(cl, "dag_penalty"):
                reg_dag = cl.dag_penalty()
            if hasattr(cl, "get_last_graph"):
                A_last = cl.get_last_graph()
                eps = 1e-12
                P = A_last.clamp_min(eps)
                row_ent = -(P * P.log()).sum(dim=-1)
                reg_ent = row_ent.mean()

        lambda_l1    = 3e-4
        lambda_prior = 0.0
        lambda_dag   = 1e-2
        lambda_ent   = 1e-3
        lambda_align = getattr(self, "lambda_align", 0.1)

        loss = (
            task_loss
            + lambda_align * align_loss
            + lambda_l1 * reg_l1
            + lambda_prior * reg_prior
            + lambda_dag * (reg_dag ** 2)
            + lambda_ent * reg_ent
        )
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
            raise ValueError(f"Unsupported loss type: {loss_type}")

        return metric_losses