import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LCNA(nn.Module):
    """
    极简版 Lag-aware Cross-Node Attention (L-CNA)

    输入:
        H:    [B, T, N, D]   - 每个时间步/节点的表示
        mask: [N, N] in {0,1}  先验拓扑掩码, 1 表示 i<-j 允许聚合

    逻辑:
      1) 对每个滞后 tau=1..K_eff:
         - 在 t ∈ [K_eff, T-1] 上, 计算 (i,t) 对 (j, t-τ) 的注意力并聚合得到 m^{(τ)}_{i,t}
      2) 用简单权重 λ_τ 对各滞后信息做加权合成: m_{i,t} = Σ_τ λ_τ m^{(τ)}_{i,t}
      3) 残差融合: \tilde h_{i,t} = h_{i,t} + W_m m_{i,t}  (仅对 t≥K_eff)
         早期 t<K_eff 直接拷贝输入 (直通)

    关键超参:
        K:               最大时滞
        combine:         'fixed' | 'learnable'
        decay_alpha:     'fixed' 时指数衰减系数 α, λ_τ ∝ exp(-α τ)
        out_dim:         输出维度, 默认与输入 D 相同；如不同则使用线性映射
        use_layernorm:   是否对融合结果做 LayerNorm

    属性(便于可解释 & 定位使用):
        last_alpha_per_tau:  list 长度 K_eff, 每个元素为注意力权重 [B, T-K_eff, N, N] (对 j softmax)
        last_lambda:         [K_eff] 的跨滞后权重 (固定或 softmax(θ) 后的数值)
    """
    def __init__(self, d_model: int, K: int = 5,
                 combine: str = "fixed",
                 decay_alpha: float = 0.25,
                 out_dim: int = None,
                 use_layernorm: bool = False):
        super().__init__()
        assert combine in ("fixed", "learnable")
        self.d = d_model
        self.K = K
        self.combine = combine
        self.decay_alpha = decay_alpha
        self.out_dim = out_dim if out_dim is not None else d_model
        self.use_ln = use_layernorm

        # 线性投影 (单头, 极简点积注意力)
        self.Wq = nn.Linear(self.d, self.d, bias=False)
        self.Wk = nn.Linear(self.d, self.d, bias=False)
        self.Wv = nn.Linear(self.d, self.d, bias=False)

        # 融合映射
        self.Wm = nn.Linear(self.d, self.out_dim, bias=True)
        if self.out_dim != self.d:
            self.Wproj_in = nn.Linear(self.d, self.out_dim, bias=False)
        else:
            self.Wproj_in = None

        self.ln = nn.LayerNorm(self.out_dim) if self.use_ln else nn.Identity()

        # learnable λ_τ (全局、与 i,t 无关)；'fixed' 时不使用
        if self.combine == "learnable":
            self.theta = nn.Parameter(torch.zeros(self.K))
        else:
            self.register_parameter("theta", None)

        # 缓存可解释输出
        self.last_alpha_per_tau = None  # list of [B, T-K_eff, N, N]
        self.last_lambda = None         # [K_eff]

    def _lag_weights(self, K_eff: int, device):
        if self.combine == "fixed":
            # 指数衰减: λ_τ ∝ exp(-α τ), τ=1..K_eff
            tau = torch.arange(1, K_eff + 1, device=device, dtype=torch.float32)
            w = torch.exp(-self.decay_alpha * tau)
            w = w / (w.sum() + 1e-12)
            return w  # [K_eff]
        else:
            # 可学习全局权重
            theta = self.theta[:K_eff]
            return torch.softmax(theta, dim=0)  # [K_eff]

    @torch.no_grad()
    def _make_neg_inf(self, like):
        return torch.finfo(like.dtype).min

    def forward(self, H: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            H:    [B, T, N, D]
            mask: [N, N], 0/1 邻接 (允许 i<-j)
        Returns:
            H_tilde: [B, T, N, out_dim]
        """
        B, T, N, D = H.shape
        device = H.device
        assert D == self.d, f"Input D={D} must match d_model={self.d}"

        # 有效最大滞后: 不能超过 T-1
        K_eff = min(self.K, max(1, T - 1))

        # 早期时间的直通输出准备
        if self.out_dim == D:
            H_tilde = H.clone()
        else:
            # 对所有时间步统一投影到 out_dim，避免前 K_eff 步维度不一致
            H_tilde = self.Wproj_in(H)

        if T <= 1:
            # 序列太短, 无滞后可用, 直接返回
            self.last_alpha_per_tau = []
            self.last_lambda = torch.ones(0, device=device)
            return self.ln(H_tilde)

        # 线性投影
        Q = self.Wq(H)  # [B,T,N,D]
        Kx = self.Wk(H) # [B,T,N,D]
        Vx = self.Wv(H) # [B,T,N,D]

        # 对齐时间范围: 仅在 t ∈ [K_eff, T-1] 计算跨滞后注意力 (保证每个 τ 都有 t-τ)
        Tr = T - K_eff
        # 收集每个滞后的消息与注意力
        msgs_per_tau = []
        alphas_per_tau = []

        # 预备广播的结构掩码
        M = (mask > 0).to(Q.dtype)            # [N,N]
        neg_inf = self._make_neg_inf(Q)

        # 归一化尺度
        scale = math.sqrt(D)

        # 统一取 Q 的对齐切片 (t = K_eff..T-1)
        Q_aligned = Q[:, K_eff:, :, :]        # [B,Tr,N,D]

        for tau in range(1, K_eff + 1):
            # keys/values 对齐到 t-τ
            K_tau = Kx[:, K_eff - tau:T - tau, :, :]   # [B,Tr,N,D]
            V_tau = Vx[:, K_eff - tau:T - tau, :, :]   # [B,Tr,N,D]

            # 计算 (i,j) 点积注意力: [B,Tr,N,N]
            logits = torch.einsum('btid,btjd->btij', Q_aligned, K_tau) / scale

            # 结构掩码: 仅允许 j ∈ N(i)
            # mask(i,j)=1 keep, 0 -> -inf
            mask_j = (M.view(1, 1, N, N) > 0)
            logits = torch.where(mask_j, logits, logits.new_full(logits.shape, neg_inf))

            # 对 j softmax
            alpha = torch.softmax(logits, dim=-1)      # [B,Tr,N,N]
            # 聚合消息: sum_j alpha * V_tau
            msg_tau = torch.einsum('btij,btjd->btid', alpha, V_tau)  # [B,Tr,N,D]

            msgs_per_tau.append(msg_tau)
            alphas_per_tau.append(alpha)

        # 跨滞后简单加权合成
        lam = self._lag_weights(K_eff, device=device)   # [K_eff]
        self.last_lambda = lam.detach()
        # [B,Tr,N,D]
        msg_all = sum(lam[tau-1] * msgs_per_tau[tau-1] for tau in range(1, K_eff + 1))

        # 残差融合仅作用于 t>=K_eff
        base = H_tilde[:, K_eff:, :, :]                 # [B,Tr,N,out_dim]
        fused = base + self.Wm(msg_all)                 # [B,Tr,N,out_dim]
        H_tilde[:, K_eff:, :, :] = fused

        # 记录可解释注意力 (不回传梯度)
        self.last_alpha_per_tau = [a.detach() for a in alphas_per_tau]

        return self.ln(H_tilde)