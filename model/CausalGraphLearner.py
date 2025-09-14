# -*- coding: utf-8 -*-
"""
Multi-Lag Causal Graph Learner
学习一组有向、非负、行稀疏、行归一化的多时滞邻接矩阵 {A^(1),...,A^(L)}。
A^(tau)[i,j] 表示“节点 j 在滞后 tau 步对节点 i 的直接影响强度”（列为来源、行为指向）。
可选：与先验图混合（调用链/部署关系）、候选掩膜限制、每滞后单独混合系数。
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------- 一些小工具 -----------------

def _row_normalize(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """按行归一化，使每行求和为 1；A >= 0 时可被解释为“入边权重分布”"""
    denom = A.sum(dim=-1, keepdim=True).clamp_min(eps)
    return A / denom

def _row_topk_normalize(A: torch.Tensor, k: int, eps: float = 1e-12) -> torch.Tensor:
    """每行保留 top-k，其余置零，然后按行归一化"""
    N = A.size(0)
    if k >= N:
        return _row_normalize(A, eps)
    topv, topi = torch.topk(A, k=k, dim=1)
    mask = torch.zeros_like(A)
    mask.scatter_(1, topi, 1.0)
    A = A * mask
    return _row_normalize(A, eps)

def _zero_diag(A: torch.Tensor) -> torch.Tensor:
    """清零对角线（无自环）"""
    return A - torch.diag_embed(torch.diag(A))

def _expand_lags(X: torch.Tensor, L: int) -> torch.Tensor:
    """若 X 为 [N,N]，复制到 [L,N,N]；若已为 [L,N,N] 直接返回。"""
    if X.dim() == 2:
        return X.unsqueeze(0).repeat(L, 1, 1)
    assert X.dim() == 3 and X.size(0) == L, "shape must be [N,N] or [L,N,N]"
    return X

def notears_dag_penalty(A: torch.Tensor) -> torch.Tensor:
    r"""NOTEARS 连续无环约束：\( h(A)=\mathrm{tr}(\exp(A\odot A)) - N \)"""
    expm = torch.matrix_exp(A * A)
    return torch.trace(expm) - A.size(0)


# ------------- 多时滞因果图学习器主体 --------------

class CausalGraphLearner(nn.Module):
    r"""
    学习 \(\{A^{(1)},...,A^{(L)}\}\)，每个 \(A^{(\tau)}\in\mathbb{R}^{N\times N}\)。

    生成过程（每个滞后 \(\tau\)）：
      1) 得到打分矩阵：\(S^{(\tau)} = (E_u W_\tau)(E_v W'_\tau)^\top\)，其中 \(E_u,E_v\in\mathbb{R}^{N\times d}\)。
      2) 非负化：\(A^{(\tau)}_{\text{learn}} = \text{softplus}(S^{(\tau)})\)，并清零对角避免自环。
      3) 候选掩膜（可选）：只在允许的边上学习（例如先验候选）。
      4) 与先验混合（可选）：\(A^{(\tau)}_{\text{mix}} = \eta_\tau A^{(\tau)}_{\text{prior}} + (1-\eta_\tau) A^{(\tau)}_{\text{learn}}\)。
      5) 行 top-k 稀疏化 + 行归一化，得到最终 \(A^{(\tau)}\)。

    方向性由 \(E_u\) / \(E_v\) 解耦保证；行归一使每行权重和为 1（入边分布）。
    """

    def __init__(
        self,
        num_nodes: int,
        embed_dim: int = 32,
        num_lags: int = 2,
        topk_per_row: int = 15,
        nonneg: bool = True,
        use_prior: bool = True,
        per_lag_eta: bool = True,       # 每个滞后独立的混合系数 \eta_\tau
    ):
        super().__init__()
        self.N = num_nodes
        self.d = embed_dim
        self.L = num_lags
        self.k = topk_per_row
        self.nonneg = nonneg
        self.use_prior = use_prior
        self.per_lag_eta = per_lag_eta

        # 出入端节点嵌入（解耦以产生有向性）
        self.Eu = nn.Parameter(torch.randn(self.N, self.d) * 0.1)
        self.Ev = nn.Parameter(torch.randn(self.N, self.d) * 0.1)

        # 每个滞后一对线性投影（低秩打分）
        self.W_tau  = nn.Parameter(torch.randn(self.L, self.d, self.d) * math.sqrt(2.0 / self.d))
        self.Wp_tau = nn.Parameter(torch.randn(self.L, self.d, self.d) * math.sqrt(2.0 / self.d))

        # 先验（缓冲），可通过 set_prior 传入；支持 [N,N] 或 [L,N,N]
        self.register_buffer("A_prior_multi", torch.zeros(self.L, self.N, self.N))
        self.register_buffer("cand_mask_multi", torch.ones(self.L, self.N, self.N))  # 1=允许学习

        # 混合系数 \eta：取 sigmoid 后落到 (0,1)
        if self.per_lag_eta:
            self.eta = nn.Parameter(torch.full((self.L,), 0.7))  # 初期更信先验
        else:
            self.eta = nn.Parameter(torch.tensor(0.7))

        # 正则项的缓存（最近一次前向）
        self._last_As: List[torch.Tensor] = []
        self._last_A_learn: List[torch.Tensor] = []

    @torch.no_grad()
    def set_prior(
        self,
        A_prior: torch.Tensor,                 # [N,N] 或 [L,N,N]
        cand_mask: Optional[torch.Tensor] = None,  # 同上（不传则全 1）
        row_normalize_prior: bool = True
    ):
        """设置先验与候选掩膜（比如调用链/部署关系映射到实例级图）"""
        A_prior_multi = _expand_lags(A_prior, self.L).to(self.A_prior_multi.device)
        if row_normalize_prior:
            A_prior_multi = _row_normalize(A_prior_multi.clamp_min(0))
        self.A_prior_multi.copy_(A_prior_multi)
        if cand_mask is not None:
            cand = _expand_lags(cand_mask, self.L).to(self.cand_mask_multi.device)
            self.cand_mask_multi.copy_(cand)
        else:
            self.cand_mask_multi.fill_(1.0)
        # 先验无自环
        for t in range(self.L):
            self.A_prior_multi[t] = _zero_diag(self.A_prior_multi[t])

    # --------- 内部：构造单个滞后层的打分与混合 ----------

    def _scores(self, tau: int) -> torch.Tensor:
        """\(S^{(\tau)}=(E_u W_\tau)(E_v W'_\tau)^\top\) -> [N,N]（可有正负）"""
        U = self.Eu @ self.W_tau[tau]       # [N,d]
        V = self.Ev @ self.Wp_tau[tau]      # [N,d]
        return U @ V.t()                    # [N,N]

    def _mix(self, tau: int, A_learn: torch.Tensor) -> torch.Tensor:
        """与先验按 \(\eta_\tau\) 混合；若未启用先验则退化为纯 learned"""
        if self.use_prior:
            Ap = self.A_prior_multi[tau]
        else:
            Ap = torch.zeros_like(A_learn)

        Cm = self.cand_mask_multi[tau]      # 候选掩膜：0=不允许学习的边
        if self.per_lag_eta:
            eta_tau = torch.sigmoid(self.eta[tau])
        else:
            eta_tau = torch.sigmoid(self.eta)

        # A_mix = eta*prior + (1-eta)*(learned ⊙ mask)
        return eta_tau * Ap + (1.0 - eta_tau) * (A_learn * Cm)

    # ---------------- 前向：返回 [A^(1),...,A^(L)] ----------------

    def forward(self) -> List[torch.Tensor]:
        A_list, A_learn_list = [], []

        for tau in range(self.L):
            S = self._scores(tau)                             # [N,N]
            A_learn = F.softplus(S) if self.nonneg else S    # 非负化（可解释）
            A_learn = _zero_diag(A_learn)                    # 无自环
            A_learn_list.append(A_learn)

            A_mix = self._mix(tau, A_learn)                  # 与先验/掩膜混合
            A_tau = _row_topk_normalize(A_mix, k=self.k)     # 行 top-k + 行归一
            A_tau = _zero_diag(A_tau)                        # 再次确保无自环
            A_list.append(A_tau)

        self._last_As = A_list
        self._last_A_learn = A_learn_list
        return A_list  # list of L tensors, each [N,N]

    # ---------------- 一些常用正则（可选用） ----------------

    def l1_learned(self) -> torch.Tensor:
        r"""对 \(A_{\text{learn}}\) 做 \(\ell_1\) 稀疏正则（鼓励“小残差”）"""
        if not self._last_A_learn:
            return torch.tensor(0.0, device=self.Eu.device)
        return sum(A.abs().sum() for A in self._last_A_learn)

    def l1_residual_vs_prior(self) -> torch.Tensor:
        r"""对 \((A - A_{\text{prior}})\) 做 \(\ell_1\)（更强调“贴近先验”）"""
        if not self._last_As or not self.use_prior:
            return torch.tensor(0.0, device=self.Eu.device)
        loss = 0.0
        for t, A_tau in enumerate(self._last_As):
            loss = loss + (A_tau - self.A_prior_multi[t]).abs().sum()
        return loss

    def dag_penalty(self) -> torch.Tensor:
        r"""NOTEARS 无环约束：对 \(A=\sum_{\tau}A^{(\tau)}\) 施加 \(h(A)\)"""
        if not self._last_As:
            return torch.tensor(0.0, device=self.Eu.device)
        A_sum = torch.stack(self._last_As, 0).sum(0)
        A_sum = _zero_diag(A_sum)
        return notears_dag_penalty(A_sum)

    def get_last_graph(self) -> torch.Tensor:
        """返回最近一次前向得到的图 [L,N,N]（用于日志/可视化）"""
        if not self._last_As:
            return torch.zeros(self.L, self.N, self.N, device=self.Eu.device)
        return torch.stack(self._last_As, 0).detach()
