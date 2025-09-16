# module/PodEmbedding.py  —— 新版，支持“逐时刻→序列”聚合
import torch
import torch.nn as nn
import torch.nn.functional as F

class PodEmbedding(nn.Module):
    def __init__(
        self,
        input_dim,              # = T（时间窗长度），保留该参数以兼容旧签名
        hidden_dim,             # 兼容旧签名（self_attention 分支用），可忽略
        agg_type="timewise_attn",
        d_model=64,             # 每个时刻的实例嵌维度 D_out
        dropout=0.0
    ):
        """
        支持两类聚合：
        - timewise_attn: 对每个时刻，在指标维上做注意力池化（带掩码），输出 [B,N,d_model,T]
        - timewise_mean: 对每个时刻，在指标维上做简单均值（带掩码），输出 [B,N,1,T] 或投到 d_model
        """
        super().__init__()
        self.agg_type = agg_type
        self.d_model = d_model

        if agg_type == "timewise_attn":
            # 将“单个指标在时刻 t 的标量/小特征”映射到 token（共享权重）
            # 这里假定每个指标在每个时刻是“标量”，故 Linear(1->d_model)；若是多维，可改 Linear(Dm->d_model)
            self.metric_proj = nn.Linear(1, d_model)
            # 注意力打分器（逐指标）
            self.score = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1)
            )
        elif agg_type == "timewise_mean":
            # 简单均值：可选把标量先投到 d_model，再均值
            self.metric_proj = nn.Linear(1, d_model)
        else:
            raise ValueError(f"Unsupported agg_type: {agg_type}")

    def forward(self, x, mapping=None):
        """
        推荐输入契约：
          x: [B, N, M_max, T]  —— 每个实例对齐到相同的 M_max，padding 的位置将用 mapping 做 mask
          mapping: dict {instance_name: m_i} —— 每个实例的真实指标数
        也兼容旧契约：
          x: [B, M_total, T] + mapping （按 mapping 切回各实例，再逐时刻聚合）

        输出：
          pod_seq: [B, N, D_out, T]  —— 逐时刻的实例嵌入序列（D_out = d_model）
        """
        if x.dim() == 4:  # [B,N,M_max,T]
            assert mapping is not None, "mapping is required for [B,N,M_max,T]"
            return self._forward_batched_instances(x, mapping)
        elif x.dim() == 3:  # [B,M_total,T] —— 旧契约
            assert mapping is not None, "mapping is required for [B,M_total,T]"
            return self._forward_flat_then_group(x, mapping)
        else:
            raise ValueError("x must be [B,N,M_max,T] or [B,M_total,T]")

    # —— 实现：直接吃 [B,N,M_max,T] 的高效路径
    def _forward_batched_instances(self, x, mapping):
        B, N, M_max, T = x.shape
        device = x.device
        # 构造 mask: [N, M_max]，第 n 个实例前 m_n 有效
        m_list = torch.tensor(list(mapping.values()), device=device)
        idx = torch.arange(M_max, device=device).unsqueeze(0)      # [1,M_max]
        valid_nm = (idx < m_list.unsqueeze(1)).bool()              # [N,M_max]
        valid = valid_nm.unsqueeze(0).unsqueeze(-1)                # [B,N,M_max,1]
        valid = valid.expand(B, -1, -1, T)                         # [B,N,M_max,T]

        # 到“时刻优先”的布局，便于做逐时刻聚合
        # x_btmt: [B,N,T,M_max]
        x_bntm = x.permute(0, 1, 3, 2).contiguous()
        valid_bntm = valid.permute(0, 1, 3, 2).contiguous()        # [B,N,T,M_max]

        # 将每个时刻的每个指标标量 -> token
        # 先扩展出 feature 维度以适配 Linear(1 -> d_model)
        x_feat = x_bntm.unsqueeze(-1)                              # [B,N,T,M_max,1]
        H = self.metric_proj(x_feat)                               # [B,N,T,M_max,d_model]

        if self.agg_type == "timewise_attn":
            # 打分并在指标维 softmax（带 mask）
            logits = self.score(H)                                 # [B,N,T,M_max,1]
            logits = logits.squeeze(-1)                            # [B,N,T,M_max]

            # mask 到非常小的数，避免参与 softmax
            very_neg = torch.finfo(H.dtype).min
            logits = logits.masked_fill(~valid_bntm, very_neg)

            attn = torch.softmax(logits, dim=-1)                   # [B,N,T,M_max]
            attn = attn.unsqueeze(-1)                              # [B,N,T,M_max,1]

            # 加权求和（指标维）
            z = (attn * H).sum(dim=3)                              # [B,N,T,d_model]
            # 回到 [B,N,d_model,T]
            z = z.permute(0, 1, 3, 2).contiguous()                 # [B,N,d_model,T]
            return z

            # 备注：若想观察每时刻“最重要指标”，topk(attn[..., t, :], k=K) 即可

        elif self.agg_type == "timewise_mean":
            # 对 padding 做 0 处理，再按有效个数均值
            H = H * valid_bntm.unsqueeze(-1)                       # [B,N,T,M_max,d_model]
            denom = valid_bntm.sum(dim=3, keepdim=True).clamp_min(1)  # [B,N,T,1]
            z = H.sum(dim=3) / denom.unsqueeze(-1)                 # [B,N,T,d_model]
            z = z.permute(0, 1, 3, 2).contiguous()                 # [B,N,d_model,T]
            return z

        else:
            raise ValueError(f"Unsupported agg_type: {self.agg_type}")

    # —— 兼容旧契约 [B,M_total,T]：按 mapping 切成 [B,N,m_i,T] 再复用上面的逻辑
    def _forward_flat_then_group(self, x, mapping):
        B, M_total, T = x.shape
        pods = []
        start = 0
        for _, m_i in mapping.items():
            seg = x[:, start:start+m_i, :]                         # [B,m_i,T]
            seg = seg.unsqueeze(1)                                 # [B,1,m_i,T]
            z = self._forward_batched_instances(seg, {"dummy": m_i})  # [B,1,d_model,T]
            pods.append(z)                                         # list of [B,1,d_model,T]
            start += m_i
        return torch.cat(pods, dim=1)                              # [B,N,d_model,T]