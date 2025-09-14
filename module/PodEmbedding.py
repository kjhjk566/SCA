import torch
import torch.nn as nn
import torch.nn.functional as F

class PodEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, agg_type="weighted_mean"):
        """
        :param input_dim: 每个指标的时间序列长度 T
        :param hidden_dim: Attention 的隐藏维度
        :param agg_type: "weighted_mean" 或 "self_attention"
        """
        super(PodEmbedding, self).__init__()
        self.agg_type = agg_type
        
        # 加权平均
        if agg_type == "weighted_mean":
            self.weight_mlp = nn.Linear(input_dim, 1)

        # Self-Attention
        elif agg_type == "self_attention":
            self.query = nn.Linear(input_dim, hidden_dim)
            self.key = nn.Linear(input_dim, hidden_dim)
            self.value = nn.Linear(input_dim, hidden_dim)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mapping=None):
        """
        :param x: [B, M, T]
                  B=batch, M=指标数, T=时间步
        :param mapping: dict {pod_name: metric_count}
        :return: pod_embeddings: [B, num_pods, D]
                 D=hidden_dim (attention) 或 input_dim (weighted mean)
        """
        B, M, T = x.shape

        if mapping is not None:
            pod_embeddings = []
            start_idx = 0
            for pod_name, metric_count in mapping.items():
                end_idx = start_idx + metric_count
                metrics = x[:, start_idx:end_idx, :]   # [B, m_i, T]

                if self.agg_type == "weighted_mean":
                    # [B, m_i, T] -> [B, m_i, 1]
                    weights = torch.sigmoid(self.weight_mlp(metrics))
                    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
                    # 聚合 -> [B, T]
                    pod_emb = (metrics * weights).sum(dim=1)

                elif self.agg_type == "self_attention":
                    Q = self.query(metrics)  # [B, m_i, H]
                    K = self.key(metrics)    # [B, m_i, H]
                    V = self.value(metrics)  # [B, m_i, H]

                    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)  # [B, m_i, m_i]
                    attn_weights = self.softmax(attn_scores)
                    context = torch.matmul(attn_weights, V)  # [B, m_i, H]
                    pod_emb = context.mean(dim=1)            # [B, H]

                pod_embeddings.append(pod_emb)
                start_idx = end_idx

            pod_embeddings = torch.stack(pod_embeddings, dim=1)  # [B, num_pods, D]

        else:
            # 没有 mapping 时，直接把所有指标看作一个 pod
            if self.agg_type == "weighted_mean":
                weights = torch.sigmoid(self.weight_mlp(x))
                weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
                pod_embeddings = (x * weights).sum(dim=1, keepdim=True)  # [B, 1, T]

            elif self.agg_type == "self_attention":
                Q = self.query(x)  # [B, M, H]
                K = self.key(x)    # [B, M, H]
                V = self.value(x)  # [B, M, H]

                attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
                attn_weights = self.softmax(attn_scores)
                context = torch.matmul(attn_weights, V)  # [B, M, H]
                pod_emb = context.mean(dim=1)            # [B, H]
                pod_embeddings = pod_emb.unsqueeze(1)    # [B, 1, H]

        return pod_embeddings