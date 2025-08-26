import torch
import torch.nn as nn
import torch.nn.functional as F

class PodEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, temperature=0.5, lambda_reg=0.001):
        super(PodEmbedding, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)

        # Causal gate components
        self.causal_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.temperature = temperature
        self.lambda_reg = lambda_reg

    def gumbel_softmax_sample(self, logits):
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + 1e-9) + 1e-9)
        y = logits + gumbel_noise
        return torch.sigmoid(y / self.temperature)

    def forward(self, metric_embeddings,mapping=None):
        """
        :param metric_embeddings: Tensor of shape [B, M, D]
            - B: batch size
            - M: number of metrics per pod
            - D: embedding dimension per metric
        :return: pod_embeddings: Tensor of shape [B, D], l1_reg: scalar regularization term
        """
        if mapping is not None:
            instance_embeddings = []
            l1_regs = []
            start_idx = 0
            for instance_name, metric_count in mapping.items():
                end_idx = start_idx + metric_count
                instance_metric_embeddings = metric_embeddings[:, start_idx:end_idx, :]  # [B, m_i, D]

                Q = self.query(instance_metric_embeddings)
                K = self.key(instance_metric_embeddings)
                V = self.value(instance_metric_embeddings)

                attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
                attn_weights = self.softmax(attn_scores)
                context = torch.matmul(attn_weights, V)

                rho = torch.sigmoid(self.causal_mlp(instance_metric_embeddings)).squeeze(-1)
                beta = self.gumbel_softmax_sample(torch.log(rho + 1e-9))
                beta = beta.unsqueeze(-1)
                context = context * beta

                pod_embedding = context.mean(dim=1)
                instance_embeddings.append(pod_embedding)
                l1_regs.append(rho.sum())
                start_idx = end_idx

            pod_embedding = torch.stack(instance_embeddings, dim=1)  # 保留所有实例的 embedding
            l1_reg = self.lambda_reg * sum(l1_regs)
            return pod_embedding, l1_reg
        else:
            Q = self.query(metric_embeddings)
            K = self.key(metric_embeddings)
            V = self.value(metric_embeddings)

            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
            attn_weights = self.softmax(attn_scores)
            context = torch.matmul(attn_weights, V)

            rho = torch.sigmoid(self.causal_mlp(metric_embeddings)).squeeze(-1)
            beta = self.gumbel_softmax_sample(torch.log(rho + 1e-9))
            beta = beta.unsqueeze(-1)
            context = context * beta

            pod_embedding = context.mean(dim=1)
            l1_reg = self.lambda_reg * rho.sum()
            return pod_embedding, l1_reg