import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalGraphLearner(nn.Module):
    def __init__(self, num_instances, node_mapping,hidden_dim=64, use_dag_constraint=True):
        super().__init__()
        self.num_instances = num_instances
        self.node_mapping = node_mapping
        self.use_dag_constraint = use_dag_constraint

        # Learnable node embeddings
        self.emb1 = nn.Parameter(torch.randn(num_instances, hidden_dim))
        self.emb2 = nn.Parameter(torch.randn(num_instances, hidden_dim))
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self):
        # Compute adjacency matrix A ∈ ℝ^{N×N}
        e1 = self.proj(self.emb1)  # [N, D]
        e2 = self.proj(self.emb2)  # [N, D]
        logits = torch.matmul(e1, e2.T)  # [N, N]
        adj = F.relu(logits)  # 可选：加入 sigmoid 或 softmax
        return adj  # [N, N]

    def dag_penalty(self, adj):
        if not self.use_dag_constraint:
            return torch.tensor(0.0, device=adj.device)
        d = adj.shape[0]
        expm = torch.matrix_exp(adj * adj)  # NOTEARS DAG 约束
        return torch.trace(expm) - d

    def loss_from_residual(self, residual: torch.Tensor):
        """
        residual: [N] 实例的异常程度
        """
        instance_residual = self.get_instance_residual(residual)
        adj = self.forward()  # [N, N]
        propagated = torch.matmul(adj, instance_residual.unsqueeze(-1)).squeeze()  # [N]
        loss = F.mse_loss(propagated, instance_residual)  # 希望传播后的结果接近原始异常得分
        penalty = self.dag_penalty(adj)
        return loss, adj, penalty
    
    def get_instance_residual(self, residual: torch.Tensor):
        """
        将节点级残差聚合为实例级残差
        :param residual: [num_total_nodes] 节点级残差
        :param instance_names: List[str]，每个实例的名称顺序
        :return: [num_instances] 实例级残差向量
        """
        residual_per_instance = []
        idx = 0
        for name in self.node_mapping.keys():
            num_nodes = self.node_mapping.get(name, 0)
            if num_nodes == 0:
                residual_per_instance.append(torch.tensor(0.0, device=residual.device))
                continue
            res = residual[idx:idx + num_nodes].mean()
            residual_per_instance.append(res)
            idx += num_nodes
        residual_vector = torch.stack(residual_per_instance)
        return residual_vector 