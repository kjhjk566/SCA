import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalGraphLearner(nn.Module):
    def __init__(self,config, num_instances, node_mapping,hidden_dim=64, use_dag_constraint=True):
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
        adj = torch.clamp(adj, min=0, max=3)  # 缩小范围防爆炸
        try:
            expm = torch.matrix_exp(adj * adj)
            # 防止 nan/inf 并限幅
            expm = torch.nan_to_num(expm, nan=1e3, posinf=1e3, neginf=0.0)
            expm = torch.clamp(expm, min=0.0, max=1e3)
            penalty = torch.trace(expm) - d
        except RuntimeError:
            penalty = torch.tensor(1e3, device=adj.device)
        if torch.isnan(penalty) or torch.isinf(penalty):
            penalty = torch.tensor(1e3, device=adj.device)
        return penalty

    def loss_from_residual(self, residual: torch.Tensor):
        """
        residual: [N] 实例的异常程度
        """
        
        instance_residual = self.get_instance_residual(residual)
        batch_size = instance_residual.shape[0]
        all_loss = 0
        all_penalty = 0
        all_l1_reg = 0
        adj = self.forward()  # [N, N]
        for i in range(batch_size):
            
            propagated = torch.matmul(adj, instance_residual[i].unsqueeze(-1)).squeeze()  # [N]
            #propagated的nan
            #print("propagated:", propagated)
            #print("instance_residual:", instance_residual)
            #print("adj:", adj)
            loss = F.mse_loss(propagated, instance_residual[i])  # 希望传播后的结果接近原始异常得分
            
            penalty = self.dag_penalty(adj)
            all_loss += loss 
            all_penalty += penalty
            #L1正则化
            l1_reg = torch.norm(adj, p=1)
            all_l1_reg += l1_reg
        return all_loss/batch_size, adj, all_penalty/batch_size, all_l1_reg/batch_size
    
    def get_instance_residual(self, residual: torch.Tensor):
        """
        将节点级残差聚合为实例级残差
        :param residual: [num_total_nodes] 节点级残差
        :param instance_names: List[str]，每个实例的名称顺序
        :return: [num_instances] 实例级残差向量
        """
        batch_size = residual.shape[0]
        ans = []
        for i in range(batch_size):
            residual_per_instance = []
            idx = 0
            for name in self.node_mapping.keys():
                num_nodes = self.node_mapping.get(name, 0)
                if num_nodes == 0:
                    residual_per_instance.append(torch.tensor(0.0, device=residual.device))
                    continue
                res = residual[i][idx:idx + num_nodes].mean()
                residual_per_instance.append(res)
                idx += num_nodes
            ans.append(torch.tensor(residual_per_instance, device=residual.device))
        residual_vector = torch.stack(ans)
        
        return residual_vector 