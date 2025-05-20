import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeDecoder(nn.Module):
    def __init__(self, input_dim, metric_embbeding_dim,hidden_dim, node_mapping):
        """
        :param input_dim: 输入实例向量维度 (比如64)
        :param hidden_dim: 中间隐层大小
        :param node_mapping: dict，instance_name → num_nodes（指标节点数量）
        """
        super(NodeDecoder, self).__init__()
        self.node_mapping = node_mapping
        self.hidden_dim = hidden_dim
        self.metric_embbeding_dim = metric_embbeding_dim
        # 第一步，编码每个实例向量
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # 第二步，不同实例内部节点扩展时，要共享一套节点生成器
        self.node_generator = nn.Sequential(
            nn.Linear(metric_embbeding_dim+hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )  # 三层MLP，每个节点输出一个数值

    def forward(self, h_instances, instance_names, metric_embeddings):
        """
        :param h_instances: [B,N, input_dim]
        :param instance_names: List[str]，每个h_i对应的实例名，长度为N
        :param metric_embeddings: List[List[Tensor]]，外层为 batch，内层为每个实例的指标特征 Tensor [num_nodes, metric_dim]
        :return: Tensor [B, max_metric_nodes]
        """
        B, N, _ = h_instances.shape
        hidden = F.relu(self.fc1(h_instances))  # [B, N, hidden_dim]
        batch_preds = []
        for b in range(B):
            node_preds = []
            for i, instance_name in enumerate(instance_names):
                num_nodes = self.node_mapping.get(instance_name, 0)
                if num_nodes == 0:
                    continue
                metric = metric_embeddings[b][i]  # [num_nodes, metric_dim]
                if metric.shape[0] == 0:
                    continue
                h_instance = hidden[b, i].unsqueeze(0).repeat(metric.shape[0], 1)  # [num_nodes, hidden_dim]
                h = torch.cat([h_instance, metric], dim=1)  # [num_nodes, hidden_dim + metric_dim]
                
                node_pred = self.node_generator(h)  # [num_nodes, 1]
                node_preds.append(node_pred)
            if len(node_preds) > 0:
                node_preds = torch.cat(node_preds, dim=0)  # [sum_nodes, 1]
            else:
                node_preds = torch.zeros((0, 1), device=h_instances.device)
            batch_preds.append(node_preds)
        # Pad to max_metric_nodes per batch
        max_nodes = max([p.shape[0] for p in batch_preds]) if batch_preds else 0
        padded_preds = []
        for preds in batch_preds:
            if preds.shape[0] < max_nodes:
                pad = torch.zeros((max_nodes - preds.shape[0], preds.shape[1]), device=preds.device)
                preds = torch.cat([preds, pad], dim=0)
            padded_preds.append(preds.squeeze(-1))  # [max_nodes]
        if len(padded_preds) == 0:
            return torch.zeros((B, 0), device=h_instances.device)
        final_node_preds = torch.stack(padded_preds, dim=0)  # [B, max_metric_nodes]
        return final_node_preds