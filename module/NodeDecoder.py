    
import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeDecoder(nn.Module):
    def __init__(self, node_input_dim, hidden_dim, node_mapping,metric_embbeding_dim=64):
        """
        :param input_dim: 输入实例向量维度 (比如128)
        :param hidden_dim: 中间隐层大小
        :param node_mapping: dict，instance_name → num_metrics（指标数量）
        """
        super(NodeDecoder, self).__init__()
        self.node_mapping = node_mapping
        self.hidden_dim = hidden_dim
        self.node_input_dim = node_input_dim
        self.fc1 = nn.Linear(node_input_dim, hidden_dim)
        
        # 实例编码器
        self.instance_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 指标预测器（共享）
        self.metric_predictor = nn.Sequential(
            nn.Linear(node_input_dim + metric_embbeding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.predictor_gnet = nn.Sequential(
            nn.Linear(node_input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def predict_next_step_gnet(self, x):
        """
        对输入特征 [B, total_metrics, input_dim] 进行单步预测，输出 [B, total_metrics]
        """
        # x: [B, total_metrics, input_dim]
        out = self.predictor_gnet(x)
        out = out.squeeze(-1)
        return out
    def forward(self, h_instances, instance_names, metric_embeddings):
        """
        :param h_instances: [B, N, input_dim] - N是实例数量
        :param instance_names: List[str]，每个h_i对应的实例名，长度为N
        :param metric_embeddings: [B, total_metrics, 64] - 所有指标的嵌入
        :return: [B, total_metrics] - 每个指标的预测值
        """
        B, N, _ = h_instances.shape
        B_metric, total_metrics, metric_dim = metric_embeddings.shape
        
        assert B == B_metric, f"Batch sizes don't match: {B} vs {B_metric}"
        
        # 对实例特征进行编码
        encoded_instances = h_instances  # [B, N, hidden_dim]
        
        # 为每个指标分配对应的实例特征
        batch_predictions = []
        
        for b in range(B):
            metric_predictions = []
            start_idx = 0
            
            for i, instance_name in enumerate(instance_names):
                num_metrics = self.node_mapping.get(instance_name, 0)
                if num_metrics == 0:
                    continue
                
                # 获取当前实例的特征
                instance_feature = encoded_instances[b, i]  # [hidden_dim]
                
                # 获取当前实例对应的指标嵌入
                end_idx = start_idx + num_metrics
                if end_idx > total_metrics:
                    print(f"Warning: end_idx {end_idx} > total_metrics {total_metrics}")
                    break
                    
                instance_metric_embeddings = metric_embeddings[b, start_idx:end_idx]  # [num_metrics, 64]
                
                # 将实例特征扩展到与指标数量匹配
                expanded_instance_feature = instance_feature.unsqueeze(0).repeat(num_metrics, 1)  # [num_metrics, hidden_dim]
                
                # 拼接实例特征和指标嵌入
                combined_features = torch.cat([expanded_instance_feature, instance_metric_embeddings], dim=1)  # [num_metrics, hidden_dim + 64]
                
                # 预测每个指标的值
                metric_preds = self.metric_predictor(combined_features)  # [num_metrics, 1]
                metric_predictions.append(metric_preds.squeeze(-1))  # [num_metrics]
                
                start_idx = end_idx
            
            # 拼接当前batch的所有预测
            if metric_predictions:
                batch_pred = torch.cat(metric_predictions, dim=0)  # [total_metrics_for_this_batch]
            else:
                batch_pred = torch.zeros(total_metrics, device=h_instances.device)
            
            # 如果预测的指标数少于total_metrics，进行填充
            if batch_pred.shape[0] < total_metrics:
                padding = torch.zeros(total_metrics - batch_pred.shape[0], device=h_instances.device)
                batch_pred = torch.cat([batch_pred, padding], dim=0)
            
            batch_predictions.append(batch_pred)
        
        # 堆叠所有batch的预测
        final_predictions = torch.stack(batch_predictions, dim=0)  # [B, total_metrics]
        
        return final_predictions
    
    def forward_padded(self, h_instances, instance_names):
        """
        返回填充后的张量版本，方便批量处理
        :param h_instances: [B, N, input_dim] = [16, 45, 128]
        :param instance_names: List[str]，每个实例对应的名称，长度为45
        :return: [B, max_total_metrics] 填充后的预测结果
        """
        batch_predictions = self.forward(h_instances, instance_names)
        
        # 计算每个batch的最大指标总数
        max_total_metrics = 0
        for b in range(len(batch_predictions)):
            total_metrics = sum(len(pred) for pred in batch_predictions[b])
            max_total_metrics = max(max_total_metrics, total_metrics)
        
        if max_total_metrics == 0:
            return torch.zeros((len(batch_predictions), 0), device=h_instances.device)
        
        # 填充到相同长度
        padded_predictions = []
        for b in range(len(batch_predictions)):
            # 拼接当前batch的所有预测
            batch_pred = torch.cat(batch_predictions[b]) if batch_predictions[b] else torch.tensor([], device=h_instances.device)
            
            # 填充到最大长度
            if len(batch_pred) < max_total_metrics:
                pad_size = max_total_metrics - len(batch_pred)
                padding = torch.zeros(pad_size, device=h_instances.device)
                batch_pred = torch.cat([batch_pred, padding])
            
            padded_predictions.append(batch_pred)
        
        return torch.stack(padded_predictions, dim=0)  # [B, max_total_metrics]