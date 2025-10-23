import torch
import torch.nn as nn
import torch.nn.functional as F

class RootCauseScorer(nn.Module):

    """
    根因定位模块：根据异常程度 + 传播影响，计算每个节点的根因得分。
    """

    def __init__(self, alpha=1.0, beta=1.0, config=None, loss_type='mse',device='cpu'):
        """
        :param alpha: 异常分数权重
        :param beta: 传播能力权重
        :param config: 配置对象，包含instance_metric_count_dict等信息
        :param loss_type: 损失函数类型 ('mse', 'mae', 'rmse')
        """
        super(RootCauseScorer, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.topk = config.topk  # 取前多少个实例
        self.config = config
        self.device = device
        self.metric_topk = 2
        self.instance_metric_count_dict = config.instance_metric_count_dict
        
        # 关键修复：使用与DataProcessor一致的实例顺序
        if hasattr(config, 'instance_metric_names'):
            self.instance_names = list(config.instance_metric_names.keys())
            self.instance_metric_names_dict = config.instance_metric_names
        else:
            # 向后兼容
            self.instance_names = list(config.instance_metric_count_dict.keys())
            self.instance_metric_names_dict = None
            
        self.loss_type = loss_type

    def get_top_loss_instance_info(self, instance_losses,top_error_metrics, label_instances=None, topk=None):
        # 获取topk的服务名称
        if topk is None:
            topk = self.topk
        topk = min(topk, len(self.instance_names))
        _, indices = torch.topk(instance_losses, topk)
        topk_names = [self.instance_names[i] for i in indices.tolist()]
        
        # 返回top-k实例的详细信息
        topk_details = {}
        for i, instance_name in enumerate(topk_names):
            instance_idx = indices[i].item()
            topk_details[instance_name] = {
                'rank': i + 1,
                'instance_loss': instance_losses[instance_idx].item(),
                'top_error_metrics': top_error_metrics[instance_name]
            }
        
        print("\n=== 根因分析结果 ===")
        for instance_name, details in topk_details.items():
            print(f"\n排名 {details['rank']}: {instance_name} (实例损失: {details['instance_loss']:.6f})")
            print("  前3个异常指标:")
            for metric_info in details['top_error_metrics']:
                print(f"    {metric_info['rank']}. {metric_info['metric_name']}: {metric_info['error_value']:.6f}")
        
        # 处理标签实例的异常信息（直接利用已计算的数据）
        label_anomaly_info = {}
        if label_instances is not None:
            # 确保label_instances是列表格式
            if isinstance(label_instances, str):
                label_instances = [label_instances]
            
            # 获取所有实例的排名（按异常度降序排列）
            sorted_losses, sorted_indices = torch.sort(instance_losses, descending=True)
            
            for label_instance in label_instances:
                # 寻找匹配的实例（支持前缀匹配）
                matched_instances = []
                for i, instance_name in enumerate(self.instance_names):
                    if instance_name.startswith(label_instance):
                        matched_instances.append((instance_name, i))
                
                if not matched_instances:
                    print(f"警告: 未找到与标签 '{label_instance}' 匹配的实例")
                    continue
                
                # 处理所有匹配的实例
                for instance_name, instance_idx in matched_instances:
                    instance_loss = instance_losses[instance_idx].item()
                    
                    # 获取该实例在所有实例中的排名
                    rank_in_all = (sorted_indices == instance_idx).nonzero(as_tuple=True)[0].item() + 1
                    
                    label_anomaly_info[instance_name] = {
                        'instance_loss': instance_loss,
                        'rank_in_all_instances': rank_in_all,
                        'total_instances': len(self.instance_names),
                        'top_error_metrics': top_error_metrics[instance_name]
                    }
            
            # 输出标签实例的异常信息
            print("\n=== 真实标签实例异常分析 ===")
            if label_anomaly_info:
                for instance_name, info in label_anomaly_info.items():
                    print(f"\n标签实例: {instance_name}")
                    print(f"  异常损失: {info['instance_loss']:.6f}")
                    print(f"  在所有实例中的排名: {info['rank_in_all_instances']}/{info['total_instances']}")
                    print(f"  前3个异常指标:")
                    for metric_info in info['top_error_metrics']:
                        print(f"    {metric_info['rank']}. {metric_info['metric_name']}: {metric_info['error_value']:.6f}")
            else:
                print(f"  未找到与标签 '{label_instances}' 匹配的实例")
        return topk_details, label_anomaly_info
        


    def calculate_instance_loss(self, metric_losses):
        """
        无batch输入：根据已计算的指标级别误差，计算实例级别的误差
        Args:
            metric_losses: [total_metrics] 每个指标的误差（已计算好的loss）
        Returns:
            instance_losses: [num_instances] 每个实例的平均误差
            total_loss: scalar 总体损失
            top_error_metrics: dict 每个实例前3个误差最大的指标名称和误差值
        """
        device = metric_losses.device
        instance_losses = []
        top_error_metrics = {}  # 保存每个实例的top3误差指标
        
        metric_losses = metric_losses.flatten()
        
        # 使用配置中的实例顺序，确保与DataProcessor一致
        for instance_name in self.instance_names:
            if self.instance_metric_names_dict:
                # 使用新的配置信息
                instance_info = self.instance_metric_names_dict[instance_name]
                start_idx = instance_info['start_idx']
                end_idx = instance_info['end_idx']
                metric_names = instance_info['full_names']
                num_metrics = end_idx - start_idx
            else:
                raise ValueError("缺少实例名称映射信息")
            
            if num_metrics == 0:
                instance_losses.append(torch.tensor(0.0, device=device))
                top_error_metrics[instance_name] = []
            else:
                instance_metric_errors = metric_losses[start_idx:end_idx]  # [num_metrics]
                
                # 获取top3误差指标的信息
                if len(instance_metric_errors) >= self.metric_topk:
                    topk_errors, topk_indices = torch.topk(instance_metric_errors, k=self.metric_topk)
                    instance_avg_loss = torch.mean(topk_errors)

                    # 保存topk指标名称和误差值
                    top_metrics_info = []
                    for i, (error_val, metric_idx) in enumerate(zip(topk_errors, topk_indices)):
                        metric_name = metric_names[metric_idx.item()]
                        top_metrics_info.append({
                            'rank': i + 1,
                            'metric_name': metric_name,
                            'error_value': error_val.item(),
                            'global_idx': start_idx + metric_idx.item()
                        })
                else:
                    # 如果指标数量少于3个，使用所有指标
                    instance_avg_loss = torch.mean(instance_metric_errors)
                    
                    # 保存所有指标信息
                    top_metrics_info = []
                    sorted_errors, sorted_indices = torch.sort(instance_metric_errors, descending=True)
                    for i, (error_val, metric_idx) in enumerate(zip(sorted_errors, sorted_indices)):
                        metric_name = metric_names[metric_idx.item()]
                        top_metrics_info.append({
                            'rank': i + 1,
                            'metric_name': metric_name,
                            'error_value': error_val.item(),
                            'global_idx': start_idx + metric_idx.item()
                        })
                
                instance_losses.append(instance_avg_loss)
                top_error_metrics[instance_name] = top_metrics_info
                
                # 调试信息
                # print(f"实例 {instance_name} (索引 {start_idx}-{end_idx}):")
                # for metric_info in top_metrics_info:
                #     print(f"  排名{metric_info['rank']}: {metric_info['metric_name']} = {metric_info['error_value']:.6f}")
        
        instance_losses = torch.stack(instance_losses)  # [num_instances]
        total_loss = torch.mean(metric_losses)
        return instance_losses, total_loss, top_error_metrics

    def get_ans_from_loss(self, metric_losses, label_instances=None):
        """
        基于预测误差进行根因定位（无batch输入）
        :param metric_losses: [total_metrics] 每个指标的误差
        :param label_instances: List[str] 或 str 可选的真实标签实例名称，用于输出其异常信息
        :return: List[str] Top-K实例名称, dict 详细的误差信息, dict 标签实例异常信息
        """
        instance_losses, _, top_error_metrics = self.calculate_instance_loss(metric_losses)
        topk_details, label_anomaly_info = self.get_top_loss_instance_info(instance_losses, top_error_metrics, label_instances)
        
        # # 获取topk的服务名称
        _, indices = torch.topk(instance_losses, self.topk)
        topk_names = [self.instance_names[i] for i in indices.tolist()]
      
        
        return topk_names, topk_details, label_anomaly_info

   

    def get_topk(self, root_score, instance_names, topk=5):
        """
        返回根因得分Top-K实例名
        :param root_score: [N] 每个实例的根因得分
        :param instance_names: List[str] 实例名称对应顺序
        :param topk: 返回前多少个
        :return: List[str] Top-K实例名称
        """
        _, indices = torch.topk(root_score, topk)
        topk_names = [instance_names[i] for i in indices.tolist()]
        return topk_names

    def get_root_cause_by_walk(self,metric_losses,A_list,label_instances=None):
        instance_losses, _, top_error_metrics = self.calculate_instance_loss(metric_losses)
        topk_details, label_anomaly_info = self.get_top_loss_instance_info(instance_losses, top_error_metrics, label_instances)
        #这里对instance_losses进行归一化
        instance_losses = instance_losses / (instance_losses.sum() + 1e-12)

        A_inst = torch.stack(A_list, 0)                         # [L, N_inst, N_inst]

        # 3) 合成 P
        gamma = 0.7
        L = A_inst.size(0)
        coeff = torch.tensor([gamma**(t+1) for t in range(L)], device=A_inst.device).view(L,1,1)
        P = (coeff * A_inst).sum(dim=0)                         # [N_inst, N_inst]
        P = P / (P.sum(dim=1, keepdim=True) + 1e-12)

        # 4) 个性化 PageRank
        alpha = 0.8
        r = instance_losses / (instance_losses.sum() + 1e-12)       # [N_inst]
        pi = r.clone()
       
        r = r.to(self.device)
        P = P.to(self.device)
        pi = pi.to(self.device)

        for _ in range(5):                                     # 迭代 10~30 次足够
            pi = (1-alpha) * (pi @ P) + alpha * r

        topk_idx = torch.topk(pi, k=self.config.topk).indices
        topk_instances = [self.instance_names[i] for i in topk_idx.tolist()]

        
        return  topk_instances,topk_details,label_anomaly_info

    def get_root_cause_by_upstream_adjustment(self, metric_losses, A_list, label_instances=None, topk=10, gamma=None):
        """
        基于上游传播调整的根因排序：
        1) 计算实例级重构误差，并选取Top-K异常实例；
        2) 在因果图中查找这些实例的上游节点，提取其重构误差；
        3) 将上游误差按边权加权，得到传播异常；
        4) 用实例自身重构误差减去传播异常，依据调整后的分数重新排序。

        :param metric_losses: [total_metrics] 指标级误差向量
        :param A_list: List[Tensor] 因果图的多阶邻接矩阵列表，形状 [L, N_inst, N_inst]
        :param label_instances: 可选标签实例名（或前缀），用于输出诊断信息
        :param topk: 需要分析的异常实例数量，默认10
        :param gamma: 合成多阶邻接矩阵的衰减因子；若为None则优先使用config.walk_gamma，否则回退0.7
        :return: (调整后Top-K实例名列表, Top-K实例详细信息, 标签实例异常信息)
        """

        instance_losses, _, top_error_metrics = self.calculate_instance_loss(metric_losses)
        instance_losses = instance_losses.to(metric_losses.device)
        raw_instance_losses = instance_losses.clone()

        num_instances = len(self.instance_names)
        if num_instances == 0:
            return [], {}, {}

        topk = min(topk, num_instances)
        if topk <= 0:
            return [], {}, {}

        if not A_list:
            raise ValueError("A_list 为空，无法进行上游传播分析")

        device = raw_instance_losses.device
        stacked_adj = torch.stack([adj.to(device) for adj in A_list], dim=0)

        if gamma is None:
            gamma = getattr(self.config, 'walk_gamma', 0.7)

        L = stacked_adj.size(0)
        coeff = torch.tensor([gamma ** (t + 1) for t in range(L)], device=device).view(L, 1, 1)
        causal_weights = (coeff * stacked_adj).sum(dim=0)  # [N_inst, N_inst]

        base_topk_details, label_anomaly_info = self.get_top_loss_instance_info(
            raw_instance_losses, top_error_metrics, label_instances=label_instances, topk=topk
        )
        topk_details = {name: detail.copy() for name, detail in base_topk_details.items()}

        _, topk_indices = torch.topk(raw_instance_losses, k=topk)
        adjusted_scores = []

        for rank_before, inst_idx in enumerate(topk_indices.tolist(), start=1):
            instance_name = self.instance_names[inst_idx]
            upstream_weights = causal_weights[:, inst_idx]
            upstream_mask = upstream_weights.abs() > 1e-12
            upstream_indices = torch.nonzero(upstream_mask, as_tuple=True)[0]

            propagated_contrib = torch.tensor(0.0, device=device)
            upstream_contributions = []

            if upstream_indices.numel() > 0:
                upstream_losses = raw_instance_losses[upstream_indices]
                edge_weights = upstream_weights[upstream_indices]
                propagated_per_edge = upstream_losses * edge_weights
                sorted_vals, sort_order = torch.sort(propagated_per_edge, descending=True)
                upstream_indices = upstream_indices[sort_order]
                upstream_losses = upstream_losses[sort_order]
                edge_weights = edge_weights[sort_order]

                propagated_contrib = sorted_vals.sum()

                for rank_up, (u_idx, loss_val, weight_val, prop_val) in enumerate(zip(upstream_indices.tolist(),
                                                                                     upstream_losses.tolist(),
                                                                                     edge_weights.tolist(),
                                                                                     sorted_vals.tolist()), start=1):
                    upstream_contributions.append({
                        'rank': rank_up,
                        'instance_name': self.instance_names[u_idx],
                        'upstream_loss': loss_val,
                        'edge_weight': weight_val,
                        'propagated_anomaly': prop_val
                    })

            adjusted_loss = torch.clamp(raw_instance_losses[inst_idx] - propagated_contrib, min=0.0)

            detail_record = topk_details.get(instance_name, {}).copy()
            detail_record.update({
                'rank_before_adjustment': rank_before,
                'instance_loss': raw_instance_losses[inst_idx].item(),
                'propagated_anomaly': propagated_contrib.item(),
                'adjusted_loss': adjusted_loss.item(),
                'upstream_contributions': upstream_contributions,
                'top_error_metrics': top_error_metrics[instance_name]
            })
            topk_details[instance_name] = detail_record
            adjusted_scores.append((instance_name, adjusted_loss.item()))

        adjusted_scores.sort(key=lambda item: item[1], reverse=True)
        adjusted_top_instances = [name for name, _ in adjusted_scores]

        for rank_after, instance_name in enumerate(adjusted_top_instances, start=1):
            topk_details[instance_name]['rank_after_adjustment'] = rank_after

        return adjusted_top_instances, topk_details, label_anomaly_info





  

   