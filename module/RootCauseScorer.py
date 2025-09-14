import torch
import torch.nn as nn
import torch.nn.functional as F

class RootCauseScorer(nn.Module):

    """
    根因定位模块：根据异常程度 + 传播影响，计算每个节点的根因得分。
    """

    def __init__(self, alpha=1.0, beta=1.0, config=None, loss_type='mse'):
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
                if len(instance_metric_errors) >= 3:
                    top3_errors, top3_indices = torch.topk(instance_metric_errors, k=3)
                    instance_avg_loss = torch.mean(top3_errors)
                    
                    # 保存top3指标名称和误差值
                    top_metrics_info = []
                    for i, (error_val, metric_idx) in enumerate(zip(top3_errors, top3_indices)):
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
        
        # 获取topk的服务名称
        _, indices = torch.topk(instance_losses, self.topk)
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
        
        return topk_names, topk_details, label_anomaly_info

    # def get_ans_from_loss(self, predictions, y_true):
    #     """
    #     基于预测误差进行根因定位
    #     :param predictions: [B, total_metrics] 预测值
    #     :param y_true: [B, total_metrics] 真实值
    #     :return: List[str] Top-K实例名称
    #     """
    #     # 计算实例级别的误差
    #     _, instance_losses, _ = self.calculate_instance_loss(predictions, y_true)
        
    #     # 取第一个batch的结果（如果需要处理多个batch，可以修改这里）
    #     instance_anomaly_scores = instance_losses[0]  # [num_instances]
        
    #     # 获取topk的服务名称
    #     _, indices = torch.topk(instance_anomaly_scores, self.topk)
    #     topk_names = [self.instance_names[i] for i in indices.tolist()]
    #     return topk_names

    def get_ans(self, anomaly_score, edge_index, edge_weight, node_mapping, instance_names, adj):
        """
        :param anomaly_score: [num_total_nodes] 节点级异常分数
        :param edge_index: [2, num_edges] 实例调用边
        :param node_mapping: dict[str -> int] 每个实例对应的节点数
        :param instance_names: List[str] 当前 batch 的实例名顺序
        :return: root_score: [num_instances]
        """
        instance_anomaly = []
        idx = 0
        for name in instance_names:
            num_nodes = node_mapping.get(name, 0)
            if num_nodes == 0:
                continue
            score_slice = anomaly_score[:, idx:idx+num_nodes]  # 保留 batch 维度
            instance_anomaly.append(score_slice.mean())
            idx += num_nodes

        if len(instance_anomaly) == 0:
            return []

        instance_anomaly = torch.stack(instance_anomaly)

        N = instance_anomaly.size(0)
        A = torch.zeros(N, N, device=instance_anomaly.device)
        src, dst = edge_index
        A[src, dst] = 1.0

        influence = torch.matmul(A, instance_anomaly.unsqueeze(-1)).squeeze()
        root_score = self.alpha * instance_anomaly + self.beta * influence
        _, indices = torch.topk(root_score, self.topk)
        topk_names = [instance_names[i] for i in indices.tolist()]
        return topk_names

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

    def get_root_cause_by_walk(self,metric_losses,graph):
        instance_losses, _, top_error_metrics = self.calculate_instance_loss(metric_losses)
        instance_scores = {name: loss.item() for name, loss in zip(self.instance_names, instance_losses)}
        inferer = InstanceRootCauseInferer(instance_scores, graph)
        root_causes = inferer.infer_root_causes()
        root_causes = root_causes[:5]
        root_causes = [instance for instance, score in root_causes]
        
        # 同时返回详细的误差信息
        print("\n=== 随机游走根因分析结果 ===")
        for i, instance_name in enumerate(root_causes):
            if instance_name in top_error_metrics:
                print(f"\n排名 {i+1}: {instance_name}")
                print("  前3个异常指标:")
                for metric_info in top_error_metrics[instance_name]:
                    print(f"    {metric_info['rank']}. {metric_info['metric_name']}: {metric_info['error_value']:.6f}")
        
        return root_causes






import numpy as np
import networkx as nx
import pandas as pd


class InstanceRootCauseInferer:
    def __init__(self, instance_scores: dict, deployment_graph: dict, lambda_weight=0.2, rho=0.2):
        """
        :param error_df: DataFrame，每行是一个实例，每列是一个指标，值是重构误差
        :param deployment_graph: dict，实例之间的邻接表（部署图）
        :param lambda_weight: λ，偏相关性与异常度之间的权重，默认为 0.2
        :param rho: ρ，控制是否允许“反向跳转”，默认为 0.2
        """

        self.instance_scores = instance_scores
        self.graph = self._build_graph(deployment_graph)
        self.lambda_weight = lambda_weight
        self.rho = rho
        self.transition_matrix, self.node_idx = self._build_transition_matrix()

    def _build_graph(self, adj_dict: dict) -> nx.DiGraph:
        """根据邻接表构建有向图。确保所有节点（含无出边的）均被加入图中。"""
        G = nx.DiGraph()
        # 先加入所有键（有出边的）
        for src in adj_dict.keys():
            G.add_node(src)
        # 再加入边与可能只作为目标出现的节点
        for src, targets in adj_dict.items():
            for tgt in targets:
                G.add_edge(src, tgt)
        return G

    def _relative_anomaly(self, v_from: str, v_to: str) -> float:
        """
        相对异常程度：沿边 v_from -> v_to 转移时，源节点的异常度相对大小。
        公式： score(from) / (score(from) + score(to) + eps)
        """
        eps = 1e-6
        s_from = self.instance_scores.get(v_from, 0.0)
        s_to = self.instance_scores.get(v_to, 0.0)
        return s_from / (s_from + s_to + eps)

    def _build_transition_matrix(self):
        """
        构造随机游走的转移矩阵 H。
        约定：H[row=i, col=j] 表示“当前在 i，下一步转移到 j 的概率”。
        注意：此前版本在填充 H 时将概率放到了列方向，导致与采样时按“行”读取不一致。
        此处修正为严格的“行→列”为出度分布，并处理无出边/出度为 0 的“悬挂节点（dangling node）”。
        """
        nodes = list(self.graph.nodes)
        n = len(nodes)
        idx = {n_: i for i, n_ in enumerate(nodes)}
        H_prime = np.zeros((n, n), dtype=float)  # 未归一化的权重矩阵

        # 正向与反向权重填充（行→列）
        for v_from in nodes:
            j = idx[v_from]
            for v_to in self.graph.successors(v_from):
                i = idx[v_to]
                # 正向（从因到果）：行=j 到 列=i
                w_forward = (1 - self.lambda_weight) * self._relative_anomaly(v_from, v_to)
                H_prime[j, i] += w_forward
                # 反向（从果回因）：行=i 到 列=j
                w_backward = self.rho * (1 - self.lambda_weight) * self._relative_anomaly(v_to, v_from)
                H_prime[i, j] += w_backward

        # 自环（停留）分量：若某行出度权重和 < 1，则把剩余质量补到对角线，保证可采样且含“停留”含义
        for r in range(n):
            row_sum = H_prime[r].sum()
            if row_sum < 1.0:
                H_prime[r, r] += (1.0 - row_sum)

        # 对每一行进行归一化，得到真正的转移概率矩阵 H
        row_sums = H_prime.sum(axis=1, keepdims=True)
        # 处理完全无边且前一步没有被补偿到 1 的极端情况
        row_sums[row_sums == 0] = 1.0
        H = H_prime / row_sums
        return H, idx

    def infer_root_causes(self, steps: int = 10000):
        """
        从最异常实例出发，执行随机游走，返回根因排序列表（按访问频次降序）。
        - 若遇到“悬挂节点”（整行概率为 0），会以自环概率 1 处理。
        """
        H, idx = self.transition_matrix, self.node_idx
        inv_idx = {i: n for n, i in idx.items()}
        visit_count = {n: 0 for n in idx}

        # 以最异常的实例为起点
        start_instance = max(self.instance_scores.items(), key=lambda x: x[1])[0]
        current_idx = idx[start_instance]

        for _ in range(steps):
            probs = H[current_idx]
            s = probs.sum()
            if s <= 0:  # 极端防御：若当前行为全 0，则自环
                next_idx = current_idx
            else:
                # numpy 要求 p 合法（非负、和为 1），前面已归一化，这里仍做一次稳健处理
                next_idx = np.random.choice(len(probs), p=probs)
            visit_count[inv_idx[next_idx]] += 1
            current_idx = next_idx

        # 按访问次数排序，作为根因可能性的排名
        sorted_instances = sorted(visit_count.items(), key=lambda x: x[1], reverse=True)
        return sorted_instances