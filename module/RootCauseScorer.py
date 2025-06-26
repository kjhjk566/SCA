import torch
import torch.nn as nn

class RootCauseScorer(nn.Module):
    """
    根因定位模块：根据异常程度 + 传播影响，计算每个节点的根因得分。
    """

    def __init__(self, alpha=1.0, beta=1.0,config = None):
        """
        :param alpha: 异常分数权重
        :param beta: 传播能力权重
        """
        super(RootCauseScorer, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.topk = config.topk  # 取前多少个实例、
    def get_ans_causal(self, anomaly_score, edge_index,edge_weight, node_mapping, instance_names,adj):
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
        # A = torch.zeros(N, N, device=instance_anomaly.device)
        # # 使用edge_index和edge_weight构建图
        # src, dst = edge_index
        # A[src, dst] = edge_weight  # 使用edge_weight作为边的权重

        # 计算影响分数
        influence = torch.matmul(adj.T, instance_anomaly.unsqueeze(-1)).squeeze()
        
        #A小于0.1的设置为0
        # adj[adj<0.1] = 0
        # print("adj",adj)
        
        # 计算最终的根因得分
        root_score = instance_anomaly-influence
        # print("instance_anomaly",instance_anomaly)
        # print("influence",influence)
        # print("root_score",root_score)
        
        # 获取topk的服务名称
        _, indices = torch.topk(root_score, self.topk)
        topk_names = [instance_names[i] for i in indices.tolist()]
        return topk_names


        

    def get_ans(self, anomaly_score, edge_index,edge_weight, node_mapping, instance_names,adj):
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
        print("instance_anomaly.shape:",instance_anomaly.shape)

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