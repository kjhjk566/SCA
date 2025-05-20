import torch
import torch.nn as nn
import torch.nn.functional as F
from module.graph.InstanceGraphEncoder import InstanceGraphEncoder
from module.graph.ServiceGraphEncoder import ServiceGraphEncoder
from module.NodeDecoder import NodeDecoder
from module.graph.StaticCallGraphBuilder import StaticCallGraphBuilder
from torch_geometric.utils import dense_to_sparse
from module.TemporalEncoder import TemporalEncoder
from module.graph.CausalGraphLearner import CausalGraphLearner
class FullGraphRCA(nn.Module):
    def __init__(self, config, input_dim, metric_embbeding_dim,instance_hidden_dim, service_hidden_dim):
        super(FullGraphRCA, self).__init__()
        
       

        self.config = config
        self.alpha = 0.1
        self.beta = 0.01

        self.temporal_encoder = TemporalEncoder(
            input_length=input_dim,
            hidden_dim=metric_embbeding_dim,
            num_layers=3,
            nhead=4,
            dropout=0.1
        )

        # 子图编码器
        self.instance_encoder = InstanceGraphEncoder(
            input_dim=instance_hidden_dim, hidden_dim=instance_hidden_dim, output_dim=instance_hidden_dim
        )

        # 服务传播器
        self.service_encoder = ServiceGraphEncoder(
            input_dim=instance_hidden_dim, hidden_dim=service_hidden_dim, output_dim=service_hidden_dim
        )

        # 节点级解码器
        self.decoder = NodeDecoder(
            input_dim=service_hidden_dim,metric_embbeding_dim = metric_embbeding_dim, hidden_dim=64, node_mapping=None  # node_mapping要外面set一下
        )
        self.causal_graph_learner = CausalGraphLearner(
            num_instances=len(config.instance_metric_mapping),
            node_mapping=config.instance_metric_mapping,
            hidden_dim=64,
            use_dag_constraint=True
        )

        # edge_index 和 edge_weight 也要初始化后 set
        self.edge_index = None
        self.edge_weight = None

    def set_call_graph(self, call_graph,device=None):
        """设置服务调用图"""
        from torch_geometric.utils import dense_to_sparse
        edge_index, edge_weight = dense_to_sparse(torch.tensor(call_graph, dtype=torch.float))
        print("edge_index:", edge_index)
        print("edge_weight:", edge_weight)
        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device)

    def set_node_mapping(self, node_mapping):
        """设置实例到节点数量的映射"""
        self.decoder.node_mapping = node_mapping
    def get_instance_metric_tensor(self, batch_x):
        """
        拆分拼接后的节点特征 batch_x，按 batch 和实例名切分为变长结构
        :param batch_x: Tensor [total_nodes, feature_dim]
        :param instance_names: List[str]，每个 batch 中实例的名称列表
        :param instance_metric_mapping: Dict[str, int]，每个实例拥有的指标节点数量
        :param batch_size: int，每个 batch 的图数量
        :return: List[List[Tensor]]，外层为 batch，内层为该 batch 中每个实例的指标特征 Tensor [V_i, D]
        """
        batch_size = self.config.batch_size
        instance_metric_mapping = self.config.instance_metric_mapping
        instance_names = list(map(str, instance_metric_mapping.keys()))


        split_result = []
        cur_index = 0  # 当前在 batch_x 中的位置

        for b in range(batch_size):
            inst_tensors = []
            for inst in instance_names:
                num_metrics = instance_metric_mapping.get(inst, 0)
                if num_metrics == 0:
                    inst_tensors.append(torch.zeros((0, batch_x.shape[1]), device=batch_x.device))
                    continue
                inst_feat = batch_x[cur_index:cur_index + num_metrics]  # shape: [num_metrics, D]
                inst_tensors.append(inst_feat)
                cur_index += num_metrics
            split_result.append(inst_tensors)

        return split_result



    def _encode_and_predict(self, batch, instance_names):
        #print("batch.x:", batch.x.shape)
        all_metric_embeddings = self.temporal_encoder(batch.x)
        #print("all_metric_embeddings:", all_metric_embeddings.shape)
        metric_embeddings = self.get_instance_metric_tensor(all_metric_embeddings)
        #print("metric_embeddings:", type(metric_embeddings[0][45]))

        h_instances = self.instance_encoder(all_metric_embeddings, batch.edge_index, batch.batch)
        h_instances_prime = self.service_encoder(h_instances, self.edge_index, self.edge_weight)

        num_instances = len(instance_names)
        batch_size = h_instances.size(0) // num_instances
        h_instances_prime = h_instances_prime.view(batch_size, num_instances, -1)

        node_preds = self.decoder(h_instances_prime, instance_names, metric_embeddings)
        return node_preds

    def get_anomaly_score(self,  batch, instance_names, y_true):
        node_preds = self._encode_and_predict(batch, instance_names)
        anomaly_score = torch.abs(node_preds - y_true)
        return anomaly_score

    def forward(self, batch, instance_names, y_true):
        node_preds = self._encode_and_predict(batch, instance_names)
        causal_loss, adj, penalty = self.causal_graph_learner.loss_from_residual(node_preds)
        loss = F.mse_loss(node_preds, y_true)
        all_loss = loss + self.alpha * causal_loss + self.beta * penalty
        #将adj的边作为self.edge_index
        #self.edge_index = adj
        return loss

