import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
class StaticCallGraphBuilder:
    def __init__(self, trace_data,deploy_data,config):
        self.trace_data = trace_data
        self.config = config
        self.call_matrix = None
        self.deploy_data = deploy_data
        self.call_graph = call_matrix = np.zeros((len(self.config.all_enum), len(self.config.all_enum)))

    def get_call_graph(self):
        all_enum = self.config.all_enum
        call_matrix = self.call_graph

        # 遍历调用链trace
        if self.trace_data is not None:
            for src_name, dst_name in self.trace_data:
            
                if src_name in all_enum and dst_name in all_enum:
                    src_idx = all_enum[src_name]
                    dst_idx = all_enum[dst_name]
                    call_matrix[src_idx, dst_idx] = 1

        # 遍历部署关系deploy
        if self.deploy_data is not None:
            for node_name, services in self.deploy_data:
                for service_name in services:
                    if node_name in all_enum and service_name in all_enum:
                        node_idx = all_enum[node_name]
                        service_idx = all_enum[service_name]
                        call_matrix[node_idx, service_idx] = 1

        self.call_matrix = call_matrix
        return call_matrix

        

    def get_callees(self, function):
        return self.call_graph.get(function, [])