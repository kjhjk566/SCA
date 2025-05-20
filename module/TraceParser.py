import pandas as pd
from tqdm import tqdm
import pickle


class TraceParser:
    def __init__(self,trace_data,deploy_data):
        self.parsed_trace = None
        self.deploy_edges = None
        self.trace_data = trace_data
        self.deploy_data = deploy_data

    def parse_trace(self,save_path=None):
        if save_path is not None:
            # 如果提供了保存路径，则尝试加载数据
            try:
                with open(save_path, 'rb') as f:
                    self.parsed_trace = pickle.load(f)
                print("数据加载成功")
                return
            except FileNotFoundError:
                print("数据文件未找到，开始解析数据")
        
        call_edges = set()
        for _, row in tqdm(self.trace_data.iterrows(), total=len(self.trace_data), desc="提取调用边"):
            parent = row.get("parent_cmdb_id")
            child = row.get("cmdb_id")
            if pd.notna(parent) and pd.notna(child):
                parent_service = parent
                child_service = child
                if parent_service != child_service:  # 去掉自环
                    call_edges.add((parent_service, child_service))

            # 去重后输出
        call_edges = sorted(call_edges)
        self.parsed_trace = call_edges
        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(self.parsed_trace, f)
            print(f"数据已保存到 {save_path}")
        else:
            print("解析完成，未保存数据")
            
    def extract_node_service_mapping(self,save_path=None):
        """
        从指标数据DataFrame中提取node与service之间的部署关系
        :param dataframe: 包含 'cmdb_id' 列的数据
        :param save_path: 可选，保存映射的路径
        :return: list，包含 (node_name, service_name) 的元组
        """
        if save_path is not None:
            try:
                with open(save_path, 'rb') as f:
                    node_service_list = pickle.load(f)
                print(f"部署关系加载成功：{save_path}")
                self.deploy_edges = node_service_list
                return node_service_list
            except FileNotFoundError:
                print("部署关系文件未找到，开始解析数据")
        
        node_service_map = dict()

        for cmdb_id in self.deploy_data['cmdb_id'].dropna().unique():
            try:
                node_name, service_name = cmdb_id.split('.', 1)  # 以第一个'.'分割
                if node_name not in node_service_map:
                    node_service_map[node_name] = set()
                node_service_map[node_name].add(service_name)
            except ValueError:
                # 防止异常，万一cmdb_id没有点分割
                continue

        node_service_list = [(node, service) for node, services in node_service_map.items() for service in services]

        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(node_service_list, f)
            print(f"部署关系已保存到 {save_path}")

        self.deploy_edges = node_service_list
        

if __name__ == "__main__":
    df = pd.read_csv('/home/kuangjunhua/research/data/aiops2022-pre/2022-05-05/cloudbed/trace/all/trace_jaeger-span_new_2.csv')
    
    parser = TraceParser(df)
    parser.parse_trace('/home/kuangjunhua/research/data/aiops22_dataset/5/trace_jaeger-span_new_2.pkl')