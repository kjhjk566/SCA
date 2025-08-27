from ast import arg
import sys
import os

from sympy import root


sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 提到最上面！！

import pandas as pd
import argparse
from utils.visualization import plot_metrics_comparison
from visualize import visualize_instance_reconstruction
from config import Config
from module.TraceParser import TraceParser
from module.DataProcessor import DataProcessor,TimeWindowDataset
from module.graph.StaticCallGraphBuilder import StaticCallGraphBuilder
from module.graph.InstanceGraphEncoder import InstanceGraphEncoder
from module.graph.ServiceGraphEncoder import ServiceGraphEncoder
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader
from module.NodeDecoder import NodeDecoder
from model.FullGraphModel import FullGraphRCA
from model.SCA_test import SCA
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pickle
import numpy as np
from module.RootCauseScorer import RootCauseScorer
#day = 7



def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='根因分析模型训练和测试')
    parser.add_argument('-M', '--mode', type=str, default='train', choices=['train', 'test'],
                      help='运行模式：train 或 test')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='训练时的批次大小')
    parser.add_argument('--epochs', type=int, default=2,
                      help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='学习率')
    parser.add_argument('--window_size', type=int, default=20,
                      help='时间窗口大小')
    parser.add_argument('--stride', type=int, default=1,
                      help='时间窗口步长')
    parser.add_argument('-ds','--dataset', type=str, default="aiops25",
                      help='')
    parser.add_argument('-dr','--data_range', type=str, default='day',choices=['all','day'],
                      help='')
    parser.add_argument('-D','--day', type=str, default='20',
                      help='数据集日期')
    return parser.parse_args()

def evaluate_topk_accuracy(all_ans, all_labels, topk_list=[1,3, 5]):
    """
    计算 Top-K 准确率，支持前缀匹配。
    
    参数：
        all_ans: List[List[str]]，每个样本的预测 Top-K 根因指标（如 ['frontend-2', 'adservice-1', ...]）
        all_labels: List[List[str]]，每个样本的真实根因标签（如 ['frontend']）
        topk_list: List[int]，要评估的 K 值（如 [1, 5]）
        
    返回：
        dict: {k: accuracy}，如 {1: 0.75, 5: 0.92}
    """
    results = {k: 0 for k in topk_list}
    total = len(all_ans)

    for pred_list, label in zip(all_ans, all_labels):
        # 确保 label 是列表格式
        label = [label] if isinstance(label, str) else label
        for k in topk_list:
            topk_pred = pred_list[:k]
            hit = any(
                any(p.startswith(l) for p in topk_pred) for l in label
            )
            if hit:
                results[k] += 1

    return {k: results[k] / total for k in topk_list}

def print_prediction_result(case_id, pred_list, true_label, is_correct):
    """
    打印单个case的预测结果
    """
    print(f"\n{'='*50}")
    print(f"Case {case_id} 预测结果:")
    print(f"预测的根因服务: {pred_list}")
    print(f"真实的根因服务: {true_label}")
    print(f"预测是否正确: {'✓' if is_correct else '✗'}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    print("运行模式:", args.mode)
    if args.data_range == 'all':
        args.data_path = f'/home/kuangjunhua/research/data/{args.dataset}'
        args.model_path = f'/home/kuangjunhua/research/new_method/model_save/model_{args.dataset}_all.pth'

    elif args.data_range == 'day':
        args.data_path = f'/home/kuangjunhua/research/data/{args.dataset}/{args.day}'
        args.model_path = f'/home/kuangjunhua/research/SCA/model_save/model_{args.dataset}_{args.day}.pth'
    print("数据路径:", args.data_path)
    day = args.day


    # 读取数据
    config = Config(args.dataset)
    config.batch_size = args.batch_size
    trace_df = None
    deploy_df = None
    # trace_df = pd.read_csv(os.path.join(args.data_path, 'trace_jaeger-span_new_2.csv'))
    # deploy_df = pd.read_csv(f'/home/kuangjunhua/research/data/aiops2022-pre/2022-05-0{day}/cloudbed/metric/container/kpi_container_cpu_cfs_periods.csv')
    
    # parser = TraceParser(trace_df,deploy_df)
    # #parser.parse_trace(os.path.join(args.data_path, 'trace_jaeger-span_new_2.pkl'))
    # parser.extract_node_service_mapping(os.path.join(args.data_path, 'deploy.pkl'))
    
    # s = StaticCallGraphBuilder(parser.parsed_trace,parser.deploy_edges, config)
    # call_graph = s.get_call_graph()
    
    #原始数据
    # if args.dataset == 'all':
    #     path = '/home/kuangjunhua/research/data/aiops22_dataset/train_df.pkl'
    # elif args.dataset == 'normal':
    #     path = os.path.join(args.data_path, 'normal_data.pkl')
    path = os.path.join(args.data_path, 'normal_data.pkl')
    
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print("raw_data.shape:",data.shape)
    #数据是否有nan
    print("数据是否有nan:",data.isna().sum().sum())
    
    
    data_processor = DataProcessor(data, config, window_size=args.window_size, stride=args.stride)
    config.instance_metric_count_dict = data_processor.instance_metric_count_dict
    #print(config.instance_metric_count_dict)
    train_dataset = TimeWindowDataset(data_processor, config)
    #print("第一个数据样本:")
    #print(train_dataset[0])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    for x_window, instance_names, y_window in train_loader:
        print("x_window:", x_window.shape)
        #print("instance_names:", instance_names)
        print("y_window:", y_window.shape)
        metric_num = y_window.shape[1]
        break
    model = SCA(config, metric_num = metric_num,input_dim=x_window.shape[-1], hidden_dim=64, sca_hidden_dim=64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    root_cause_scorer = RootCauseScorer(alpha=1.0, beta=0.01,config=config)

    if args.mode == 'train':
        print("\n开始训练...")
        for epoch in tqdm(range(args.epochs), desc="Epochs"):
            epoch_loss = 0.0
            batch_count = 0
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
            for x_window, instance_names, y_window in batch_pbar:
                x_window = x_window.to(device)
                y_window = y_window.to(device)
                loss = model(x_window, y_window)

                epoch_loss += loss.item()
                batch_count += 1
                
                # 更新batch进度条显示的loss
                batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # 计算并输出每个epoch的平均loss
            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
            
        torch.save(model.state_dict(), args.model_path)
        print(f"模型已保存到: {args.model_path}")
    # 方法1: 查看特定实例
    viz_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    

    #加载模型
    model = SCA(config, metric_num=metric_num,input_dim=x_window.shape[-1], hidden_dim=64, sca_hidden_dim=64).to(device)
    model.load_state_dict(torch.load(args.model_path),strict=False)

    model.to(device)
    if args.dataset == 'all':
        path = '/home/kuangjunhua/research/data/aiops22_dataset/case_data.pkl'
    elif args.dataset == 'normal':
        path = os.path.join(args.data_path, 'case_20min_data.pkl')
    case_path = os.path.join(args.data_path, 'case_20min_data.pkl')
    with open(case_path, 'rb') as f:
        test_case = pickle.load(f)

    model.eval()
    all_ans = []
    all_labels = []
    

    
    print("\n开始测试...")
    failed_cases = []  # 存储预测失败的案例
    case_pic_file = '/home/kuangjunhua/research/SCA/case_pic'
    with open('/home/kuangjunhua/research/data/aiops25/20/deploy_graph.pkl', 'rb') as f:
        graph = pickle.load(f)
    for case_id, (case_data, label, ts) in enumerate(tqdm(test_case)):
        all_labels.append(label)

        data_processor = DataProcessor(case_data, config, window_size=args.window_size, stride=args.stride)
        test_dataset = TimeWindowDataset(data_processor, config)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        #print(len(test_loader))
        
        #print("case_data:",case_data.shape)
        if case_data.shape[0]<args.window_size:
            print(f"Case {case_id+1} 数据长度不足，跳过该案例。")
            continue
        all_score = None
        for x_window, instance_names, y_window in test_loader:
           
            x_window = x_window.to(device)
            y_window = y_window.to(device)
            y_window = y_window[0].to(device)
            
            anomaly_score = model.calculate_metric_loss(x_window, y_window)
            if all_score is None:
                all_score = anomaly_score
            else:
                all_score = all_score + anomaly_score
         


        all_score = all_score / len(test_loader)
        #修改为一维张量
        all_score = all_score.squeeze(-1).cpu()
        #print('all_score',all_score)

        ans = root_cause_scorer.get_ans_from_loss(all_score)
        
        #ans = root_cause_scorer.get_root_cause_by_walk(all_score, graph)
        all_ans.append(ans)
        
        # 检查预测是否正确
        # print("ans:",ans)
        # print("label:",label)
        is_correct = any(any(pred.startswith(true) for pred in ans) for true in ([label] if isinstance(label, str) else label))
        
        # 打印预测结果
        print_prediction_result(case_id, ans, label, is_correct)
      

       

    # 计算并打印总体准确率
    accuracy = evaluate_topk_accuracy(all_ans, all_labels, topk_list=[1, 5])
    print("\n总体评估结果:")
    print(f"Top-1 准确率: {accuracy[1]:.2%}")
    print(f"Top-5 准确率: {accuracy[5]:.2%}")
    print(f"day:{day}")

    