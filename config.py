
import torch
import os

class Config:
    def __init__(self,data_set):
        self.model_name = "new_model"
        self.model_path = "path/to/model"
        self.data_set = data_set
        self.max_length = 512
        self.batch_size = 16
        self.learning_rate = 5e-5
        self.epochs = 3
        self.topk = 5
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42
        if data_set == "aiops22":
            self.all_enum = {'emailservice-0': 0, 'emailservice2-0': 1, 'emailservice-2': 2, 'emailservice-1': 3, 'adservice-0': 4, 'adservice2-0': 5, 'adservice-2': 6, 'adservice-1': 7, 'checkoutservice-0': 8, 'checkoutservice2-0': 9, 'checkoutservice-2': 10, 'checkoutservice-1': 11, 'paymentservice-0': 12, 'paymentservice2-0': 13, 'paymentservice-2': 14, 'paymentservice-1': 15, 'productcatalogservice-0': 16, 'productcatalogservice2-0': 17, 'productcatalogservice-2': 18, 'productcatalogservice-1': 19, 'shippingservice-0': 20, 'shippingservice2-0': 21, 'shippingservice-2': 22, 'shippingservice-1': 23, 'frontend-0': 24, 'frontend2-0': 25, 'frontend-2': 26, 'frontend-1': 27, 'recommendationservice-0': 28, 'recommendationservice2-0': 29, 'recommendationservice-2': 30, 'recommendationservice-1': 31, 'cartservice-0': 32, 'cartservice2-0': 33, 'cartservice-2': 34, 'cartservice-1': 35, 'currencyservice-0': 36, 'currencyservice2-0': 37, 'currencyservice-2': 38, 'currencyservice-1': 39, 'node-1': 40, 'node-2': 41, 'node-3': 42, 'node-4': 43, 'node-5': 44, 'node-6': 45}
        elif data_set == "aiops25":
            self.all_enum = {'adservice-0': 0, 'adservice-1': 1, 'adservice-2': 2, 'aiops-k8s-01': 3, 'aiops-k8s-02': 4, 'aiops-k8s-03': 5, 'aiops-k8s-04': 6, 'aiops-k8s-05': 7, 'aiops-k8s-06': 8, 'aiops-k8s-07': 9, 'aiops-k8s-08': 10, 'cartservice-0': 11, 'cartservice-1': 12, 'cartservice-2': 13, 'checkoutservice-0': 14, 'checkoutservice-1': 15, 'checkoutservice-2': 16, 'currencyservice-0': 17, 'currencyservice-1': 18, 'currencyservice-2': 19, 'emailservice-0': 20, 'emailservice-1': 21, 'emailservice-2': 22, 'frontend-0': 23, 'frontend-1': 24, 'frontend-2': 25, 'k8s-master1': 26, 'k8s-master2': 27, 'k8s-master3': 28, 'paymentservice-0': 29, 'paymentservice-1': 30, 'paymentservice-2': 31, 'productcatalogservice-0': 32, 'productcatalogservice-1': 33, 'productcatalogservice-2': 34, 'recommendationservice-0': 35, 'recommendationservice-1': 36, 'recommendationservice-2': 37, 'redis-cart-0': 38, 'shippingservice-0': 39, 'shippingservice-1': 40, 'shippingservice-2': 41}
        # self.instance_enum = {'emailservice-0': 0, 'emailservice2-0': 1, 'emailservice-2': 2, 'emailservice-1': 3, 'adservice-0': 4, 'adservice2-0': 5, 'adservice-2': 6, 'adservice-1': 7, 'checkoutservice-0': 8, 'checkoutservice2-0': 9, 'checkoutservice-2': 10, 'checkoutservice-1': 11, 'paymentservice-0': 12, 'paymentservice2-0': 13, 'paymentservice-2': 14, 'paymentservice-1': 15, 'productcatalogservice-0': 16, 'productcatalogservice2-0': 17, 'productcatalogservice-2': 18, 'productcatalogservice-1': 19, 'shippingservice-0': 20, 'shippingservice2-0': 21, 'shippingservice-2': 22, 'shippingservice-1': 23, 'frontend-0': 24, 'frontend2-0': 25, 'frontend-2': 26, 'frontend-1': 27, 'recommendationservice-0': 28, 'recommendationservice2-0': 29, 'recommendationservice-2': 30, 'recommendationservice-1': 31, 'cartservice-0': 32, 'cartservice2-0': 33, 'cartservice-2': 34, 'cartservice-1': 35, 'currencyservice-0': 36, 'currencyservice2-0': 37, 'currencyservice-2': 38, 'currencyservice-1': 39}
        # self.node_enum = {'node-1': 0, 'node-2': 1, 'node-3': 2, 'node-4': 3, 'node-5': 4, 'node-6': 5}
