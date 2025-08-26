"""
FCSTGNN时空模块配置文件
"""

class FCSTGNNConfig:
    """FCSTGNN模型配置类"""
    
    def __init__(self):
        # 时空模块基础参数
        self.conv_out = 8  # 1D CNN输出通道数
        self.lstm_hidden_dim = 32  # LSTM隐藏层维度
        self.lstm_out_dim = 16  # LSTM输出维度
        self.conv_kernel = 3  # 卷积核大小
        
        # 图神经网络参数
        self.num_windows = 2  # 时间窗口数量
        self.moving_window = [3, 5]  # 多尺度滑动窗口大小
        self.stride = [1, 2]  # 滑动步长
        self.decay = 0.9  # 时间衰减因子
        self.pooling_choice = 'mean'  # 池化方式: 'mean' 或 'max'
        
        # 训练参数
        self.dropout = 0.1  # Dropout概率
        self.learning_rate = 0.001  # 学习率
        
        # 模型架构参数
        self.use_positional_encoding = True  # 是否使用位置编码
        self.use_batch_norm = True  # 是否使用批归一化
        
    def get_model_params(self, input_dim, hidden_dim, time_length, num_nodes):
        """
        获取模型参数字典
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            time_length: 时间序列长度
            num_nodes: 节点数量
            
        Returns:
            参数字典
        """
        return {
            'input_dim': input_dim,
            'conv_out': self.conv_out,
            'lstm_hidden_dim': self.lstm_hidden_dim,
            'lstm_out_dim': self.lstm_out_dim,
            'conv_kernel': self.conv_kernel,
            'hidden_dim': hidden_dim,
            'time_length': time_length,
            'num_node': num_nodes,
            'num_windows': self.num_windows,
            'moving_window': self.moving_window,
            'stride': self.stride,
            'decay': self.decay,
            'pooling_choice': self.pooling_choice,
            'dropout': self.dropout
        }

    def validate_params(self):
        """验证参数配置的有效性"""
        assert len(self.moving_window) == len(self.stride), "moving_window和stride长度必须相等"
        assert self.num_windows == len(self.moving_window), "num_windows必须等于moving_window的长度"
        assert all(w > 0 for w in self.moving_window), "所有窗口大小必须大于0"
        assert all(s > 0 for s in self.stride), "所有步长必须大于0"
        assert 0 < self.decay <= 1, "衰减因子必须在(0,1]范围内"
        assert self.pooling_choice in ['mean', 'max'], "池化方式必须是'mean'或'max'"
        assert 0 <= self.dropout <= 1, "Dropout概率必须在[0,1]范围内"


# 预定义配置
DEFAULT_CONFIG = FCSTGNNConfig()

# 小规模数据配置
SMALL_CONFIG = FCSTGNNConfig()
SMALL_CONFIG.lstm_hidden_dim = 16
SMALL_CONFIG.lstm_out_dim = 8
SMALL_CONFIG.conv_out = 4
SMALL_CONFIG.moving_window = [2, 3]
SMALL_CONFIG.num_windows = 2

# 大规模数据配置
LARGE_CONFIG = FCSTGNNConfig()
LARGE_CONFIG.lstm_hidden_dim = 64
LARGE_CONFIG.lstm_out_dim = 32
LARGE_CONFIG.conv_out = 16
LARGE_CONFIG.moving_window = [5, 7, 9]
LARGE_CONFIG.stride = [1, 2, 3]
LARGE_CONFIG.num_windows = 3
