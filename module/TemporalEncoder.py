import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, window_size, hidden_dim, num_layers=2, nhead=2, dropout=0.1):
        super(TemporalEncoder, self).__init__()
        self.input_proj = nn.Linear(window_size, hidden_dim)  # [6, 5] -> [6, hidden_dim]
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)  # 可选

    def forward(self, x):
        # x: [bs, seq_num, window_num, window_size]
        bs, seq_num, window_num, window_size = x.shape
        # 先把每个时间窗口投影到hidden_dim
        x = self.input_proj(x)  # [bs, seq_num, window_num, hidden_dim]
        # 合并batch和seq_num，方便送入transformer
        x = x.view(bs * seq_num, window_num, -1)  # [bs*seq_num, window_num, hidden_dim]
        # transformer编码
        x = self.transformer_encoder(x)  # [bs*seq_num, window_num, hidden_dim]
        # 取最后一个时间步的输出（也可以mean pooling等）
        x = x[:, -1, :]  # [bs*seq_num, hidden_dim]
        # 恢复batch和seq_num
        x = x.view(bs, seq_num, -1)  # [bs, seq_num, hidden_dim]
        x = self.output_proj(x)  # [bs, seq_num, hidden_dim]
        return x
