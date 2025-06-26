import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    def __init__(self, input_length, hidden_dim, num_layers=3, nhead=4, dropout=0.1):
        super().__init__()
        self.input_length = input_length
        self.hidden_dim = hidden_dim
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, input_length, hidden_dim))
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输入投影层
        self.input_proj = nn.Linear(1, hidden_dim)
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x: [N, W] -> [N, W, 1]
        x = x.unsqueeze(-1)
        # print("TE input x shape:", x.shape)
        # print("TE input x nan:", torch.isnan(x).any())
        #是否有大于2的数
        # print("TE input x max:", x.max().item())
        # print("TE input x sample:", x.flatten()[:10])
        
        # 投影到hidden_dim维度
        x = self.input_proj(x)  # [N, W, H]
        # print("after input_proj nan:", torch.isnan(x).any())
        # print("after input_proj sample:", x.flatten()[:10])
        
        # 添加位置编码
        x = x + self.pos_encoder
        
        # Transformer编码
        x = self.transformer_encoder(x)  # [N, W, H]
        # print("after transformer_encoder nan:", torch.isnan(x).any())
        # print("after transformer_encoder sample:", x.flatten()[:10])
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]  # [N, H]
        
        # 最终投影
        x = self.output_proj(x)
        
        #print("TE input x min/max/mean:", x.min().item(), x.max().item(), x.mean().item())
        
        return x

class TemporalEncoder(nn.Module):
    def __init__(self, input_length, hidden_dim, num_layers=3, nhead=4, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(
            input_length=input_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=nhead,
            dropout=dropout
        )

    def forward(self, x):
        # x: [N, W]
        #print("x shape before transformer:", x.shape)
        return self.encoder(x)  # 输出: [N, hidden_dim]