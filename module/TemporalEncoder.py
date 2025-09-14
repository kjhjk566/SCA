import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim, output_dim=None, num_layers=2, nhead=2, dropout=0.1, pooling="last"):
        """
        Args:
            hidden_dim: Transformer hidden size
            output_dim: 最终输出维度 (默认 = hidden_dim)
            num_layers: number of TransformerEncoder layers
            nhead: number of attention heads
            dropout: dropout rate
            pooling: "last", "mean" or "attention"
        """
        super(TemporalEncoder, self).__init__()
        self.pooling = pooling
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        
        # 把单点(1)特征投影到 hidden_dim
        self.input_proj = nn.Linear(1, hidden_dim)  
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层 (hidden_dim -> output_dim)
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)

        # attention pooling 参数
        if pooling == "attention":
            self.attention_vector = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x):
        # x: [bs, num_metrics, seq_len]
        bs, num_metrics, seq_len = x.shape

        # 加一维: [bs*num_metrics, seq_len, 1]
        x = x.view(bs * num_metrics, seq_len, 1)

        # 投影到 hidden_dim: [bs*num_metrics, seq_len, hidden_dim]
        x = self.input_proj(x)

        # Transformer 编码: [bs*num_metrics, seq_len, hidden_dim]
        x = self.transformer_encoder(x)

        # ---- pooling ----
        if self.pooling == "last":
            x = x[:, -1, :]  # [bs*num_metrics, hidden_dim]

        elif self.pooling == "mean":
            x = x.mean(dim=1)  # [bs*num_metrics, hidden_dim]

        elif self.pooling == "attention":
            attn_scores = torch.matmul(x, self.attention_vector)  # [bs*num_metrics, seq_len]
            attn_weights = F.softmax(attn_scores, dim=1)
            x = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [bs*num_metrics, hidden_dim]

        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")

        # 恢复 batch & num_metrics
        x = x.view(bs, num_metrics, -1)  # [bs, num_metrics, hidden_dim]

        return self.output_proj(x)  # [bs, num_metrics, output_dim]