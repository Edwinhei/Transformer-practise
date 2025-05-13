"""
Transformer 模型概述

Transformer 模型由编码器（Encoder）和解码器（Decoder）两部分组成

每部分包含多层堆叠的 Transformer 层

每个 Transformer 层包含以下核心组件：
    - 多头自注意力（Multi-Head Self-Attention）：捕获序列内部的依赖关系。
    - 前馈网络（Feed-Forward Network, FFN）：对每个位置的特征进行非线性变换。
    - 残差连接（Residual Connection） 和 层归一化（Layer Normalization）：稳定训练并加速收敛。
    
此外，Transformer 还有两个重要机制：
    - 位置编码（Positional Encoding）：为输入序列添加位置信息，因为 Transformer 没有天然的序列顺序感知能力。
    - 掩码（Mask）：在解码器中防止关注未来信息，或屏蔽填充位置。
"""

from matplotlib import pyplot as plt
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import math 
import seaborn as sns 

# Step 1: 实现多头自注意力（Multi-Head Self-Attention）
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model  # 模型维度
        self.num_heads = num_heads  # 注意力头数
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Q、K、V 的线性变换层
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # 输出线性变换
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)  # 缩放因子
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换生成 Q、K、V
        Q = self.W_q(query)  # [batch_size, seq_len, d_model]
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 分割成多头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        # 应用掩码（可选）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 归一化并加权求和
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # 拼接多头输出并线性变换
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.W_o(out)
        
        return out
   
# Step 2: 实现前馈网络（Feed-Forward Network, FFN）
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        out = self.linear1(x)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out
    
# Step 3: 实现 Transformer 编码器层（Encoder Layer）
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 多头注意力 + 残差 + 归一化
        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN + 残差 + 归一化
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x
    
"""
Step 4: 实现 Transformer 解码器层（Decoder Layer）
解码器层比编码器层多一个多头注意力子层，用于关注编码器的输出。
原理
    - 解码器自注意力：带因果掩码，防止关注未来信息。
    - 编码器-解码器注意力：解码器关注编码器的输出。
"""
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力 + 残差 + 归一化（带因果掩码）
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        # 编码器-解码器注意力 + 残差 + 归一化
        enc_dec_attn_out = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(enc_dec_attn_out))
        
        # FFN + 残差 + 归一化
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        
        return x

# Step 5: 实现位置编码（Positional Encoding）
# Transformer 不含位置信息，需通过位置编码为序列添加位置感知能力。
# 正弦位置编码：

# 这里有疑问：为什么在d_model中，偶数元素使用sin函数，奇数元素使用cos函数。
# 为什么就不能全部使用sin函数呢？或者cos函数呢？这有什么原理吗？
# 能否通过可视化代码方式给我讲明白呢？
# 这里我还无法感知到，位置信息嵌入了吗？每个位置的表示是不是都不相同？
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
    
# Step 6: 实现完整的 Transformer 模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        # 解码器
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # 输出层
        output = self.output_linear(dec_output)
        return output
    
# Step 7: 实现掩码（Mask）
"""
    - 填充掩码（Padding Mask）：屏蔽填充位置。
    - 因果掩码（Causal Mask）：在解码器中防止关注未来位置。
""" 
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
    return mask

def create_masks(src, tgt, pad_idx):
    """
    创建 Transformer 所需的掩码。
    - src_mask：用于编码器和解码器的编码器-解码器注意力，屏蔽源序列的填充位置。
    - tgt_mask：用于解码器自注意力，结合填充掩码和因果掩码。
    """
    # src_mask: [batch_size, 1, 1, src_seq_len]
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    
    # tgt_mask: [batch_size, 1, tgt_seq_len, tgt_seq_len]
    tgt_padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, tgt_seq_len]
    tgt_subsequent_mask = generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)  # [tgt_seq_len, tgt_seq_len]
    tgt_mask = tgt_padding_mask & tgt_subsequent_mask.unsqueeze(0).unsqueeze(0)  # 广播并结合
    
    return src_mask, tgt_mask

if __name__ == "__main__":
    src = torch.full(size=(4, 32), fill_value=10)
    tgt = torch.full(size=(4, 32), fill_value=20)
    create_masks(src, tgt, pad_idx)