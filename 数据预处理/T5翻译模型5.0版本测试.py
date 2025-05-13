import math
from datasets import load_dataset
import torch
import torch.nn as nn 
from typing import Tuple
from flash_attn import flash_attn_func

dataset = load_dataset("opus100", "en-zh")

class T5Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x)
    
    


# 旋转位置编码
class RotaryEmbedding(nn.Module): 
    # 旋转位置编码实现
    def __init__(self, dim: int, max_position: int = 10000): 
        super().__init__()
        assert dim % 2 == 0, "dim必须是偶数"
        self.dim = dim
        self.max_position = max_position
        
        # 预计算频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 预计算所有位置的cos和sin
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """预计算最大长度的cos和sin"""
        positions = torch.arange(self.max_position, dtype=torch.float)
        freqs = positions[:, None] * self.inv_freq[None, :]  # (max_position, dim/2)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.register_buffer("cos_cached", cos, persistent=False)  # (max_position, dim/2)
        self.register_buffer("sin_cached", sin, persistent=False)
    
    def forward(self, seq_len: int, device: torch.device) -> tuple:
        """
        在训练端，我们固定最大尺寸输入是ok的。因为我们为了训练，对传进来的序列长度对齐的。设定了最大长度规则，截断规则。
        但是在预测段，最大固定尺寸max_position需要设置的大一些。大部分任务是单样本推理。此时，如果max_position设置过小，可能对生成结果有影响
        我们，一般可以和对话最大字符度相同。或者略低。
        """
        
        """根据序列长度返回对应的cos和sin"""
        assert seq_len <= self.max_position, f"seq_len ({seq_len}) 超过 max_position ({self.max_position})"
        
        # 从缓存中截取需要的部分
        cos = self.cos_cached[:seq_len].to(device)
        sin = self.sin_cached[:seq_len].to(device)
        return cos, sin
    
# 在单向or双向自注意力时可以这样操作。因为q，k同维度，当涉及交叉注意力时，此时就不能这么操作了，在训练时，由于我们设置max_length都一样长，可能并无感知。
# 但在训练时，编码端长度和解码端长度大多数情况下是未对齐的，此时就会报错。 
def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple:
    """应用旋转位置编码"""
    # 调整 cos 和 sin 的形状以匹配 q 和 k
    # cos, sin 原本shape是 (seq_len, d_k//2)
    cos = cos.unsqueeze(-2)  # 扩展维度为(seq_len, 1, d_k//2) 最后通过广播机制维度扩展(batch_size, seq_len, num_heads, d_k//2)
    sin = sin.unsqueeze(-2)  # 扩展维度为(seq_len, 1, d_k//2) 最后通过广播机制维度扩展(batch_size, seq_len, num_heads, d_k//2)
    q_ = q.float()
    k_ = k.float()
    trunc = q_.shape[-1] // 2
    
    q_rot = torch.cat([q_[..., :trunc] * cos - q_[..., trunc:] * sin,
                      q_[..., :trunc] * sin + q_[..., trunc:] * cos], dim=-1)
    k_rot = torch.cat([k_[..., :trunc] * cos - k_[..., trunc:] * sin,
                      k_[..., :trunc] * sin + k_[..., trunc:] * cos], dim=-1)
    return q_rot.type_as(q), k_rot.type_as(k)

# 在交叉注意力时，由于q，k的长度不一致。如果我们用常规的ROPE就不行，维度没有对齐。因此，在交叉注意力计算时，我们需要计算各自的ROPE值
def apply_rotary_emb_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """应用旋转位置编码到单个张量"""
    cos = cos.unsqueeze(-2)  # 扩展维度为(seq_len, 1, d_k//2) 最后通过广播机制维度扩展(batch_size, seq_len, num_heads, d_k//2)
    sin = sin.unsqueeze(-2)  # 扩展维度为(seq_len, 1, d_k//2) 最后通过广播机制维度扩展(batch_size, seq_len, num_heads, d_k//2)
    x_ = x.float()
    trunc = x_.shape[-1] // 2
    x_rot = torch.cat([
        x_[..., :trunc] * cos - x_[..., trunc:] * sin,
        x_[..., :trunc] * sin + x_[..., trunc:] * cos
    ], dim=-1)
    return x_rot.type_as(x)


class MultiHeadAttention(nn.Module): 
    def __init__(self, d_model, num_heads, max_position=10000, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须被num_heads整除"
        self.dropout = dropout
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)    
        self.k_linear = nn.Linear(d_model, d_model)    
        self.v_linear = nn.Linear(d_model, d_model)    
        self.out_linear = nn.Linear(d_model, d_model)
        
        # 初始化旋转位置编码
        self.rotary_emb = RotaryEmbedding(self.d_k, max_position) 
        
    # 始终应用掩码（如果 mask 为 None，则传入全 1 掩码） 这样设计的目的是，方便模型后续输出为onnx格式，该格式不建议if for等语句
    # src_mask shape可以是：(batch_size, 1, 1, seq_len)， 通过广播同步维度到 (batch_size, num_heads, seq_len, seq_len)
    # tgt_mask shape可以是：(batch_size, 1, seq_len, seq_len)， 通过广播同步维度到 (batch_size, num_heads, seq_len, seq_len)
    def forward(self, q, k, v, mask=None, use_flash_attn=False): 
        batch_size = q.size(0)
        q_seq_len = q.size(1)  # q 的序列长度
        k_seq_len = k.size(1)  # k 的序列长度
        
        # 线性变换并分割为多头
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k)
        
        # 为 q 和 k 分别生成旋转位置编码
        cos_q, sin_q = self.rotary_emb(q_seq_len, q.device)
        cos_k, sin_k = self.rotary_emb(k_seq_len, k.device)

        # 分别应用旋转位置编码
        q = apply_rotary_emb_single(q, cos_q, sin_q)
        k = apply_rotary_emb_single(k, cos_k, sin_k)


        # 单双向自注意力时可以使用，交叉注意力时，需要分别计算。
        # q, k = apply_rotary_emb(q, k, cos, sin)
        
        if use_flash_attn:
            # multi-attention 训练模式
            # multi-attention 要求输入格式为 (batch_size, num_heads, seq_len, d_k)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
            
            # -1e9在半精度等训练时，会溢出，这里选择-1e4
            # scores = scores.masked_fill(mask==0, -1e9)
            scores = scores.masked_fill(mask==0, -1e4)
            attention = torch.softmax(scores, dim=-1)
            output = torch.matmul(attention, v)
            output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        else: 
            # flash-attention 推理模式
            # Flash Attention 要求输入格式为 (batch_size, seq_len, num_heads, d_k)
            output = flash_attn_func(
                q, k, v,
                dropout_p=0, # 推理时，不使用dropout
                softmax_scale=1. / math.sqrt(self.d_k), # 缩放因子
                causal=True, # 内置生成因果掩码，但是他不对ipaddng等处理。所以，在训练端不合适
                # 在推理端生成任务时，由于没有padding的影响。解码端或gpt等使用内置掩码即可，如果是编解码，编码端还是需要传入attention_mask
            )             
            output = output.view(batch_size, -1, self.d_model)
        
        return self.out_linear(output)
    

class FeedForward(nn.Module): 
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x): 
        return self.linear2(torch.relu(self.linear1(x)))
    
class EncoderLayer(nn.Module): 
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x 

class DecoderLayer(nn.Module): 
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model) 
        self.norm2 = nn.LayerNorm(d_model) 
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, use_flash_attn=False): 
         x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask, use_flash_attn))
         x = x + self.dropout(self.cross_attn(self.norm2(x), enc_output, enc_output, src_mask, use_flash_attn))
         x = x + self.dropout(self.ff(self.norm3(x)))
         return x 
     
