"""
总体目标
1. 手动实现T5模型：参照t5-small的架构，使用PyTorch从零搭建模型。
2. 加载预训练权重：从Hugging Face的t5-small加载权重到我们的自定义模型。
3. 数据集：使用Hugging Face的wmt19（英文到中文）。
4. 输入输出处理：借助Hugging Face的T5Tokenizer和generate方法。
"""

## =============================================================== ##

"""
步骤 1：理解T5-small的架构
t5-small是一个基于Transformer的编码器-解码器模型，具体参数如下：
- 层数：编码器和解码器各6层。
- 隐藏层维度（d_model）：512。
- 注意力头数：8。
- 前馈网络维度（d_ff）：2048。
- 词汇表大小：32,128（基于SentencePiece）。
- 相对位置编码：T5使用相对位置编码而非绝对位置编码。
"""

## =============================================================== ##

"""
步骤 2：手动搭建T5模型

以下是用PyTorch实现的T5模型代码。我们会尽量贴近原始T5架构，但简化部分细节（例如相对位置编码的具体实现）。
"""
# 2.1 导入必要的库

import torch
import torch.nn as nn 
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset

# 2.2 实现T5模型

# 实现旋转位置编码
def apply_rotary_position_embeddings(q, k, max_seq_len, device):
    """
    应用旋转位置编码到Q和K上
    q, k: [batch_size, num_heads, seq_len, d_k]
    """
    # 生成旋转角度
    theta = torch.arange(0, max_seq_len, device=device).float() # [seq_len]
    theta = 10000**(-2 * (torch.arange(0, q.size(-1), 2, device=device).float())/q.size(-1)) # [d_k/2]
    angles = theta[None, None, :, None] * theta[None, None, None, :] # [1, 1, seq_len, d_k/2]
    
    # 分割Q和K的维度为[x1, x2]对
    q_ = q.view(*q.shape[:1], -1, 2) # [batch_size, num_heads, seq_len, d_k/2, 2]
    k_ = k.view(*k.shape[:-1], -1, 2)
    
    # 计算 sin 和 cos
    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)
    
    # 旋转操作：x1' = x1 * cos - x2 * sin, x2' = x1 * sin + x2 * cos
    q_rot = torch.stack([
        q_[..., 0] * cos_angles - q_[..., 1] * sin_angles,
        q_[..., 0] * sin_angles + q_[..., 1] * cos_angles
    ], dim=-1).view_as(q)
    k_rot = torch.stack([
        k_[..., 0] * cos_angles - k_[..., 1] * sin_angles,
        k_[..., 0] * sin_angles + k_[..., 1] * cos_angles
    ], dim=-1).view_as(k)
    
    return q_rot, k_rot

# 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须可以整除num_heads"
        self.d_model = d_model 
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, max_seq_len=None): 
        batch_size = q.size(0)
        
        # 线性变换
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 应用旋转位置编码
        if max_seq_len is not None: 
            q, k = apply_rotary_position_embeddings(q, k, max_seq_len, q.device)
    
        # 注意力计算
        scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(self.d_k)
        if mask is not None: 
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size,-1, self.d_model)
        
        return self.W_o(context)

# 前馈网络
class FeedForward(nn.Module): 
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x): 
        return self.linear2(self.relu(self.linear1(x)))
    
# 编码器层
class EncoderLayer(nn.Module): 
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model,num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, max_seq_len=None):
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask, max_seq_len)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x 
    
# 3. 解码器层（加入 KV Cache 和掩码逻辑）
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
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, max_seq_len=None, self_cache=None, cross_cache=None): 
        # 自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask, max_seq_len)
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_output, enc_output, src_mask)))
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x 

# T5模型
class T5Model(nn.Module): 
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None): 
        # 编码器
        enc_output = self.embedding(src)
        for layer in self.encoder_layers: 
            enc_output = layer(enc_output, src_mask)
        
        # 解码器
        dec_output = self.embedding(tgt)
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # 输入
        return self.output_layer(dec_output)

        
        
         
if __name__ == "__main__":
    # 步骤 3：加载预训练权重
    vocab_size = 32128 # t5-small的词汇表大小
    d_model = 512
    num_heads = 8
    d_ff = 2048
    num_layers = 6
    
    # 创建自定义模型实例
    custom_model = T5Model(vocab_size, d_model, num_heads, d_ff, num_layers)
    
    # 3.2 从Hugging Face加载权重
    official_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    
    # 将权重映射到自定义模型
    state_dict = official_model.state_dict()
    
    # state_dict1 = custom_model.state_dict()
    
    # for key,value in state_dict.items():
    #     print(f"key:{key}")
        
    # print("*"*20)
    
    # for key,value in state_dict1.items():
    #     print(f"key:{key}")
    
    # 调整权重名称以匹配自定义模型
    custom_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("encoder"):
            custom_key = key.replace("encoder", "encoder_layers").replace("block", "").replace("layer", "")
            custom_state_dict[custom_key] = value
        elif key.startswith("decoder"):
            custom_key = key.replace("decoder", "decoder_layers").replace("block", "").replace("layer", "")
            custom_state_dict[custom_key] = value
        elif key == "shared.weight":  # 共享的嵌入层
            custom_state_dict["embedding.weight"] = value
        elif key == "lm_head.weight":  # 输出层
            custom_state_dict["output_layer.weight"] = value
        else: 
            custom_state_dict["output_layer.weight"] = value
    
    # 加载权重到自定义模型
    custom_model.load_state_dict(custom_state_dict, strict=False)
    print("预训练权重加载完成！")
    
    # 加载数据集
    dataset = load_dataset("wmt19", "zh-en")
    small_dataset = dataset["train"].select(range(1000))  # 取1000条用于测试
    
    input_text = "translate English to Chinese: The weather is nice today."
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    
    # MultiHeadAttention(10, 2)
    # scores = torch.rand(size=(2, 4, 4)) 
    # mask = torch.zeros(2, 4, 4)
    # print(scores)
    # scores = scores.masked_fill(mask == 0, -1e9)
    # print(scores)
    # from torchsummary import summary  # 直接导入 summary

    # batch_size = 2
    # seq_len = 20
    # d_model = 64  # 修正拼写错误
    # input = torch.rand(size=(batch_size, seq_len, d_model)).to(dtype=torch.float32)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # layernorm = nn.LayerNorm(d_model).to(device)

    # for name, param in layernorm.named_parameters():
    #     print(f"参数名：{name}， 参数形状：{param.shape}, \n 参数tensor:{param}")
        
    #     """
    #     “先对 [b, s, :] 中的每个词（即 x[b, s, :]，一个长度为 d_model 的向量）计算均值和方差，然后对这 d_model 个元素进行归一化。
    #     接着，对归一化后的每个元素 x_normalized[b, s, i] 应用对应的 weight[i] 和 bias[i] 进行仿射变换。
    #     其中，在 d_model 维度上的相同序号 i，无论 (b, s) 如何变化，应用的 weight[i] 和 bias[i] 是相同的。”
    #     """