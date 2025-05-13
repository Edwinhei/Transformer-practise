# import torch
# import math
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import torch.nn as nn 

# # 位置编码类（与你的代码相同）
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer("pe", pe)
    
#     def forward(self, x):
#         seq_len = x.size(1)
#         x = x + self.pe[:, :seq_len, :]
#         return x

# # 参数
# d_model = 128
# max_len = 50  # 只可视化前 50 个位置

# # 初始化位置编码
# pe_module = PositionalEncoding(d_model, max_len)
# pe = pe_module.pe.squeeze(0).numpy()  # [max_len, d_model] -> [50, 128]

# # 可视化 1：位置编码的热图
# plt.figure(figsize=(15, 5))
# sns.heatmap(pe, cmap='RdBu', vmin=-1, vmax=1)
# plt.title("Positional Encoding (d_model=128, max_len=50)")
# plt.xlabel("Dimension (d_model)")
# plt.ylabel("Position (pos)")
# plt.show()

# # 可视化 2：不同位置的编码向量
# positions = [0, 1, 10, 20, 30]  # 选择几个位置
# plt.figure(figsize=(10, 5))
# for pos in positions:
#     plt.plot(pe[pos], label=f"Position {pos}")
# plt.title("Positional Encoding Vectors at Different Positions")
# plt.xlabel("Dimension (d_model)")
# plt.ylabel("Value")
# plt.legend()
# plt.show()

# # 可视化 3：不同维度的 sin/cos 变化
# dims = [0, 1, 20, 21, 40, 41]  # 选择几个维度（偶数和奇数）
# plt.figure(figsize=(10, 5))
# for dim in dims:
#     plt.plot(pe[:, dim], label=f"Dimension {dim}")
# plt.title("Positional Encoding Values Across Positions for Different Dimensions")
# plt.xlabel("Position (pos)")
# plt.ylabel("Value")
# plt.legend()
# plt.show()


# import torch
# import math
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import torch.nn as nn 

# # 位置编码类
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer("pe", pe)
    
#     def forward(self, x):
#         seq_len = x.size(1)
#         x = x + self.pe[:, :seq_len, :]
#         return x

# # 参数
# d_model = 128
# max_len = 50

# # 初始化位置编码
# pe_module = PositionalEncoding(d_model, max_len)
# pe = pe_module.pe.squeeze(0).numpy()  # [max_len, d_model] -> [50, 128]

# # 可视化 1：不同维度的值随位置变化
# dims = [0, 1, 20, 21, 60, 61, 126, 127]  # 选择低、中、高维度
# plt.figure(figsize=(10, 5))
# for dim in dims:
#     plt.plot(pe[:, dim], label=f"Dim {dim}")
# plt.title("Positional Encoding Values Across Positions")
# plt.xlabel("Position (pos)")
# plt.ylabel("Value")
# plt.legend()
# plt.show()

# # 可视化 2：相邻位置的差异
# positions = [0, 1, 2, 3, 4]  # 相邻位置
# diffs = []
# for pos in range(len(positions)-1):
#     diff = pe[positions[pos+1]] - pe[positions[pos]]
#     diffs.append(diff)

# plt.figure(figsize=(10, 5))
# for i, diff in enumerate(diffs):
#     plt.plot(diff, label=f"Diff between Pos {positions[i]} and {positions[i+1]}")
# plt.title("Difference Between Adjacent Positions")
# plt.xlabel("Dimension (d_model)")
# plt.ylabel("Difference")
# plt.legend()
# plt.show()

# # 可视化 3：不同维度的差异分布
# diffs_all = np.abs(pe[1:] - pe[:-1])  # [49, 128]
# plt.figure(figsize=(15, 5))
# sns.heatmap(diffs_all, cmap='Blues', vmin=0, vmax=0.5)
# plt.title("Absolute Difference Between Adjacent Positions")
# plt.xlabel("Dimension (d_model)")
# plt.ylabel("Position Pair (pos to pos+1)")
# plt.show()

import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn as nn 

# class UniformPositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000, base_freq=0.1):
#         super().__init__()
#         assert d_model % 2 == 0, "d_model must be even for sin/cos pairs"
        
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        
#         # 每对维度 (sin, cos) 对应一个频率
#         num_pairs = d_model // 2
#         # 频率线性变化，从 base_freq 到 base_freq * 2
#         freqs = torch.linspace(base_freq, base_freq * 2, num_pairs)  # [d_model/2]
#         freqs = freqs.unsqueeze(0)  # [1, d_model/2]
        
#         # 计算角度
#         angles = position * freqs  # [max_len, d_model/2]
        
#         # 偶数维度用 sin，奇数维度用 cos
#         pe[:, 0::2] = torch.sin(angles)
#         pe[:, 1::2] = torch.cos(angles)
        
#         pe = pe.unsqueeze(0)  # [1, max_len, d_model]
#         self.register_buffer("pe", pe)
    
#     def forward(self, x):
#         seq_len = x.size(1)
#         x = x + self.pe[:, :seq_len, :]
#         return x

# # 参数
# d_model = 128
# max_len = 50
# base_freq = 0.1  # 基础频率

# # 初始化位置编码
# pe_module = UniformPositionalEncoding(d_model, max_len, base_freq)
# pe = pe_module.pe.squeeze(0).numpy()  # [max_len, d_model] -> [50, 128]

# # 可视化 1：位置编码热图
# plt.figure(figsize=(15, 5))
# sns.heatmap(pe, cmap='RdBu', vmin=-1, vmax=1)
# plt.title("Uniform Positional Encoding (d_model=128, max_len=50)")
# plt.xlabel("Dimension (d_model)")
# plt.ylabel("Position (pos)")
# plt.show()

# # 可视化 2：不同维度的值随位置变化
# dims = [0, 1, 20, 21, 60, 61, 126, 127]
# plt.figure(figsize=(10, 5))
# for dim in dims:
#     plt.plot(pe[:, dim], label=f"Dim {dim}")
# plt.title("Uniform Positional Encoding Values Across Positions")
# plt.xlabel("Position (pos)")
# plt.ylabel("Value")
# plt.legend()
# plt.show()

# # 可视化 3：相邻位置的差异
# positions = [0, 1, 2, 3, 4]
# diffs = []
# for pos in range(len(positions)-1):
#     diff = pe[positions[pos+1]] - pe[positions[pos]]
#     diffs.append(diff)

# plt.figure(figsize=(10, 5))
# for i, diff in enumerate(diffs):
#     plt.plot(diff, label=f"Diff between Pos {positions[i]} and {positions[i+1]}")
# plt.title("Difference Between Adjacent Positions (Uniform PE)")
# plt.xlabel("Dimension (d_model)")
# plt.ylabel("Difference")
# plt.legend()
# plt.show()

# # 可视化 4：不同维度的差异分布
# diffs_all = np.abs(pe[1:] - pe[:-1])  # [49, 128]
# plt.figure(figsize=(15, 5))
# sns.heatmap(diffs_all, cmap='Blues', vmin=0, vmax=0.5)
# plt.title("Absolute Difference Between Adjacent Positions (Uniform PE)")
# plt.xlabel("Dimension (d_model)")
# plt.ylabel("Position Pair (pos to pos+1)")
# plt.show()


# class EuclideanPositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000, scale=1.0):
#         super().__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        
#         # 每个维度对应一个方向，位置 pos 在该方向上移动
#         direction = torch.randn(d_model)  # 随机方向
#         direction = direction / torch.norm(direction)  # 归一化到单位向量
#         pe = position * direction * scale  # 每个位置沿方向移动 scale 距离
        
#         pe = pe.unsqueeze(0)  # [1, max_len, d_model]
#         self.register_buffer("pe", pe)
    
#     def forward(self, x):
#         seq_len = x.size(1)
#         x = x + self.pe[:, :seq_len, :]
#         return x

# # 参数
# d_model = 128
# max_len = 50
# scale = 1.0

# # 初始化位置编码
# pe_module = EuclideanPositionalEncoding(d_model, max_len, scale)
# pe = pe_module.pe.squeeze(0).numpy()  # [50, 128]

# # 验证欧氏距离
# dists = np.linalg.norm(pe[1:] - pe[:-1], axis=1)
# print("Euclidean distances between adjacent positions:", dists[:5])

# # 可视化：不同维度的值随位置变化
# dims = [0, 1, 20, 21, 60, 61, 126, 127]
# plt.figure(figsize=(10, 5))
# for dim in dims:
#     plt.plot(pe[:, dim], label=f"Dim {dim}")
# plt.title("Euclidean Positional Encoding Values Across Positions")
# plt.xlabel("Position (pos)")
# plt.ylabel("Value")
# plt.legend()
# plt.show()


import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"
        
        # 计算频率
        self.d_model = d_model
        theta = 10000 ** (-torch.arange(0, d_model, 2).float() / d_model)  # [d_model/2]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        angles = position * theta  # [max_len, d_model/2]
        
        # 预计算 sin 和 cos
        self.sin = torch.sin(angles)  # [max_len, d_model/2]
        self.cos = torch.cos(angles)  # [max_len, d_model/2]
        self.register_buffer("sin_cached", self.sin)
        self.register_buffer("cos_cached", self.cos)
    
    def rotate(self, x, pos):
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.view(batch_size, num_heads, seq_len, d_k // 2, 2)  # [batch_size, num_heads, seq_len, d_k/2, 2]
        
        # 提取 sin 和 cos
        sin = self.sin_cached[pos, :]  # [seq_len, d_k/2]
        cos = self.cos_cached[pos, :]  # [seq_len, d_k/2]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_k/2]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_k/2]
        
        # 旋转操作
        x0, x1 = x[..., 0], x[..., 1]  # [batch_size, num_heads, seq_len, d_k/2]
        x_rotated = torch.stack([
            x0 * cos - x1 * sin,
            x0 * sin + x1 * cos
        ], dim=-1)  # [batch_size, num_heads, seq_len, d_k/2, 2]
        
        return x_rotated.view(batch_size, num_heads, seq_len, d_k)

# 参数
d_model = 128
max_len = 50
num_heads = 8
d_k = d_model // num_heads  # 16

# 初始化 RoPE
rope = RotaryPositionalEncoding(d_k, max_len)

# 模拟 Q 向量（固定值以观察旋转效果）
Q = torch.ones(1, num_heads, max_len, d_k)  # [1, 8, 50, 16]
positions = torch.arange(max_len)
Q_rotated = rope.rotate(Q, positions).squeeze(0).numpy()  # [8, 50, 16]

# 可视化 1：不同维度的旋转值
head = 0  # 选择第一个头
dims = [0, 1, 6, 7, 14, 15]  # 选择几对维度
plt.figure(figsize=(10, 5))
for dim in dims:
    plt.plot(Q_rotated[head, :, dim], label=f"Dim {dim}")
plt.title(f"RoPE Rotated Q Values (Head {head})")
plt.xlabel("Position (pos)")
plt.ylabel("Value")
plt.legend()
plt.show()

# 可视化 2：相邻位置的差异
diffs = np.abs(Q_rotated[:, 1:] - Q_rotated[:, :-1])  # [8, 49, 16]
diffs = diffs.mean(axis=0)  # [49, 16]，平均所有头
plt.figure(figsize=(10, 5))
sns.heatmap(diffs, cmap='Blues', vmin=0, vmax=0.5)
plt.title("Absolute Difference Between Adjacent Positions (RoPE)")
plt.xlabel("Dimension (d_k)")
plt.ylabel("Position Pair (pos to pos+1)")
plt.show()

# 可视化 3：不同维度的角度变化
angles = rope.sin_cached**2 + rope.cos_cached**2  # 验证 sin^2 + cos^2 = 1
plt.figure(figsize=(10, 5))
for i in [0, 3, 7]:
    plt.plot(rope.sin_cached[:, i].numpy(), label=f"sin (dim {2*i})")
    plt.plot(rope.cos_cached[:, i].numpy(), label=f"cos (dim {2*i+1})")
plt.title("RoPE Angles (sin and cos)")
plt.xlabel("Position (pos)")
plt.ylabel("Value")
plt.legend()
plt.show()