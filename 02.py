import torch
import torch.nn as nn
import math


# =================================
# 1. Scaled Dot-Product Attention
# =================================

class ScaledDotProductAttention(nn.Module):
    """
    实现缩放点积注意力机制
    公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    参数:
        d_k: 键向量的维度（用于缩放因子）
        dropout: Dropout概率

    输入:
        Q: 查询向量 [batch_size, seq_len, d_model]
        K: 键向量 [batch_size, seq_len, d_model]
        V: 值向量 [batch_size, seq_len, d_model]
        mask: 可选掩码 [batch_size, seq_len, seq_len]

    输出:
        attention_output: 注意力输出 [batch_size, seq_len, d_model]
        attention_weights: 注意力权重 [batch_size, seq_len, seq_len]
    """

    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # 计算Q和K的点积
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, seq_len, seq_len]

        # 缩放点积
        scores = scores / math.sqrt(self.d_k)

        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attention_weights = nn.Softmax(dim=-1)(scores)
        attention_weights = self.dropout(attention_weights)

        # 计算加权值向量
        output = torch.matmul(attention_weights, V)  # [batch_size, seq_len, d_model]

        return output, attention_weights


# =================================
# 2. Multi-Head Attention
# =================================

class MultiHeadAttention(nn.Module):
    """
    实现多头注意力机制
    将输入拆分为多个头，每个头独立计算注意力，最后拼接结果

    参数:
        d_model: 模型维度
        num_heads: 注意力头数量
        dropout: Dropout概率

    输入:
        Q: 查询向量 [batch_size, seq_len, d_model]
        K: 键向量 [batch_size, seq_len, d_model]
        V: 值向量 [batch_size, seq_len, d_model]
        mask: 可选掩码 [batch_size, seq_len, seq_len]

    输出:
        output: 多头注意力输出 [batch_size, seq_len, d_model]
        attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换层：将输入投影到Q、K、V空间
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # 缩放点积注意力层
        self.attention = ScaledDotProductAttention(self.d_k, dropout)

        # 输出线性层
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        将输入张量拆分为多个头
        输入: [batch_size, seq_len, d_model]
        输出: [batch_size, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        将多个头的输出拼接回原始形状
        输入: [batch_size, num_heads, seq_len, d_k]
        输出: [batch_size, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # 线性投影
        Q = self.W_Q(Q)  # [batch_size, seq_len, d_model]
        K = self.W_K(K)  # [batch_size, seq_len, d_model]
        V = self.W_V(V)  # [batch_size, seq_len, d_model]

        # 拆分为多个头
        Q_heads = self.split_heads(Q)  # [batch_size, num_heads, seq_len, d_k]
        K_heads = self.split_heads(K)
        V_heads = self.split_heads(V)

        # 如果提供掩码，需要扩展维度以匹配多头
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

        # 计算每个头的注意力
        attn_output, attn_weights = self.attention(
            Q_heads, K_heads, V_heads, mask
        )  # [batch_size, num_heads, seq_len, d_k], [batch_size, num_heads, seq_len, seq_len]

        # 拼接所有头的输出
        combined = self.combine_heads(attn_output)  # [batch_size, seq_len, d_model]

        # 输出线性变换
        output = self.W_O(combined)  # [batch_size, seq_len, d_model]
        output = self.dropout(output)

        return output, attn_weights


# =================================
# 3. Add & Norm层
# =================================

class AddNorm(nn.Module):
    """
    实现残差连接和层归一化
    公式: LayerNorm(x + Sublayer(x))

    参数:
        d_model: 模型维度
        dropout: Dropout概率

    输入:
        x: 原始输入 [batch_size, seq_len, d_model]
        sublayer_output: 子层输出 [batch_size, seq_len, d_model]

    输出:
        output: 归一化后的输出 [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model, dropout=0.1):
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        # 残差连接 + Dropout
        output = x + self.dropout(sublayer_output)
        # 层归一化
        output = self.layer_norm(output)
        return output


# =================================
# 4. 前馈神经网络
# =================================

class FeedForward(nn.Module):
    """
    实现位置式前馈神经网络
    公式: FFN(x) = max(0, xW1 + b1)W2 + b2

    参数:
        d_model: 模型维度
        d_ff: 隐藏层维度（通常为4*d_model）
        dropout: Dropout概率

    输入:
        x: 输入张量 [batch_size, seq_len, d_model]

    输出:
        output: 前馈网络输出 [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 使用GELU激活函数

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# =================================
# 5. Transformer Encoder层
# =================================

class EncoderLayer(nn.Module):
    """
    实现Transformer Encoder的单层结构
    包含:
        1. 多头自注意力机制
        2. Add & Norm层
        3. 前馈神经网络
        4. Add & Norm层

    参数:
        d_model: 模型维度
        num_heads: 注意力头数量
        d_ff: 前馈网络隐藏层维度
        dropout: Dropout概率

    输入:
        x: 输入张量 [batch_size, seq_len, d_model]
        mask: 可选掩码 [batch_size, seq_len, seq_len]

    输出:
        output: 编码器层输出 [batch_size, seq_len, d_model]
        attention_weights: 注意力权重 [batch_size, num_heads, seq_len, seq_len]
    """

    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super(EncoderLayer, self).__init__()
        # 多头自注意力
        self.multi_head_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 第一个Add & Norm
        self.add_norm1 = AddNorm(d_model, dropout)
        # 前馈神经网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        # 第二个Add & Norm
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x, mask=None):
        # 多头自注意力
        attn_output, attn_weights = self.multi_head_attn(x, x, x, mask)
        # 第一个Add & Norm
        x = self.add_norm1(x, attn_output)
        # 前馈神经网络
        ff_output = self.feed_forward(x)
        # 第二个Add & Norm
        output = self.add_norm2(x, ff_output)

        return output, attn_weights


# =================================
# 6. Transformer Encoder模型
# =================================

class TransformerEncoder(nn.Module):
    """
    完整的Transformer Encoder模型
    包含多个Encoder层堆叠

    参数:
        num_layers: Encoder层数量
        d_model: 模型维度
        num_heads: 注意力头数量
        d_ff: 前馈网络隐藏层维度
        dropout: Dropout概率

    输入:
        x: 输入张量 [batch_size, seq_len, d_model]
        mask: 可选掩码 [batch_size, seq_len, seq_len]

    输出:
        output: 编码器输出 [batch_size, seq_len, d_model]
        all_attention_weights: 所有层的注意力权重列表
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        all_attention_weights = []

        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attention_weights.append(attn_weights)

        return x, all_attention_weights


# =================================
# 7. 测试Transformer Encoder
# =================================

def test_transformer_encoder():
    """
    测试Transformer Encoder模型
    创建随机输入并查看输出
    """
    # 模型参数
    num_layers = 6
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.1

    # 创建模型
    encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)

    # 创建随机输入 (batch_size=4, seq_len=10, d_model=512)
    x = torch.randn(4, 10, 512)

    # 前向传播
    output, all_attn_weights = encoder(x)

    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    print("注意力权重数量:", len(all_attn_weights))
    print("第一层注意力权重形状:", all_attn_weights[0].shape)

    return output, all_attn_weights


# 运行测试
if __name__ == "__main__":
    test_transformer_encoder()