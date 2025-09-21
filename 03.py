import torch
import torch.nn as nn
import math
from einops import rearrange, repeat


class PatchEmbedding(nn.Module):
    """
    图像分块嵌入层
    将输入图像分割为固定大小的块，并将每个块线性投影到指定维度

    参数:
        img_size: 输入图像尺寸 (H, W)
        patch_size: 块大小 (P, P)
        in_channels: 输入通道数 (默认为3)
        embed_dim: 嵌入维度 (D)

    输入:
        x: 输入图像张量 [batch_size, in_channels, H, W]

    输出:
        x: 块嵌入序列 [batch_size, num_patches, embed_dim]
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])

        # 投影层：将每个块展平并投影到嵌入维度
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

    def forward(self, x):
        # 输入: [batch_size, in_channels, H, W]
        x = self.projection(x)  # [batch_size, embed_dim, H/P, W/P]
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        return x


class ClassToken(nn.Module):
    """
    类别标记 (Class Token)
    在输入序列前添加一个可学习的分类标记

    参数:
        embed_dim: 嵌入维度 (D)

    输入:
        x: 输入序列 [batch_size, num_patches, embed_dim]

    输出:
        x: 添加类别标记后的序列 [batch_size, num_patches+1, embed_dim]
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        # 扩展类别标记以匹配批次大小
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        # 在序列开头添加类别标记
        x = torch.cat([cls_tokens, x], dim=1)
        return x


class PositionalEmbedding(nn.Module):
    """
    位置嵌入层
    为每个位置添加可学习的位置嵌入

    参数:
        num_patches: 块数量 (不包括类别标记)
        embed_dim: 嵌入维度 (D)

    输入:
        x: 输入序列 [batch_size, num_patches+1, embed_dim]

    输出:
        x: 添加位置嵌入后的序列 [batch_size, num_patches+1, embed_dim]
    """

    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

    def forward(self, x):
        # 添加位置嵌入
        x = x + self.pos_embedding
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 完整模型

    参数:
        img_size: 输入图像尺寸
        patch_size: 块大小
        in_channels: 输入通道数
        num_classes: 分类类别数
        embed_dim: 嵌入维度 (D)
        depth: Transformer编码器层数
        num_heads: 多头注意力头数
        mlp_ratio: MLP隐藏层维度与嵌入维度的比例
        qkv_bias: 是否在QKV投影中使用偏置
        dropout: Dropout概率
        attention_dropout: 注意力Dropout概率

    输入:
        x: 输入图像 [batch_size, in_channels, img_size, img_size]

    输出:
        logits: 分类logits [batch_size, num_classes]
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, dropout=0.1, attention_dropout=0.0):
        super().__init__()

        # 1. 图像分块嵌入
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # 2. 添加类别标记
        self.cls_token = ClassToken(embed_dim)

        # 3. 位置嵌入
        self.pos_embed = PositionalEmbedding(num_patches, embed_dim)

        # 4. Dropout层
        self.dropout = nn.Dropout(dropout)

        # 5. Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout=dropout,
                attention_dropout=attention_dropout
            ) for _ in range(depth)
        ])

        # 6. 层归一化
        self.norm = nn.LayerNorm(embed_dim)

        # 7. 分类头
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        # 图像分块嵌入
        x = self.patch_embed(x)  # [batch_size, num_patches, embed_dim]

        # 添加类别标记
        x = self.cls_token(x)  # [batch_size, num_patches+1, embed_dim]

        # 添加位置嵌入
        x = self.pos_embed(x)  # [batch_size, num_patches+1, embed_dim]

        # 应用Dropout
        x = self.dropout(x)

        # 通过Transformer编码器
        for block in self.blocks:
            x = block(x)

        # 层归一化
        x = self.norm(x)

        # 提取类别标记 (序列的第一个位置)
        cls_token = x[:, 0]

        # 分类头
        logits = self.head(cls_token)

        return logits


# =================================
# Transformer编码器层 (复用之前实现)
# =================================

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力机制"""

    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = nn.Softmax(dim=-1)(scores)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.W_O = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(self.head_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 线性投影
        Q = self.W_Q(x)  # [batch_size, seq_len, embed_dim]
        K = self.W_K(x)
        V = self.W_V(x)

        # 拆分多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        attn_output, attn_weights = self.attention(Q, K, V)

        # 拼接多头输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # 输出投影
        output = self.W_O(attn_output)
        output = self.dropout(output)
        return output


class FeedForward(nn.Module):
    """前馈神经网络"""

    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0,
                 qkv_bias=True, dropout=0.1, attention_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attention_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, mlp_ratio, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 残差连接 + 层归一化 + 多头注意力
        attn_output = self.attn(self.norm1(x))
        x = x + attn_output

        # 残差连接 + 层归一化 + 前馈网络
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        return x


# =================================
# ViT模型测试
# =================================

def test_vit_model():
    """
    测试ViT模型
    创建随机输入并查看输出
    """
    # 模型参数 (ViT-Base配置)
    img_size = 224
    patch_size = 16
    in_channels = 3
    num_classes = 1000
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_ratio = 4.0

    # 创建ViT模型
    vit = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio
    )

    # 创建随机输入 (batch_size=4, channels=3, H=224, W=224)
    x = torch.randn(4, 3, img_size, img_size)

    # 前向传播
    logits = vit(x)

    print("输入形状:", x.shape)
    print("输出形状 (logits):", logits.shape)

    return logits


if __name__ == "__main__":
    test_vit_model()