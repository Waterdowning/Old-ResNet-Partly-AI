import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time


# ======================
# 基础残差块：BasicBlock（适配ResNet-18/34等浅层网络）
# 对应原论文Section 4.1："Identity Mappings in Deep Residual Networks"
# ======================
class BasicBlock(nn.Module):
    """ResNet基础残差块（用于≤34层的浅层网络，仅包含两个3x3卷积）"""
    expansion = 1  # 扩展系数：残差连接时需将shortcut输出的通道数扩展至out_channels*expansion（此处为1倍，即无需额外扩展）

    def __init__(self, in_channels, out_channels, stride=1):
        """
        初始化基础残差块
        :param in_channels: 输入特征图的通道数（如前一层的输出通道）
        :param out_channels: 当前块的「基准输出通道数」（未乘expansion）
        :param stride: 第一个卷积层的步长（核心作用：①下采样特征图尺寸（stride=2时宽高减半）；②调整通道数（若需））
        """
        super(BasicBlock, self).__init__()

        # ---------------------- 主路径（Main Path）：两个卷积层 + BN + ReLU ----------------------
        # 第一个卷积层：3x3卷积，实现「特征提取」或「下采样」
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,  # 输入通道数
            out_channels=out_channels,  # 输出通道数（基准通道）
            kernel_size=3,  # 卷积核大小（3x3是小卷积核，平衡感受野与计算量）
            stride=stride,  # 步长（控制下采样/通道调整）
            padding=1,  # 填充1圈，保持特征图尺寸（若stride=1）或减半（若stride=2）
            bias=False  # 不使用偏置：因后续接BN层，BN会学习偏置，避免冗余
        )
        self.bn1 = nn.BatchNorm2d(out_channels)  # 批归一化：加速训练（减少「内部协变量偏移」），稳定梯度

        # 第二个卷积层：3x3卷积，仅提取特征（保持尺寸不变）
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,  # 输入通道数=上一层的输出通道数
            out_channels=out_channels,  # 输出通道数=基准通道数
            kernel_size=3,  # 3x3卷积核
            stride=1,  # 步长=1：保持特征图尺寸不变
            padding=1,  # 填充1圈，保持尺寸
            bias=False  # 无偏置（BN替代）
        )
        self.bn2 = nn.BatchNorm2d(out_channels)  # 第二个BN层

        # ---------------------- 快捷连接（Shortcut Connection）：解决深层网络退化问题 ----------------------
        # 核心原则：保证「主路径输出」与「shortcut输出」的通道数、空间尺寸完全一致，才能直接相加
        self.shortcut = nn.Sequential()  # 默认是恒等映射（Identity Mapping）
        # 若需调整通道数 或 特征图尺寸，则用1x1卷积「投影」shortcut输出，使其匹配主路径
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,  # 输入通道数=原输入
                    out_channels=self.expansion * out_channels,  # 输出通道数=主路径最终通道数（out_channels*expansion）
                    kernel_size=1,  # 1x1卷积：仅调整通道/尺寸，无空间特征提取
                    stride=stride,  # 步长=主路径第一个卷积的stride（同步下采样）
                    bias=False  # 无偏置（BN替代）
                ),
                nn.BatchNorm2d(self.expansion * out_channels)  # 投影后的BN层
            )

    def forward(self, x):
        """前向传播：主路径输出 + shortcut输出 → 残差连接"""
        # 主路径计算：Conv1 → BN1 → ReLU → Conv2 → BN2
        out = nn.ReLU()(self.bn1(self.conv1(x)))  # 第一步：3x3卷积 + BN + ReLU
        out = self.bn2(self.conv2(out))  # 第二步：3x3卷积 + BN（无ReLU，留待残差相加后再激活）

        # 残差连接：主路径输出 + shortcut输出（维度完全一致）
        out += self.shortcut(x)

        # 最终激活：ReLU（原论文强调「残差相加后再做ReLU」，而非每个卷积后都做）
        out = nn.ReLU()(out)
        return out


# ======================
# ResNet-18完整模型（适配CIFAR-10的32x32小图像）
# 对应原论文Table 1："ResNet-18"配置（2+2+2+2个BasicBlock per stage）
# ======================
class ResNet18(nn.Module):
    """ResNet-18模型（专为CIFAR-10优化，初始卷积层不进行下采样）"""

    def __init__(self, block, num_blocks, num_classes=10):
        """
        初始化ResNet-18
        :param block: 基础残差块类型（此处为BasicBlock）
        :param num_blocks: 各阶段的残差块数量（ResNet-18固定为[2,2,2,2]，即每个阶段2个块）
        :param num_classes: 分类类别数（CIFAR-10为10类）
        """
        super(ResNet18, self).__init__()
        self.in_channels = 64  # 初始化输入通道数（对应第一个卷积层的输出通道）

        # ---------------------- 初始卷积层（Stem Layer）：适配CIFAR-10的小图像 ----------------------
        # 原论文针对ImageNet用7x7卷积+stride=2下采样，但CIFAR-10是32x32小图像，故调整为：
        # 3x3卷积+stride=1+padding=1 → 保持32x32尺寸，避免过早丢失细节
        self.conv1 = nn.Conv2d(
            in_channels=3,  # 输入通道：CIFAR-10是RGB图像→3通道
            out_channels=64,  # 输出通道：64
            kernel_size=3,  # 3x3卷积核
            stride=1,  # 步长=1：保持尺寸
            padding=1,  # 填充1圈
            bias=False  # 无偏置（BN替代）
        )
        self.bn1 = nn.BatchNorm2d(64)  # 初始BN层

        # ---------------------- 四个残差阶段（Stage 1~4）：逐步下采样+增加通道数 ----------------------
        # 每个阶段通过_make_layer构建，包含num_blocks个残差块
        # Stage 1：输入32x32x64 → 输出32x32x64（无下采样，通道不变）
        self.layer1 = self._make_layer(block, out_channels=64, num_blocks=num_blocks[0], stride=1)
        # Stage 2：输入32x32x64 → 输出16x16x128（stride=2下采样，通道翻倍）
        self.layer2 = self._make_layer(block, out_channels=128, num_blocks=num_blocks[1], stride=2)
        # Stage 3：输入16x16x128 → 输出8x8x256（stride=2下采样，通道翻倍）
        self.layer3 = self._make_layer(block, out_channels=256, num_blocks=num_blocks[2], stride=2)
        # Stage 4：输入8x8x256 → 输出4x4x512（stride=2下采样，通道翻倍）
        self.layer4 = self._make_layer(block, out_channels=512, num_blocks=num_blocks[3], stride=2)

        # ---------------------- 分类头（Classifier Head）：从特征图到类别概率 ----------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化：将任意尺寸的特征图压缩到1x1（此处4x4→1x1）
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层：将512*expansion维特征映射到类别数

        # ---------------------- 权重初始化：遵循原论文的Kaiming初始化 ----------------------
        self._init_weights()

    def _init_weights(self):
        """统一初始化模型权重（Kaiming初始化+BN层初始化）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层：Kaiming正态初始化（适配ReLU激活函数，避免梯度消失/爆炸）
                # mode='fan_out'：按输出神经元数量初始化；nonlinearity='relu'：针对ReLU优化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # BN层：weight初始化为1（保持输入分布），bias初始化为0（后续学习调整）
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        构建一个「残差阶段」（包含num_blocks个相同类型的残差块）
        :param block: 残差块类型（如BasicBlock）
        :param out_channels: 该阶段的「基准输出通道数」（未乘expansion）
        :param num_blocks: 该阶段的残差块数量
        :param stride: 第一个残差块的步长（控制下采样/通道调整）
        :return: 包含num_blocks个块的Sequential模块
        """
        # 设置每个块的步长：第一个块用给定的stride（下采样/调整通道），后续块用stride=1（保持尺寸）
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []  # 存储当前阶段的所有残差块

        for s in strides:
            # 创建残差块：输入通道是当前self.in_channels，输出通道是out_channels
            layers.append(block(self.in_channels, out_channels, s))
            # 更新下一块的输入通道数：当前块的输出通道是out_channels*expansion（因shortcut可能扩展）
            self.in_channels = out_channels * block.expansion

        # 将块列表转为Sequential模块（按顺序执行）
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播：从输入图像到分类logits"""
        # Stage 0（初始卷积）：3x3卷积 → BN → ReLU → 32x32x64
        x = nn.ReLU()(self.bn1(self.conv1(x)))

        # Stage 1~4：依次通过四个残差阶段 → 特征图逐步缩小（32→16→8→4），通道逐步增加（64→128→256→512）
        x = self.layer1(x)  # 输出：32x32x64
        x = self.layer2(x)  # 输出：16x16x128
        x = self.layer3(x)  # 输出：8x8x256
        x = self.layer4(x)  # 输出：4x4x512

        # 分类头：全局平均池化 → 展平 → 全连接
        x = self.avgpool(x)  # 输出：1x1x512
        x = torch.flatten(x, 1)  # 展平为512维向量（去掉batch维度）
        x = self.fc(x)  # 输出：num_classes维logits（未归一化的类别分数）
        return x


# ======================
# 瓶颈残差块：Bottleneck（适配ResNet-50/101/152等深层网络）
# 对应原论文Section 4.2："Identity Mappings in Deep Residual Networks"（解决深层网络计算量过大问题）
# ======================
class Bottleneck(nn.Module):
    """ResNet瓶颈块（用于≥50层的深层网络，用1x1卷积降维→3x3卷积→1x1卷积升维，减少计算量）"""
    expansion = 4  # 扩展系数：瓶颈块的最终输出通道数是基准通道数的4倍（如out_channels=64 → 最终输出256）

    def __init__(self, in_channels, out_channels, stride=1):
        """
        初始化瓶颈块
        :param in_channels: 输入通道数
        :param out_channels: 瓶颈块的「基准通道数」（未乘expansion）
        :param stride: 第一个卷积层的步长（控制下采样/通道调整）
        """
        super(Bottleneck, self).__init__()

        # ---------------------- 主路径（Main Path）：三步卷积（降维→特征提取→升维） ----------------------
        # 1. 1x1卷积：降维（将输入通道数从in_channels压缩到out_channels，减少计算量）
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,  # 1x1卷积：无空间特征提取，仅调整通道
            stride=1,  # 步长=1：不改变尺寸
            bias=False  # 无偏置（BN替代）
        )
        self.bn1 = nn.BatchNorm2d(out_channels)  # 降维后的BN层

        # 2. 3x3卷积：核心特征提取（保持尺寸不变，提取局部特征）
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,  # 步长=主路径的stride（下采样/调整尺寸）
            padding=1,  # 填充1圈，保持尺寸
            bias=False  # 无偏置（BN替代）
        )
        self.bn2 = nn.BatchNorm2d(out_channels)  # 特征提取后的BN层

        # 3. 1x1卷积：升维（将通道数从out_channels扩展到out_channels*expansion，恢复计算量）
        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels * self.expansion,
            kernel_size=1,  # 1x1卷积：仅调整通道
            stride=1,  # 步长=1：不改变尺寸
            bias=False  # 无偏置（BN替代）
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)  # 升维后的BN层

        # ---------------------- 快捷连接（Shortcut Connection）：匹配维度 ----------------------
        self.shortcut = nn.Sequential()
        # 若需调整通道数 或 特征图尺寸，则用1x1卷积投影shortcut输出，使其匹配主路径的最终输出（out_channels*expansion）
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,  # 同步主路径的下采样
                    bias=False
                ),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        """前向传播：三步卷积 → 残差连接 → 激活"""
        # 主路径计算：1x1降维 → BN → ReLU → 3x3特征提取 → BN → ReLU → 1x1升维 → BN
        out = nn.ReLU()(self.bn1(self.conv1(x)))  # 1x1降维 + BN + ReLU
        out = nn.ReLU()(self.bn2(self.conv2(out)))  # 3x3特征提取 + BN + ReLU
        out = self.bn3(self.conv3(out))  # 1x1升维 + BN（无ReLU）

        # 残差连接：主路径输出 + shortcut输出（维度完全一致）
        out += self.shortcut(x)

        # 最终激活：ReLU
        out = nn.ReLU()(out)
        return out


# ======================
# 可扩展ResNet基类（支持BasicBlock/Bottleneck，适配不同层数）
# 对应原论文Table 1：ResNet-50/101/152的配置（3+4+6+3、3+4+23+3、3+8+36+3个Bottleneck per stage）
# ======================
class ResNet(nn.Module):
    """可扩展的ResNet基类（通过block和layers参数生成不同版本的ResNet）"""

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
        """
        初始化可扩展ResNet
        :param block: 残差块类型（BasicBlock/Bottleneck）
        :param layers: 各阶段的残差块数量（如ResNet-50为[3,4,6,3]）
        :param num_classes: 分类类别数
        :param zero_init_residual: 是否将Bottleneck的最后一个BN层初始化为0（原论文技巧，加速收敛）
        """
        super(ResNet, self).__init__()
        self.in_channels = 64  # 初始化输入通道数

        # ---------------------- 初始卷积层（Stem Layer）：同ResNet18 ----------------------
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ---------------------- 四个残差阶段：通过_make_layer构建 ----------------------
        self.layer1 = self._make_layer(block, out_channels=64, blocks=layers[0])
        self.layer2 = self._make_layer(block, out_channels=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, out_channels=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, out_channels=512, blocks=layers[3], stride=2)

        # ---------------------- 分类头：同ResNet18 ----------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # ---------------------- 权重初始化：包含Bottleneck的特殊初始化 ----------------------
        self._init_weights(zero_init_residual)

    def _init_weights(self, zero_init_residual):
        """统一初始化权重（含Bottleneck的zero_init_residual技巧）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 原论文技巧：将Bottleneck的最后一个BN层（bn3）初始化为0，加速收敛
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        构建残差阶段（通用逻辑，适配BasicBlock/Bottleneck）
        :param block: 残差块类型
        :param out_channels: 阶段基准通道数
        :param blocks: 块数量
        :param stride: 第一个块的步长
        :return: Sequential模块
        """
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播：通用逻辑（同ResNet18）"""
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ----------------------
# ResNet模型工厂函数（快速创建不同版本的ResNet）
# 对应原论文Table 1的配置：
# - ResNet-18: 2+2+2+2个BasicBlock → [2,2,2,2]
# - ResNet-34: 3+4+6+3个BasicBlock → [3,4,6,3]
# - ResNet-50: 3+4+6+3个Bottleneck → [3,4,6,3]
# - ResNet-101:3+4+23+3个Bottleneck → [3,4,23,3]
# - ResNet-152:3+8+36+3个Bottleneck → [3,8,36,3]
# ======================
def resnet18(num_classes=10):
    """创建ResNet-18（BasicBlock + [2,2,2,2]）"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=10):
    """创建ResNet-34（BasicBlock + [3,4,6,3]）"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes=10):
    """创建ResNet-50（Bottleneck + [3,4,6,3]）"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes=10):
    """创建ResNet-101（Bottleneck + [3,4,23,3]）"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes=10):
    """创建ResNet-152（Bottleneck + [3,8,36,3]）"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


# ======================
# 数据加载与预处理（CIFAR-10）
# 遵循原论文的「训练集增强+测试集无增强」策略
# ======================
def get_cifar10_dataloaders(batch_size=128):
    """
    加载CIFAR-10数据集，返回训练/测试DataLoader
    :param batch_size: 批次大小（默认128）
    :return: (trainloader, testloader)
    """
    # ---------------------- 训练集数据增强（防止过拟合） ----------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪（带4像素填充）：增加数据多样性，防止过拟合
        transforms.RandomHorizontalFlip(),  # 随机水平翻转：模拟镜像变换，增加泛化性
        transforms.ToTensor(),  # 转为Tensor（像素值从0~255→0~1）
        transforms.Normalize(  # 标准化：按CIFAR-10官方均值/方差调整，加速收敛
            mean=(0.4914, 0.4822, 0.4465),  # CIFAR-10训练集的RGB均值
            std=(0.2023, 0.1994, 0.2010)  # CIFAR-10训练集的RGB标准差
        )
    ])

    # ---------------------- 测试集预处理（无增强，保持原始分布） ----------------------
    transform_test = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize(  # 标准化（同训练集）
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        )
    ])

    # ---------------------- 加载数据集 ----------------------
    trainset = torchvision.datasets.CIFAR10(
        root='./data',  # 数据存储路径（若不存在会自动创建）
        train=True,  # 加载训练集
        download=True,  # 若未下载则自动下载
        transform=transform_train  # 应用训练集增强
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,  # 训练集
        batch_size=batch_size,  # 批次大小
        shuffle=True,  # 训练集打乱顺序（防止模型记住顺序）
        num_workers=2  # 并行加载数据的进程数（加速数据读取）
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,  # 加载测试集
        download=True,
        transform=transform_test  # 应用测试集预处理
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,  # 测试集无需打乱
        num_workers=2
    )

    return trainloader, testloader


# ======================
# 训练函数：模型训练与指标记录
# 遵循原论文的训练策略：SGD+Momentum+Weight Decay+Multi-Step LR Decay
# ======================
def train_model(model, trainloader, testloader, epochs=100, lr=0.1, title="ResNet-18训练"):
    """
    训练模型并返回训练/测试指标
    :param model: 待训练的模型
    :param trainloader: 训练数据加载器
    :param testloader: 测试数据加载器
    :param epochs: 训练轮次（默认100）
    :param lr: 初始学习率（默认0.1，原论文ResNet-18的初始LR）
    :param title: 训练标题（用于保存模型/图表）
    :return: 包含训练/测试指标的字典
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择GPU/CPU
    model = model.to(device)  # 将模型移至选定的设备

    # ---------------------- 损失函数：交叉熵损失（分类任务标准损失） ----------------------
    criterion = nn.CrossEntropyLoss()  # 包含Softmax+Log+CrossEntropy，优化类别概率

    # ---------------------- 优化器：SGD+Momentum+Weight Decay ----------------------
    # SGD：随机梯度下降，Momentum=0.9加速收敛（减少震荡），Weight Decay=5e-4（L2正则，防止过拟合）
    optimizer = optim.SGD(
        model.parameters(),  # 优化模型所有参数
        lr=lr,  # 初始学习率
        momentum=0.9,  # 动量项
        weight_decay=5e-4  # L2正则系数
    )

    # ---------------------- 学习率调度器：Multi-Step LR Decay ----------------------
    # 在第60/80轮将学习率衰减至原来的1/10（原论文ResNet-18的调度策略）
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,  # 关联的优化器
        milestones=[60, 80],  # 衰减的学习轮次
        gamma=0.1  # 衰减因子（LR *= gamma）
    )

    # ---------------------- 记录训练指标 ----------------------
    train_losses = []  # 每轮训练的平均损失
    test_losses = []  # 每轮测试的平均损失
    train_accs = []  # 每轮训练的准确率（%）
    test_accs = []  # 每轮测试的准确率（%）

    print(f"开始训练：{title}")
    start_time = time.time()  # 记录训练开始时间

    for epoch in range(epochs):
        model.train()  # 开启训练模式（启用Dropout/BatchNorm的更新）
        running_loss = 0.0  # 当前轮次的总损失
        correct = 0  # 当前轮次预测正确的样本数
        total = 0  # 当前轮次的总样本数

        # ---------------------- 训练循环：遍历所有训练数据 ----------------------
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)  # 数据移至设备

            optimizer.zero_grad()  # 清空梯度（避免梯度累积）
            outputs = model(inputs)  # 前向传播：输入→logits
            loss = criterion(outputs, labels)  # 计算损失：logits→损失值
            loss.backward()  # 反向传播：计算梯度（从损失到参数）
            optimizer.step()  # 更新参数：根据梯度调整参数

            # ---------------------- 统计训练指标 ----------------------
            running_loss += loss.item()  # 累加当前批次的损失
            _, predicted = outputs.max(1)  # 获取预测类别（logits最大的索引）
            total += labels.size(0)  # 累加当前批次的样本数
            correct += predicted.eq(labels).sum().item()  # 累加预测正确的样本数

        # ---------------------- 计算当前轮次的训练指标 ----------------------
        train_loss = running_loss / len(trainloader)  # 平均训练损失（总损失/批次数量）
        train_acc = 100. * correct / total  # 训练准确率（正确样本数/总样本数*100）
        train_losses.append(train_loss)  # 记录训练损失
        train_accs.append(train_acc)  # 记录训练准确率

        # ---------------------- 评估测试集：计算泛化能力 ----------------------
        test_loss, test_acc = evaluate_model(model, testloader, criterion, device)
        test_losses.append(test_loss)  # 记录测试损失
        test_accs.append(test_acc)  # 记录测试准确率

        # ---------------------- 更新学习率 ----------------------
        scheduler.step()  # 调用调度器，衰减学习率

        # ---------------------- 打印当前轮次进度 ----------------------
        print(f"Epoch [{epoch + 1}/{epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    # ---------------------- 训练结束：统计与保存 ----------------------
    total_time = time.time() - start_time
    print(f"训练完成! 总耗时: {total_time // 60:.0f}分{total_time % 60:.0f}秒")
    print(f"最佳测试准确率: {max(test_accs):.2f}%")

    # 保存模型：仅保存参数（state_dict），节省空间
    model_path = f"{title.replace(' ', '_')}_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存至: {model_path}")

    # 返回指标字典，用于后续可视化
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs
    }


# ----------------------
# 评估函数：计算模型在测试集上的性能（无梯度计算，节省内存）
# ======================
def evaluate_model(model, testloader, criterion, device):
    """
    评估模型在测试集上的性能
    :param model: 待评估的模型
    :param testloader: 测试数据加载器
    :param criterion: 损失函数
    :param device: 设备（CPU/GPU）
    :return: (测试损失, 测试准确率%)
    """
    model.eval()  # 开启评估模式（关闭Dropout/BatchNorm的更新，使用训练好的统计量）
    test_loss = 0.0  # 测试总损失
    correct = 0  # 测试正确样本数
    total = 0  # 测试总样本数

    with torch.no_grad():  # 上下文管理器：不计算梯度（节省内存，加速计算）
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # 前向传播：输入→logits
            loss = criterion(outputs, labels)  # 计算损失

            # 统计指标
            test_loss += loss.item()  # 累加损失
            _, predicted = outputs.max(1)  # 获取预测类别
            total += labels.size(0)  # 累加样本数
            correct += predicted.eq(labels).sum().item()  # 累加正确样本数

    # 计算平均测试损失和准确率
    test_loss /= len(testloader)  # 平均测试损失
    test_acc = 100. * correct / total  # 测试准确率（%）
    return test_loss, test_acc


# ======================
# 可视化函数：绘制训练/测试的损失与准确率曲线
# ======================
def plot_metrics(metrics, title="ResNet-18训练指标"):
    """
    绘制训练/测试的损失与准确率曲线，并保存为高清图片
    :param metrics: 包含训练/测试指标的字典（来自train_model的返回值）
    :param title: 图表标题（默认"ResNet-18训练指标"）
    """
    plt.figure(figsize=(12, 10))  # 设置图表尺寸

    # ---------------------- 损失曲线：训练损失 vs 测试损失 ----------------------
    plt.subplot(2, 1, 1)  # 2行1列的第1个子图
    plt.plot(metrics['train_losses'], label='训练损失', linewidth=1.5)  # 训练损失曲线
    plt.plot(metrics['test_losses'], label='测试损失', linewidth=1.5)  # 测试损失曲线
    plt.xlabel('训练轮次')  # X轴标签
    plt.ylabel('损失')  # Y轴标签
    plt.title(f'{title} - 损失曲线')  # 子图标题
    plt.legend()  # 显示图例
    plt.grid(True, alpha=0.3)  # 显示网格（透明度0.3，不刺眼）

    # ---------------------- 准确率曲线：训练准确率 vs 测试准确率 ----------------------
    plt.subplot(2, 1, 2)  # 2行1列的第2个子图
    plt.plot(metrics['train_accs'], label='训练准确率', linewidth=1.5)  # 训练准确率曲线
    plt.plot(metrics['test_accs'], label='测试准确率', linewidth=1.5)  # 测试准确率曲线
    plt.xlabel('训练轮次')  # X轴标签
    plt.ylabel('准确率 (%)')  # Y轴标签
    plt.title(f'{title} - 准确率曲线')  # 子图标题
    plt.legend()  # 显示图例
    plt.grid(True, alpha=0.3)  # 显示网格

    plt.tight_layout()  # 调整子图布局，避免重叠
    plt.savefig(f'{title.replace(" ", "_")}_metrics.png', dpi=300)  # 保存高清图片（300DPI）
    plt.show()  # 显示图表


# ======================
# 主执行函数
# ======================

def main():
    # 获取数据加载器
    trainloader, testloader = get_cifar10_dataloaders(batch_size=128)

    # 创建ResNet-18模型
    model = ResNet18(BasicBlock, [2, 2, 2, 2], num_classes=10)

    # 训练模型
    metrics = train_model(
        model,
        trainloader,
        testloader,
        epochs=200,
        lr=0.1
    )

    # 可视化指标
    plot_metrics(metrics)

    # 测试可扩展模型
    print("\n测试可扩展ResNet模型:")
    resnet50_model = resnet50(num_classes=10)
    print("ResNet-50参数量:", sum(p.numel() for p in resnet50_model.parameters()))

    resnet101_model = resnet101(num_classes=10)
    print("ResNet-101参数量:", sum(p.numel() for p in resnet101_model.parameters()))


if __name__ == "__main__":
    main()