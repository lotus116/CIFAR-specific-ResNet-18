"""CIFAR-10 最终提交模型。

本文件只包含网络结构和 ``build_model()`` 工厂函数。导入本模块不会下载数据、
启动训练或读取权重，便于在任意推理程序中安全导入。
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet 基本残差块：两个 3x3 卷积，加一条恒等或投影捷径。"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 卷积后紧接 BN，因此卷积中的 bias 是冗余参数，显式关闭。
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 当空间尺寸或通道数改变时，用 1x1 卷积把捷径分支投影到相同形状；
        # 其余情况直接使用恒等映射，既省参数又保留原始信息。
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 主分支的第二个 BN 后暂不激活，先与捷径相加，再统一 ReLU。
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CIFARResNet18(nn.Module):
    """面向 32x32 CIFAR 图像、仅由基础 PyTorch 层实现的 ResNet-18。

    与 ImageNet 版本不同，这里不用 7x7/stride-2 stem 和早期最大池化，以免在
    低分辨率输入上过早丢失空间信息。四个 stage 分别输出 64/128/256/512 通道。
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        # 全局平均池化替代大规模全连接层，可减少参数并降低过拟合。
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def _make_layer(self, channels, blocks, stride):
        """构造一个 stage；首块负责降采样，后续块保持形状。"""
        layers = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """使用适合 ReLU 的 Kaiming 初始化，并初始化 BN 的仿射参数。"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # 输入约定：[N, 3, 32, 32]，且已由外部变换归一化到 [-1, 1]。
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_model():
    """返回与 ``model.pt`` 完全匹配的新模型；评测端会无参数调用。"""
    return CIFARResNet18()
