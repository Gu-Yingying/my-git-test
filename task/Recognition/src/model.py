import torch.nn as nn


class CNN(nn.Module):
    """一个简单的卷积神经网络，输入 3x32x32 的 CIFAR-10 图像，输出 10 个类别的分数。"""

    def __init__(self, num_classes=10):
        super().__init__()
        # 特征提取部分：两层「卷积 + ReLU + 池化」，图像 32x32 -> 16x16 -> 8x8
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3通道 -> 32通道
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 尺寸减半：32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 32通道 -> 64通道
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 16x16 -> 8x8
        )
        # 分类部分：把特征图拉平后过两层全连接
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_model(num_classes=10):
    """创建一个新的模型实例（训练和评估都用这个函数，保证结构一致）。"""
    return CNN(num_classes=num_classes)