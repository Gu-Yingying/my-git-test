import os

import torch
import torchvision
import torchvision.transforms as transforms

from model import create_model

# ===== 可调参数 =====
EPOCHS = 10          # 训练轮数
BATCH_SIZE = 128     # 每次喂给网络的图片数量
LR = 1e-3            # 学习率
DATA_DIR = "./data"  # 数据集下载位置
WEIGHT_PATH = "./cifar10_cnn.pth"  # 权重保存位置

# 有显卡用显卡，没有就用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理：转成 tensor，并按 CIFAR-10 的均值/方差做标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# 第一次运行会自动下载 CIFAR-10 训练集（约 170MB）
print("正在加载训练集（首次运行需要下载）...")
train_set = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)

# 模型、损失函数、优化器
model = create_model().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ===== 训练循环 =====
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()                # 清空上一步的梯度
        outputs = model(images)              # 前向传播：得到预测分数
        loss = criterion(outputs, labels)    # 计算预测和真实标签的差距
        loss.backward()                      # 反向传播：计算梯度
        optimizer.step()                     # 更新网络参数
        running_loss += loss.item()
    print(f"第 {epoch + 1}/{EPOCHS} 轮，平均损失: {running_loss / len(train_loader):.4f}")

# 保存权重（只保存参数，文件很小，约 1MB）
torch.save(model.state_dict(), WEIGHT_PATH)
print(f"训练完成，权重已保存到: {os.path.abspath(WEIGHT_PATH)}")