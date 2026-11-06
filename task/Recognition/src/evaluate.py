import os

import torch
import torchvision
import torchvision.transforms as transforms

from model import create_model

BATCH_SIZE = 256
DATA_DIR = "./data"
WEIGHT_PATH = "./cifar10_cnn.pth"
NUM_SHOW = 8  # 展示多少个样本的真实/预测类别

CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 重新创建模型并加载训练好的权重
model = create_model().to(device)
if not os.path.exists(WEIGHT_PATH):
    raise FileNotFoundError(f"找不到权重文件 {WEIGHT_PATH}，请先运行 train.py 训练")
model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
model.eval()
print(f"已加载权重: {os.path.abspath(WEIGHT_PATH)}")

# 测试集只做评估，不参与训练
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
test_set = torchvision.datasets.CIFAR10(
    root=DATA_DIR, train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False
)

total = 0          # 总样本数
correct = 0        # 正确分类数
class_correct = [0] * 10   # 每个类别猜对的数量
class_total = [0] * 10     # 每个类别的样本总数
shown = 0

with torch.no_grad():  # 评估时不需要算梯度，更快更省显存
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # 分数最高的类别作为预测结果

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 统计每个类别的准确率
        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == labels[i]:
                class_correct[label] += 1

        # 打印前几个样本的真实类别和预测类别
        if shown < NUM_SHOW:
            n = min(labels.size(0), NUM_SHOW - shown)
            for i in range(n):
                t = CLASSES[labels[i].item()]
                p = CLASSES[predicted[i].item()]
                print(f"true={t}, predicted={p}")
            shown += n

# ===== 输出评估结果 =====
print()
print(f"测试集总样本数: {total}")
print(f"正确分类数: {correct}")
print(f"总体准确率: {100 * correct / total:.2f}%")
print("各类别准确率:")
for i in range(10):
    acc = 100 * class_correct[i] / class_total[i]
    print(f"  {CLASSES[i]}: {acc:.2f}%")