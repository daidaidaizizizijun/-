# -*- coding: utf-8 -*-
"""
食物分类 · CNN 训练
PyTorch 实现，带数据容错、数据增强、训练曲线与 checkpoint 保存。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

IMG_SIZE = 128
EPOCHS = 20
BATCH_SIZE = 16
DATA_DIR = "./food_images_split"
MODEL_PATH = "./models/food_cnn.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 中文字体兜底
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def custom_loader(path):
    """二进制流读取并强制 RGB，损坏文件返回 None。"""
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            img.verify()
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except Exception:
        print(f"  跳过损坏/无效图片：{path}")
        return None


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class FilteredDataset(Dataset):
    """过滤 loader 返回 None 的脏样本。"""
    def __init__(self, base_dataset):
        self.samples = []
        for idx in range(len(base_dataset)):
            sample, target = base_dataset[idx]
            if sample is not None:
                self.samples.append((sample, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class FoodCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"未找到数据集 {DATA_DIR}，请先依次运行 download_data.py、split_data.py")
        return

    raw_train = datasets.ImageFolder(os.path.join(DATA_DIR, "train"),
                                     transform=train_transform, loader=custom_loader)
    raw_test = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),
                                    transform=val_transform, loader=custom_loader)
    train_set = FilteredDataset(raw_train)
    test_set = FilteredDataset(raw_test)
    class_names = raw_train.classes
    print(f"计算设备：{device}")
    print(f"类别（按 Unicode 排序）：{class_names}")
    print(f"有效训练样本：{len(train_set)} | 有效测试样本：{len(test_set)}")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=0)

    model = FoodCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    history = {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}

    for epoch in range(EPOCHS):
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * images.size(0)
            run_correct += (outputs.argmax(1) == labels).sum().item()
            run_total += images.size(0)
        train_loss = run_loss / run_total
        train_acc = run_correct / run_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)
        test_loss = val_loss / val_total
        test_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(f"Epoch {epoch + 1:2d}/{EPOCHS} | 训练损失 {train_loss:.4f} 训练准确率 {train_acc:.4f}"
              f" | 测试损失 {test_loss:.4f} 测试准确率 {test_acc:.4f}")

    final_acc = history["test_acc"][-1]
    os.makedirs("./models", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "final_accuracy": final_acc,
        "epochs": EPOCHS,
        "history": history,
    }, MODEL_PATH)
    print(f"\n测试集最终准确率：{final_acc:.4f}")
    print(f"模型已保存：{MODEL_PATH}")

    # 训练曲线
    try:
        epochs_x = range(1, EPOCHS + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].plot(epochs_x, history["train_acc"], label="训练准确率")
        axes[0].plot(epochs_x, history["test_acc"], label="测试准确率")
        axes[0].set_title("准确率曲线"); axes[0].set_xlabel("Epoch"); axes[0].legend()
        axes[1].plot(epochs_x, history["train_loss"], label="训练损失")
        axes[1].plot(epochs_x, history["test_loss"], label="测试损失")
        axes[1].set_title("损失曲线"); axes[1].set_xlabel("Epoch"); axes[1].legend()
        plt.tight_layout()
        plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("训练曲线已保存：training_history.png")
    except Exception as e:
        print(f"训练曲线保存失败：{e}")


if __name__ == "__main__":
    main()
