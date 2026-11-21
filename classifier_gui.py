# -*- coding: utf-8 -*-
"""
食物分类 · Tkinter 桌面版界面
黑色背景图片显示框（图片居中），下方显示预测类别与置信度。
"""
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

MODEL_PATH = "./models/food_cnn.pth"
IMG_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# 加载模型
model, class_names = None, []
if os.path.exists(MODEL_PATH):
    ckpt = torch.load(MODEL_PATH, map_location=device)
    class_names = ckpt["class_names"]
    model = FoodCNN(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class FoodApp:
    def __init__(self, root):
        self.root = root
        self.root.title("食物智能分类系统")
        self.root.resizable(False, False)
        self.current_path = None
        self.photo = None

        # 顶部标题
        tk.Label(root, text="食物智能分类系统", font=("微软雅黑", 18, "bold"),
                 fg="#333").pack(pady=(16, 2))
        tk.Label(root, text="基于 PyTorch 卷积神经网络 · 选择图片自动识别",
                 font=("微软雅黑", 9), fg="#888").pack(pady=(0, 10))

        # 黑色背景图片显示框
        self.img_frame = tk.Frame(root, width=420, height=300, bg="black")
        self.img_frame.pack(padx=20)
        self.img_frame.pack_propagate(False)
        self.img_label = tk.Label(self.img_frame, bg="black",
                                  text="图片将显示在此处", fg="#888",
                                  font=("微软雅黑", 10))
        self.img_label.pack(expand=True)

        # 按钮区
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=14)
        tk.Button(btn_frame, text="选择图片", width=12, command=self.choose_image,
                  bg="#4a90d9", fg="white", font=("微软雅黑", 10)).pack(side="left", padx=6)
        tk.Button(btn_frame, text="开始识别", width=12, command=self.predict,
                  bg="#e5524b", fg="white", font=("微软雅黑", 10)).pack(side="left", padx=6)

        # 结果区
        self.result_label = tk.Label(root, text="预测类别：—    置信度：—",
                                     font=("微软雅黑", 13, "bold"), fg="#e5524b")
        self.result_label.pack(pady=(0, 18))

        if model is None:
            messagebox.showwarning("提示", "未找到模型文件，请先运行 train_cnn.py 训练。")

    def choose_image(self):
        path = filedialog.askopenfilename(
            title="选择食物图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.current_path = path
        image = Image.open(path).convert("RGB")
        image.thumbnail((400, 280))
        self.photo = ImageTk.PhotoImage(image)
        self.img_label.config(image=self.photo, text="")
        self.img_label.image = self.photo
        self.result_label.config(text="预测类别：—    置信度：—")

    def predict(self):
        if model is None:
            messagebox.showerror("错误", "模型尚未就绪。")
            return
        if not self.current_path:
            messagebox.showwarning("提示", "请先选择一张图片。")
            return
        try:
            image = Image.open(self.current_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]
            conf, idx = torch.max(probs, 0)
            category = class_names[idx.item()]
            self.result_label.config(
                text=f"预测类别：{category}    置信度：{conf.item() * 100:.2f}%")
        except Exception as e:
            messagebox.showerror("错误", f"识别失败：{e}")


if __name__ == "__main__":
    win = tk.Tk()
    FoodApp(win)
    win.mainloop()
