# -*- coding: utf-8 -*-
"""
食物分类 · Flask 前端服务
上传图片 → CNN 推理 → 返回类别、置信度与各类别概率。
"""
import os
import uuid
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify

MODEL_PATH = "./models/food_cnn.pth"
UPLOAD_DIR = "./static/uploads"
IMG_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
os.makedirs(UPLOAD_DIR, exist_ok=True)


class FoodCNN(nn.Module):
    """与 train_cnn.py 保持一致的网络结构。"""
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
model = None
class_names = []
if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    model = FoodCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"模型加载成功，类别：{class_names}")
else:
    print("⚠️ 未找到模型文件，请先运行 train_cnn.py 训练模型。")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@app.route("/")
def index():
    return render_template("index.html", categories=class_names)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "模型尚未就绪，请先训练模型。"}), 500
    if "file" not in request.files:
        return jsonify({"error": "未检测到上传文件。"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空。"}), 400

    ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        image = Image.open(save_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        confidence, pred_idx = torch.max(probs, 0)
        prob_list = [
            {"name": class_names[i], "prob": round(probs[i].item() * 100, 2)}
            for i in range(len(class_names))
        ]
        prob_list.sort(key=lambda x: x["prob"], reverse=True)
        return jsonify({
            "category": class_names[pred_idx.item()],
            "confidence": round(confidence.item() * 100, 2),
            "image_url": f"/static/uploads/{filename}",
            "probs": prob_list,
        })
    except Exception as e:
        return jsonify({"error": f"识别失败：{e}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
