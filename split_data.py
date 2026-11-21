# -*- coding: utf-8 -*-
"""
食物分类 · 数据集划分
逐类别按 8:2 复制到 train / test，固定随机种子保证可复现。
"""
import os
import random
import shutil

SRC_ROOT = "./food_images"
DST_ROOT = "./food_images_split"
TRAIN_RATIO = 0.8
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
random.seed(42)


def main():
    if not os.path.isdir(SRC_ROOT):
        print(f"未找到原始数据目录 {SRC_ROOT}，请先运行 download_data.py")
        return

    categories = [d for d in os.listdir(SRC_ROOT)
                  if os.path.isdir(os.path.join(SRC_ROOT, d))]
    total_train = 0
    total_test = 0

    print("========== 开始划分训练集/测试集（8:2）==========")
    for category in categories:
        src_dir = os.path.join(SRC_ROOT, category)
        # 取一次后缀、转小写、集合判断，避免 Windows 重复统计
        images = [
            name for name in os.listdir(src_dir)
            if os.path.isfile(os.path.join(src_dir, name))
            and os.path.splitext(name)[1].lower() in VALID_EXTS
        ]
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        train_dir = os.path.join(DST_ROOT, "train", category)
        test_dir = os.path.join(DST_ROOT, "test", category)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for name in train_imgs:
            shutil.copy2(os.path.join(src_dir, name), os.path.join(train_dir, name))
        for name in test_imgs:
            shutil.copy2(os.path.join(src_dir, name), os.path.join(test_dir, name))

        total_train += len(train_imgs)
        total_test += len(test_imgs)
        print(f"🍽 「{category}」：总{len(images)}张 → 训练{len(train_imgs)} | 测试{len(test_imgs)}")

    print("\n========== 划分完成 ==========")
    print(f"📊 训练集 {total_train} 张 | 测试集 {total_test} 张 | 合计 {total_train + total_test} 张")
    if total_train < 200:
        print("⚠️ 训练集不足 200 张，请先保证采集数量。")
    else:
        print("✅ 训练集数量满足不少于 200 张。")


if __name__ == "__main__":
    main()
