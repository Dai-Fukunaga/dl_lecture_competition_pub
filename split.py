import os
import shutil
import random


def split_data(source, train_dir, val_dir, split_ratio=0.8):
    # クラスごとにフォルダを作成
    for class_folder in os.listdir(source):
        class_path = os.path.join(source, class_folder)
        if os.path.isdir(class_path):
            os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_folder), exist_ok=True)

            # ファイルのリストを取得しシャッフル
            files = os.listdir(class_path)
            random.shuffle(files)

            # 分割点のインデックスを計算
            split_idx = int(len(files) * split_ratio)

            # 訓練データと検証データに分割
            train_files = files[:split_idx]
            val_files = files[split_idx:]

            # 訓練データをコピー
            for file in train_files:
                shutil.copy(
                    os.path.join(class_path, file),
                    os.path.join(train_dir, class_folder, file),
                )

            # 検証データをコピー
            for file in val_files:
                shutil.copy(
                    os.path.join(class_path, file),
                    os.path.join(val_dir, class_folder, file),
                )


def delete_other_folders(source, keep_folders):
    for folder in os.listdir(source):
        folder_path = os.path.join(source, folder)
        if os.path.isdir(folder_path) and folder not in keep_folders:
            shutil.rmtree(folder_path)
            print(f"削除しました: {folder_path}")


# 元のデータが入っているフォルダ
source_dir = "Images"

# 訓練データと検証データのフォルダを作成
train_dir = "Images/train"
val_dir = "Images/val"

# データを分割
split_data(source_dir, train_dir, val_dir, split_ratio=0.8)

# 'train'と'val'フォルダ以外を削除
delete_other_folders(source_dir, keep_folders=["train", "val"])

print("データの分割が完了しました。")
