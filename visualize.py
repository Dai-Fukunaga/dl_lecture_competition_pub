import os
import torch
from termcolor import cprint
from torch.utils.data import DataLoader
from src.datasets import ThingsMEGDataset
import matplotlib.pyplot as plt


def show_images(images, labels=None, title=None):
    batch_size = images.shape[0]
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))
    if batch_size == 1:
        axes = [axes]
    for idx in range(batch_size):
        image = images[idx]
        if image.dim() == 2:  # 2次元の場合（チャンネルがない場合）
            image = image.unsqueeze(0)  # チャンネル次元を追加
        image = image.permute(1, 2, 0).cpu().numpy()  # C x H x W -> H x W x C
        axes[idx].imshow(image)
        axes[idx].axis("off")
        if labels is not None:
            axes[idx].set_title(f"Label: {labels[idx].item()}")
    if title:
        fig.suptitle(title)
    plt.savefig("sample.png")


def main():
    data_dir = "/mnt/mp_nas_mks/labmember/d.fukunaga/data"  # データディレクトリを指定
    batch_size = 4  # 表示するバッチサイズ

    # データセットのインスタンスを作成
    train_set = ThingsMEGDataset("train", data_dir)
    val_set = ThingsMEGDataset("val", data_dir)
    test_set = ThingsMEGDataset("test", data_dir)

    # データローダーを作成
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # データセットの情報を表示
    cprint(f"Train set size: {len(train_set)}", "green")
    cprint(f"Validation set size: {len(val_set)}", "green")
    cprint(f"Test set size: {len(test_set)}", "green")
    cprint(f"Number of classes: {train_set.num_classes}", "green")
    cprint(f"Number of channels: {train_set.num_channels}", "green")
    cprint(f"Sequence length: {train_set.seq_len}", "green")

    # トレーニングデータのサンプルを表示
    for X, y, subject_idxs in train_loader:
        cprint("Training batch sample", "cyan")
        cprint(f"X shape: {X.shape}", "cyan")
        cprint(f"y shape: {y.shape}", "cyan")
        cprint(f"subject_idxs shape: {subject_idxs.shape}", "cyan")
        show_images(X, y, title="Training batch sample")
        break

    # バリデーションデータのサンプルを表示
    for X, y, subject_idxs in val_loader:
        cprint("Validation batch sample", "yellow")
        cprint(f"X shape: {X.shape}", "yellow")
        cprint(f"y shape: {y.shape}", "yellow")
        cprint(f"subject_idxs shape: {subject_idxs.shape}", "yellow")
        show_images(X, y, title="Validation batch sample")
        break

    # テストデータのサンプルを表示
    for X, subject_idxs in test_loader:
        cprint("Test batch sample", "magenta")
        cprint(f"X shape: {X.shape}", "magenta")
        cprint(f"subject_idxs shape: {subject_idxs.shape}", "magenta")
        show_images(X, title="Test batch sample")
        break


if __name__ == "__main__":
    main()
