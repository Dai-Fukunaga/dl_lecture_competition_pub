import os
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.models import resnet50_1d
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir="Images", project="Image-Classification")

    # ------------------
    #    Dataloader
    # ------------------
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    image_datasets = {
        x: datasets.ImageFolder(
            root=os.path.join("Images", x), transform=data_transforms[x]
        )
        for x in ["train", "val"]
    }

    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}

    dataloaders = {
        x: DataLoader(image_datasets[x], shuffle=(x == "train"), **loader_args)
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}
    class_names = image_datasets["train"].classes

    # ------------------
    #       Model
    # ------------------
    model = resnet50_1d(num_classes=len(class_names), in_channels=3).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------
    max_val_acc = 0
    accuracy = Accuracy(task="multiclass", num_classes=len(class_names)).to(args.device)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for inputs, labels in tqdm(dataloaders["train"], desc="Train"):
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1).to(
                args.device
            )  # 1次元に変換
            labels = labels.to(args.device)

            y_pred = model(inputs)

            loss = F.cross_entropy(y_pred, labels)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(y_pred, labels)
            train_acc.append(acc.item())

        model.eval()
        for inputs, labels in tqdm(dataloaders["val"], desc="Validation"):
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1).to(
                args.device
            )  # 1次元に変換
            labels = labels.to(args.device)

            with torch.no_grad():
                y_pred = model(inputs)

            val_loss.append(F.cross_entropy(y_pred, labels).item())
            val_acc.append(accuracy(y_pred, labels).item())

        print(
            f"Epoch {epoch + 1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}"
        )
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": np.mean(train_loss),
                    "train_acc": np.mean(train_acc),
                    "val_loss": np.mean(val_loss),
                    "val_acc": np.mean(val_acc),
                }
            )

        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(
        torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device)
    )

    # 今回は評価部分は省略していますが、必要に応じて実装してください


if __name__ == "__main__":
    run()
