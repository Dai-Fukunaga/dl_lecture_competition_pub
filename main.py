import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.densenet import DenseNetClassifier
from src.resnet2d import resnet50_2d
from src.resnet1d import resnet50_1d

import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms

from src.utils import set_seed

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        return (data - self.mean) / self.std

def calculate_mean_std(dataset):
    all_data = torch.cat([dataset[i][0] for i in range(len(dataset))], dim=1)
    mean = all_data.mean()
    std = all_data.std()
    return mean, std


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    # train_set = ThingsMEGDataset("train", args.data_dir, transforms=None)
    # mean, std = calculate_mean_std(train_set)
    train_transforms = transforms.Compose(
        [
            # Normalize(mean, std),
        ]
    )
    val_test_transforms = train_transforms
    train_set = ThingsMEGDataset("train", args.data_dir, transforms=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir, transforms=val_test_transforms)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir, transforms=val_test_transforms)
    test_loader = torch.utils.data.DataLoader(
        test_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = resnet50_1d(
        num_classes=train_set.num_classes, in_channels=train_set.num_channels, dropout_rate=args.dropout_rate,
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ------------------
    #   Start training
    # ------------------
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            y_pred = model(X)

            loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)

            with torch.no_grad():
                y_pred = model(X)

            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}"
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

    preds = []
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):
        preds.append(model(X.to(args.device)).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
