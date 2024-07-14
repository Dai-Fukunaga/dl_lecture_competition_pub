import torch
import os
import torch.nn.functional as F
from scipy.signal import butter, sosfiltfilt


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        data_dir: str = "data",
        target_fs=128,
        original_fs=1000,
        low_cutoff=1,
        high_cutoff=40,
        baseline_period=(0, 100),
        device="cuda",
    ) -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.device = device

        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt")).to(self.device)
        self.subject_idxs = torch.load(
            os.path.join(data_dir, f"{split}_subject_idxs.pt")
        )

        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert (
                len(torch.unique(self.y)) == self.num_classes
            ), "Number of classes do not match."

        # 前処理のパラメータを保存
        self.target_fs = target_fs
        self.original_fs = original_fs
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.baseline_period = baseline_period

        # データ前処理
        self.X = self.preprocess_data(self.X)
        self.X.to("cpu")

    def preprocess_data(self, data):
        # リサンプリング
        # data = self.resample(data, self.target_fs, self.original_fs)

        # フィルタリング
        # data = self.apply_filter(
        #     data, self.low_cutoff, self.high_cutoff, self.target_fs
        # )

        # スケーリング
        data = (data - data.mean(dim=-1, keepdim=True)) / data.std(dim=-1, keepdim=True)

        # ベースライン補正
        baseline = data[..., self.baseline_period[0] : self.baseline_period[1]].mean(
            dim=-1, keepdim=True
        )
        data = data - baseline

        return data

    def resample(self, data, target_fs, original_fs):
        num_samples = int(data.size(-1) * (target_fs / original_fs))
        data = data.unsqueeze(
            1
        )  # 形状を (batch_size, channels, length) -> (batch_size, 1, channels, length) に変更
        data = F.interpolate(data, size=num_samples, mode="linear", align_corners=False)
        return data.squeeze(
            1
        )  # 形状を (batch_size, 1, channels, new_length) -> (batch_size, channels, new_length) に戻す

    def apply_filter(self, data, low_cutoff, high_cutoff, fs, order=5):
        nyquist = 0.5 * fs
        low = low_cutoff / nyquist
        high = high_cutoff / nyquist

        # バンドパスフィルタの設計
        sos = butter(order, [low, high], btype="band", output="sos")

        # フィルタリングの適用
        filtered_data = []
        for d in data:
            d_np = d.cpu().numpy()
            filtered = sosfiltfilt(sos, d_np, axis=-1)
            filtered_data.append(torch.tensor(filtered, device=self.device))

        return torch.stack(filtered_data)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
