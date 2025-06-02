# Pipeline used from midquarter checkpoint for simplicity and consistency
# -------------------------------------------------------------------

# Pytorch Specific Libraries
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

# Data
import numpy as np

class TrajectoryDatasetTrain(Dataset):
    def __init__(self, data, scale=10.0, augment=True):
        """
        data: Shape (N, 50, 110, 6) Training data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        augment: Whether to apply data augmentation (only for training)
        """
        self.data = data
        self.scale = scale
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scene = self.data[idx]  # Shape: (50, 110, 6)

        # Take the first 30 agents, since smaller the agent index the more important it is
        hist = scene[:20, :50, :].copy()
        future = torch.tensor(scene[0, 50:, :2].copy(), dtype=torch.float32)

        # === Optional Data Augmentation ===
        if self.augment:
            if np.random.rand() < 0.5:
                theta = np.random.uniform(-np.pi, np.pi)
                R = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
                hist[..., :2] = hist[..., :2] @ R
                hist[..., 2:4] = hist[..., 2:4] @ R
                future = future @ R
            if np.random.rand() < 0.5:
                hist[..., 0] *= -1
                hist[..., 2] *= -1
                future[:, 0] *= -1

        # === Centering and normalization ===
        origin = hist[0, 49, :2].copy()  # Ego car position at t=49
        hist[..., :2] -= origin
        future -= origin

        hist[..., :4] /= self.scale
        future /= self.scale

        return Data(
            x=torch.tensor(hist, dtype=torch.float32),        # (10, 50, 6)
            y=future.type(torch.float32),                     # (60, 2)
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )

    

class TrajectoryDatasetTest(Dataset):
    def __init__(self, data, scale=10.0):
        """
        data: Shape (N, 50, 110, 6) Testing data
        scale: Scale for normalization (suggested to use 10.0 for Argoverse 2 data)
        """
        self.data = data
        self.scale = scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Testing data only contains historical trajectory
        scene = self.data[idx]  # (50, 50, 6)
        hist = scene[:20, :, :].copy()
        
        origin = hist[0, 49, :2].copy()
        hist[..., :2] = hist[..., :2] - origin
        hist[..., :4] = hist[..., :4] / self.scale

        data_item = Data(
            x=torch.tensor(hist, dtype=torch.float32),
            origin=torch.tensor(origin, dtype=torch.float32).unsqueeze(0),
            scale=torch.tensor(self.scale, dtype=torch.float32),
        )
        return data_item