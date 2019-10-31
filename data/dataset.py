
import os
import json

import torch
import numpy as np


class ActivityRecognitionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_json, dataset_root):
        # Load the dataset
        with open(dataset_json, 'r') as jf:
            self.dataset_map = json.load(jf)

    def __len__(self):
        return len(self.dataset_map)

    def __getitem__(self, idx):
        elem = self.dataset_map[idx]
        return {
            'video': np.load(os.path.join(dataset_root, elem['video_path'])),
            'class': elem['class'],
        }
