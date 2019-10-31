
import json
import torch
import numpy as np


class Kinetics400(torch.utils.data.Dataset):
    def __init__(self, dataset_json):
        # Load the dataset
        with open(dataset_json, 'r') as jf:
            self.dataset_map = json.load(jf)

    def __len__(self):
        return len(self.dataset_map)

    def __getitem__(self, idx):
        elem = self.dataset_map[idx]
        return {
            'video': np.load(elem['video_path']),
            'class': elem['class'],
        }
