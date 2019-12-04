
import os
import json

import torch
import numpy as np
import skvideo.io
import sys
import random

class ActivityRecognitionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_json, dataset_root, pad_frames=64):
        # Load the dataset
        with open(dataset_json, 'r') as jf:
            self.dataset_map = json.load(jf)
        self.dataset_root = dataset_root

    def __len__(self):
        return len(self.dataset_map)

    def __getitem__(self, idx):
        elem = self.dataset_map[idx]

        if 'feature_type' in elem and 'feature_type' == 'dr_mp4':
            video_frames = skvideo.io.vread(os.path.join(self.dataset_root, elem['path']))
            video_frames = video_frames.astype(np.float32) / 255.0
        else:
            video_frames = np.load(os.path.join(self.dataset_root, elem['path'])) / 255.0,

        video_frames = video_frames[:self.pad_frames]
        if video_frames.shape[0] != self.pad_frames:
            video_frames = np.pad(video_frames[:self.pad_frames], (0,self.pad_frames - video_frames.shape[0],0,0,0,0,0,0))

        return {
            'video': video_frames,
            'class': elem['class'],
        }

class RandomDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_size, n_classes):
        self.dataset_size = dataset_size
        self.n_classes = n_classes
        self.seed_map = {}

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx not in self.seed_map:
            self.seed_map[idx] = random.randint(0, 2 ** 32 - 1)
        np.random.seed(self.seed_map[idx])
        return {
            'video': np.random.rand(64, 224, 224, 3),
            'class': np.random.randint(0, self.n_classes)
        }
