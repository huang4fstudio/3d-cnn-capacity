
import os
import json

import torch
import numpy as np
import skvideo.io


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
            'video': video_frames
            'class': elem['class'],
        }
