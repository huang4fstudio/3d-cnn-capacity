import torch
import numpy as np
import click
import tqdm
from data.dataset import ActivityRecognitionDataset

def main():
    batch_size = 1
    dataset = ActivityRecognitionDataset('/data/ucf101/ucf101_train.json', '/data/ucf101/downsampled/')
    data_loader_1 = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=5)
    data_loader_2 = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=5)

    correct_cls = 0
    data_cls = 0
    for e in tqdm.tqdm(data_loader_1):
        data_1 = e['video']
        labels = e['class']
        min_dist = 1e10
        class_out = -1
        mean_frame_1 = data_1.mean(dim=1)
        for e in data_loader_2:
            data_2 = e['video']
            labels_2 = e['class']

            mean_frame_2 = data_2.mean(dim=1)
            dist = ((mean_frame_1 - mean_frame_2) ** 2).sum()
            if dist.item() < min_dist:
                class_out = labels_2[0]
                min_dist = dist.item()
        
        if class_out.item() == labels[0].item():
            correct_cls += 1
        data_cls += 1

                    
    print('Done!')
    print('Acc: {}'.format(correct_cls/data_cls))


if __name__ == '__main__':
    main()
