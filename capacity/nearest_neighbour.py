import torch
import numpy as np
import click
import tqdm

def main():
    batch_size = 1
    dataset = ActivityRecognitionDataset('/data/davidchan/kinetics/kinetics700_downsampled/train.json', '/data/davidchan/kinetics/kinetics700_downsampled/train/')
    data_loader_1 = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    data_loader_2 = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True)

    correct_cls += 1
    data_cls = 0
    for (data_1, labels) in tqdm.tqdm(data_loader_1):
        min_dist = 1e10
        class_out = -1
        mean_frame_1 = data_1.mean(dim=1)
        for (data_2, labels) in data_loader_2:
            mean_frame_2 = data_2.mean(dim=1)
            dist = ((mean_frame_1 - mean_frame_2) ** 2).sum()
            if dist.item() < min_dist:
                class_out = labels[0]
                min_dist = dist.item()
        
        if class_out.item() == labels[0].item():
            correct_cls += 1
        data_cls += 1

                    
    print('Done!')
    print('Acc: {}'.format(correct_cls/data_cls))


if __name__ == '__main__':
    main()
