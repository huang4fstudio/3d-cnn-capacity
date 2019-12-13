import torch
import numpy as np
import click
import tqdm
from data.dataset import ActivityRecognitionDataset
from sklearn.neighbors import NearestNeighbors

def main():
    batch_size = 1
    dataset = ActivityRecognitionDataset('/data/ucf101/ucf101_train.json', '/data/ucf101/downsampled/')
    data_loader_1 = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=5)
    data_loader_2 = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=5)

    data_cls = 0
    all_pts = []
    all_l = []
    for e in tqdm.tqdm(data_loader_1):
        data_1 = e['video']
        labels = e['class']
        mean_frame_1 = data_1.mean(dim=1)
        all_pts.append(mean_frame_1.item())
        all_l.append(labels[0].item())
        data_cls += 1
    
    all_pts = np.array(all_pts)
    all_l = np.array(all_l)
    
    nbrs = NearestNeighbors(n_neighbors=2).fit(all_pts)
    _, indices = nbrs.kneighbors(all_pts)

    correct_cls = sum(all_l[indices[:, 1]] == all_l[indices[:, 0]])
                    
    print('Done!')
    print('Acc: {}'.format(correct_cls/data_cls))


if __name__ == '__main__':
    main()
