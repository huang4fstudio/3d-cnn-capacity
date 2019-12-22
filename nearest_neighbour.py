import torch
import numpy as np
import click
import tqdm
from data.dataset import ActivityRecognitionDataset
from sklearn.neighbors import NearestNeighbors

TRAIN_LEN = 5201 # 52016
VAL_LEN = 578 # 5784

def load_all_pts(dataset, load_limit):
    batch_size = 1
    data_loader_1 = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=5)

    all_pts = np.ones([load_limit, 224, 224, 3], dtype=np.float32)
    all_l = np.ones([load_limit,], dtype=np.float32)
    for i in range(load_limit):
        all_pts[i, 0, 0, 0] = i
    '''
    for i, e in enumerate(tqdm.tqdm(data_loader_1)):
        data_1 = e['video']
        labels = e['class']
        mean_frame_1 = data_1.mean(dim=1)
        all_pts[i * batch_size:(i + 1) * batch_size] = mean_frame_1.numpy()
        all_l[i * batch_size:(i + 1) * batch_size] = labels.numpy()
        if i == load_limit / batch_size - 1:
            break
    
    '''
    all_l = np.array(all_l)
    all_pts = all_pts.reshape([load_limit, -1])
    all_l = all_l[:load_limit]

    return all_pts, all_l


def main():
    dataset = ActivityRecognitionDataset('/data/ucf101/ucf101_train.json', '/data/ucf101/downsampled/')
    all_pts, all_l = load_all_pts(dataset, TRAIN_LEN)
    print('Fitting...')
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(all_pts)
    
    val_dataset = ActivityRecognitionDataset('/data/ucf101/ucf101_val.json', '/data/ucf101/downsampled/')
    all_pts_val, all_l_val = load_all_pts(val_dataset, VAL_LEN)

    _, indices = nbrs.kneighbors(all_pts)
    correct_cls = sum(all_l[indices[:, 1]] == all_l)
    data_cls = len(all_l)
    print('Train Acc: {}'.format(correct_cls/data_cls))
   
    _, indices = nbrs.kneighbors(all_pts_val)

    data_cls = len(all_l_val)
    correct_cls = sum(all_l[indices[:, 1]] == all_l_val)
                    
    print('Val Acc: {}'.format(correct_cls/data_cls))
    print('Done!')


if __name__ == '__main__':
    main()
