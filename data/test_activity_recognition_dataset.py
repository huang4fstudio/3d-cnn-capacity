
from dataset import ActivityRecognitionDataset

ds = ActivityRecognitionDataset('../data/ucf101/ucf101_train.json', '/big/davidchan/ucf101/downsampled/')

for d in ds:
    print({k:v.shape if not isinstance(v, int) else v for k,v in d.items()})

