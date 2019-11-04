import os
import random
import glob

import tqdm
import numpy as np
import multiprocessing.pool

from alexandria.util import write_json, load_json
from alexandria.util.video import load_video_with_fps, downsample_video_fps, resize_video, shatter_video

# Open the dataset
dataset_root_path = '/data/ucf101/UCF101/'
output_root_path = '/data/ucf101/downsampled/'

def proc_sample(sample):

    idx, s = sample

    # Subsample and preprocess the video
    try:
        data = load_video_with_fps(s['path'])
        if data['frames'] is not None and data['frames'].shape[0] > 0:

            # Shatter the loaded video
            resized = resize_video(data['frames'], (224,224))
            clips = shatter_video(resized, 64, 32, True)

            # downsampled = downsample_video_fps(data['frames'], data['fps'] if data['fps'] else 30, 5)
            outputs = []
            for isx, clip in enumerate(clips):

                # Write the file
                clip = clip.astype(np.uint8)
                np.save(os.path.join(output_root_path, '{}_{}.npy'.format(idx, isx)), clip)

                output = {
                    'path': '{}_{}.npy'.format(idx, isx),
                    'video_path': s['path'],
                    'id': idx,
                    'class': s['class'],
                    'class_raw': s['class_raw'],
                    'split': s['split'],
                }
                outputs.append(output)

            return outputs
        return None
    except Exception as ex:
        print(s, ex)
        return None


# Discover samples for the dataset
files = glob.glob(dataset_root_path + '*.avi')
if not os.path.exists('class_map.json'):
    classes = set([f.split('_')[1] for f in files])
    class_map = {c:idx for idx, c in enumerate(classes)}
    write_json(class_map, 'data/ucf101/class_map.json')
else:
    class_map = load_json('data/ucf101/class_map.json')

dataset = [{
    'path': f,
    'class': class_map[f.split('_')[1]],
    'class_raw': f.split('_')[1],
    'split': 'train' if random.random() < 0.9 else 'val',
} for f in files]

elems = []
with multiprocessing.pool.Pool(6) as pool:
    for e in tqdm.tqdm(pool.imap_unordered(proc_sample, enumerate(dataset)), total=len(dataset)):
        if e is not None:
            elems += e


write_json([e for e in elems if e['split'] == 'train'], '/data/ucf101/ucf101_train.json')
write_json([e for e in elems if e['split'] == 'val'], '/data/ucf101/ucf101_val.json')
