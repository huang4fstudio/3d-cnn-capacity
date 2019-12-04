import os
import random
import glob

import tqdm
import numpy as np
import multiprocessing.pool

from alexandria.util import write_json, load_json
from alexandria.util.video import load_video_with_fps, downsample_video_fps, resize_video, shatter_video

# Open the dataset
dataset_root_path = '/data/davidchan/kinetics/kinetics700/train/'
output_root_path = '/data/davidchan/kinetics/kinetics700_downsampled/'

def proc_sample(sample):
    idx, s = sample

    # Subsample and preprocess the video
    try:
        data = load_video_with_fps(s['path'])
        if data['frames'] is not None and data['frames'].shape[0] > 0:

            # Shatter the loaded video
            resized = resize_video(data['frames'], (224,224))
            downsampled = downsample_video_fps(data['frames'], data['fps'] if data['fps'] else 30, 5)
            downsampled = downsampled[:64]
            if len(downsampled) < 64:
                # Pad the data
                downsampled = np.pad(downsampled, (0, 64-len(downsampled)), mode='constant')

            # Write the file
            clip = downsampled.astype(np.uint8)
            np.save(os.path.join(output_root_path, '{}.npy'.format(idx)), clip)

            output = {
                'path': '{}.npy'.format(idx),
                'video_path': s['path'],
                'id': idx,
                'class': s['class'],
                'class_raw': s['class_raw'],
                'split': s['split'],
            }
            return output

        return None
    except Exception as ex:
        print(s, ex)
        return None


# Discover samples for the dataset
files = glob.glob(dataset_root_path + '/**/*.mp4')
if not os.path.exists('/data/davidchan/kinetics/json/class_map.json'):
    classes = set([f.split(os.sep)[-2] for f in files])
    class_map = {c:idx for idx, c in enumerate(classes)}
    write_json(class_map, '/data/davidchan/kinetics/json/class_map.json')
else:
    class_map = load_json('/data/davidchan/kinetics/json/class_map.json')

print(class_map)

dataset = [{
    'path': f,
    'class': class_map[f.split(os.sep)[-2]],
    'class_raw': f.split(os.sep)[-2],
    'split': 'train' if random.random() < 0.9 else 'val',
} for f in files]

elems = []
with multiprocessing.pool.Pool(24) as pool:
    for e in tqdm.tqdm(pool.imap_unordered(proc_sample, enumerate(dataset)), total=len(dataset)):
        if e is not None:
            elems.append(e)


write_json([e for e in elems if e['split'] == 'train'], '/data/davidchan/kinetics/json/kinetics700_train.json')
write_json([e for e in elems if e['split'] == 'val'], '/data/davidchan/kinetics/json/kinetics700_val.json')
