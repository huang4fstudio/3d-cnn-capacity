import os

import tqdm
import numpy as np
import multiprocessing.pool

from alexandria.util import write_json, load_json
from alexandria.util.video import load_video_with_fps, downsample_video_fps, resize_video

# Open the dataset
dataset = load_json('data/400/kinetics_400_validate.json')
output_root_path = '/data/davidchan/kinetics/400_features/validate'

def proc_sample(sample):

    idx, s = sample

    # Subsample and preprocess the video
    try:
        data = load_video_with_fps(s['path'])
        if data['frames'] is not None and data['frames'].shape[0] > 0:
            downsampled = downsample_video_fps(data['frames'], data['fps'] if data['fps'] else 30, 5)
            resized = resize_video(downsampled, (224,224))

            # Write the file
            np.save(os.path.join(output_root_path, '{}.npy'.format(idx)), resized)

            output = {
                'path': os.path.join(output_root_path, '{}.npy'.format(idx)),
                'video_path': s['path'],
                'id': idx,
                'class': s['class'],
                'class_raw': s['class_raw'],
            }

            return output
        return None
    except Exception as ex:
        print(s, ex)
        return None

elems = []
with multiprocessing.pool.Pool(48) as pool:
    for e in tqdm.tqdm(pool.imap_unordered(proc_sample, enumerate(dataset)), total=len(dataset)):
        if e is not None:
            elems.append(e)


write_json(elems, 'kinetics_400_downsampled_validate.json')
