
import os
import tqdm
from alexandria.util import write_json, load_json

ROOT_FOLDERS = [
    '/data/davidchan/kinetics/kinetics700_downsampled/train',
    '/data/davidchan/kinetics/kinetics700_downsampled/val',
    # '/data/davidchan/kinetics/kinetics700_downsampled/test',
]
CAT_MAP = '/data/davidchan/kinetics/kinetics700_downsampled/catmap.json'
OUTPUT_PATHS = [
    '/data/davidchan/kinetics/kinetics700_downsampled/train.json',
    '/data/davidchan/kinetics/kinetics700_downsampled/val.json',
    # '/data/davidchan/kinetics/kinetics700_downsampled/test.json',
]

# Discover categories
categories = [(f.path, f.name) for f in os.scandir(ROOT_FOLDERS[0]) if f.is_dir()]
if not os.path.exists(CAT_MAP):
    cat_map = {i[1]:idx for idx, i in enumerate(categories)}
    write_json(cat_map, CAT_MAP)
else:
    cat_map = load_json(CAT_MAP)
print(categories)

for i, RF in enumerate(ROOT_FOLDERS):
    categories = [(f.path, f.name) for f in os.scandir(RF) if f.is_dir()]
    samples = []

    # Scan the categories
    for elem in tqdm.tqdm(categories):
        files = [f.path for f in os.scandir(elem[0]) if f.is_file()]
        for f in files:
            samples.append({
                'path': f,
                'class': cat_map.get(elem[1], -1),
                'class_raw': elem[1],
                'video_type': 'dr_mp4',
            })

    print(samples[-1])
    print(len(samples))
    write_json(samples, OUTPUT_PATHS[i])
