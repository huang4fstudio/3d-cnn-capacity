
import os
import tqdm
from alexandria.util import write_json, load_json

ROOT_FOLDER = '/data/davidchan/kinetics/dataset-400/test/'
CAT_MAP = '/home/davidchan/Projects/kinetics/data/400/kinetics_400_catmap.json'
OUTPUT_PATH = '/home/davidchan/Projects/kinetics/data/400/kinetics_400_test.json'

# Discover categories
categories = [(f.path, f.name) for f in os.scandir(ROOT_FOLDER) if f.is_dir()]
if CAT_MAP is None:
    cat_map = {i[1]:idx for idx, i in enumerate(categories)}
    write_json(cat_map, 'kinetics_400_catmap.json')
else:
    cat_map = load_json(CAT_MAP)

print(categories)

# Write the objects
samples = []
for elem in tqdm.tqdm(categories):
    files = [f.path for f in os.scandir(elem[0]) if f.is_file()]
    for f in files:
        samples.append({
            'path': f,
            'class': cat_map[elem[1]],
            'class_raw': elem[1],
        })

print(samples[-1])
print(len(samples))
write_json(samples, OUTPUT_PATH)
