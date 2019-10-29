
from multiprocessing.pool import Pool

import tqdm
import sys

from alexandria.data.fetch.youtube import fetch_video

def fetch_row(row):
    try:
        elems = row.split(',')
        fetch_video(elems[1], '/data/davidchan/kinetics/train/{}.mp4'.format(elems[1]), int(elems[2]), int(elems[3]), None, None)
    except Exception as ex:
        print(ex, row)

rows = []
with open(sys.argv[1], 'r') as csv_file:
    for line in csv_file:
        rows.append(line)

with Pool(64) as p:
    for elem in tqdm.tqdm(p.imap_unordered(fetch_row, rows[1:]), total=len(rows)):
        pass
    

