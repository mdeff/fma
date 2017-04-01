#!/bin/env python3

import os
import sys
import pickle
import pandas as pd
from tqdm import tqdm, trange
import utils


def download_metadata():

    fma = utils.FreeMusicArchive(os.environ.get('FMA_KEY'))

    max_tid = int(fma.get_recent_tracks()[0][0])
    print('Largest track id: {}'.format(max_tid))

    not_found = {}

    id_range = trange(20, desc='tracks')
    tracks, not_found['tracks'] = fma.get_all('track', id_range)

    id_range = tqdm(tracks['album_id'].unique(), desc='albums')
    albums, not_found['albums'] = fma.get_all('album', id_range)

    id_range = tqdm(tracks['artist_id'].unique(), desc='artists')
    artists, not_found['artists'] = fma.get_all('artist', id_range)

    genres = fma.get_all_genres()

    for dataset in 'tracks', 'albums', 'artists', 'genres':
        eval(dataset).sort_index(axis=0, inplace=True)
        eval(dataset).sort_index(axis=1, inplace=True)
        eval(dataset).to_csv(dataset + '_raw.csv')

    pickle.dump(not_found, open('not_found.pickle', 'wb'))


def download_data(dst_dir):

    # Get write access.
    dst_dir = os.path.abspath(dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    os.chmod(dst_dir, 0o777)

    # Create writable sub-directories.
    tracks = pd.read_csv('tracks_raw.csv', index_col=0)
    n_folders = max(tracks.index) // 1000 + 1
    for folder in range(n_folders):
        dst = os.path.join(dst_dir, '{:03d}'.format(folder))
        if not os.path.exists(dst):
            os.makedirs(dst)
        os.chmod(dst, 0o777)

    path = utils.build_path(dst_dir)
    fma = utils.FreeMusicArchive(os.environ.get('FMA_KEY'))
    not_found = pickle.load(open('not_found.pickle', 'rb'))
    not_found['audio'] = []

    # Download missing tracks.
    for tid in tqdm(tracks.index):
        dst = path(tid)
        if not os.path.exists(dst):
            try:
                fma.download_track(tracks.loc[tid, 'track_file'], dst)
            except:
                not_found['audio'].append(tid)

    pickle.dump(not_found, open('not_found.pickle', 'wb'))


if __name__ == "__main__":
    if sys.argv[1] == 'metadata':
        download_metadata()
    elif sys.argv[1] == 'data':
        download_data(sys.argv[2])
