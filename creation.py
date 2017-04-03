#!/bin/env python3

import os
import sys
import pickle
import subprocess as sp
from datetime import datetime
from tqdm import tqdm, trange
import pandas as pd
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
        eval(dataset).to_csv('raw_' + dataset + '.csv')

    pickle.dump(not_found, open('not_found.pickle', 'wb'))


def _create_subdirs(dst_dir, tracks):

    # Get write access.
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    os.chmod(dst_dir, 0o777)

    # Create writable sub-directories.
    n_folders = max(tracks.index) // 1000 + 1
    for folder in range(n_folders):
        dst = os.path.join(dst_dir, '{:03d}'.format(folder))
        if not os.path.exists(dst):
            os.makedirs(dst)
        os.chmod(dst, 0o777)


def download_data(dst_dir):

    dst_dir = os.path.abspath(dst_dir)
    tracks = pd.read_csv('raw_tracks.csv', index_col=0)
    _create_subdirs(dst_dir, tracks)

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


def trim_audio(dst_dir):

    dst_dir = os.path.abspath(dst_dir)
    fma_full = os.path.join(dst_dir, 'fma_full')
    fma_large = os.path.join(dst_dir, 'fma_large')
    tracks = pd.read_csv('tracks.csv', index_col=0, header=[0, 1])
    _create_subdirs(fma_large, tracks)

    path_in = utils.build_path(fma_full)
    path_out = utils.build_path(fma_large)
    # Todo: should use the fma_full subset (no need to check duration).
    for tid in tqdm(tracks.index):
        duration = tracks.loc[tid, ('track', 'duration')]
        if not os.path.exists(path_out(tid)) and duration > 30:
            start = duration // 2 - 15
            command = ['ffmpeg', '-i', path_in(tid),
                       '-ss', str(start), '-t', '30',
                       '-acodec', 'copy', path_out(tid)]
            sp.run(command, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)


def normalize_permissions_times(dst_dir):
    TIME = datetime(2017, 4, 1).timestamp()
    dst_dir = os.path.abspath(dst_dir)
    for dirpath, dirnames, filenames in tqdm(os.walk(dst_dir)):
        for name in filenames:
            dst = os.path.join(dirpath, name)
            os.chmod(dst, 0o444)
            os.utime(dst, (TIME, TIME))
        for name in dirnames:
            dst = os.path.join(dirpath, name)
            os.chmod(dst, 0o555)
            os.utime(dst, (TIME, TIME))


if __name__ == "__main__":
    if sys.argv[1] == 'metadata':
        download_metadata()
    elif sys.argv[1] == 'data':
        download_data(sys.argv[2])
    elif sys.argv[1] == 'clips':
        trim_audio(sys.argv[2])
    elif sys.argv[1] == 'normalize':
        normalize_permissions_times(sys.argv[2])
