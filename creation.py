#!/bin/env python3

import os
import sys
import shutil
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

    fma = utils.FreeMusicArchive(os.environ.get('FMA_KEY'))
    not_found = pickle.load(open('not_found.pickle', 'rb'))
    not_found['audio'] = []

    # Download missing tracks.
    for tid in tqdm(tracks.index):
        dst = utils.get_audio_path(dst_dir, tid)
        if not os.path.exists(dst):
            try:
                fma.download_track(tracks.at[tid, 'track_file'], dst)
            except:  # requests.HTTPError
                not_found['audio'].append(tid)

    pickle.dump(not_found, open('not_found.pickle', 'wb'))


def convert_duration(x):
    times = x.split(':')
    seconds = int(times[-1])
    minutes = int(times[-2])
    try:
        minutes += 60 * int(times[-3])
    except IndexError:
        pass
    return seconds + 60 * minutes


def trim_audio(dst_dir):

    dst_dir = os.path.abspath(dst_dir)
    fma_full = os.path.join(dst_dir, 'fma_full')
    fma_large = os.path.join(dst_dir, 'fma_large')
    tracks = pd.read_csv('raw_tracks.csv', index_col=0)
    _create_subdirs(fma_large, tracks)

    not_found = pickle.load(open('not_found.pickle', 'rb'))
    not_found['clips'] = []

    for tid in tqdm(tracks.index):
        duration = convert_duration(tracks.at[tid, 'track_duration'])
        src = utils.get_audio_path(fma_full, tid)
        dst = utils.get_audio_path(fma_large, tid)
        if tid in not_found['audio']:
            continue
        elif os.path.exists(dst):
            continue
        elif duration <= 30:
            shutil.copyfile(src, dst)
        else:
            start = duration // 2 - 15
            command = ['ffmpeg', '-i', src,
                       '-ss', str(start), '-t', '30',
                       '-acodec', 'copy', dst]
            try:
                sp.run(command, check=True, stderr=sp.DEVNULL)
            except sp.CalledProcessError:
                not_found['clips'].append(tid)

    for tid in not_found['clips']:
        try:
            os.remove(utils.get_audio_path(fma_large, tid))
        except FileNotFoundError:
            pass

    pickle.dump(not_found, open('not_found.pickle', 'wb'))


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
