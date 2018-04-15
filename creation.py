#!/usr/bin/env python3

# FMA: A Dataset For Music Analysis
# Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson, EPFL LTS2.

import os
import sys
import shutil
import pickle
import zipfile
import subprocess as sp
from datetime import datetime
from tqdm import tqdm, trange
import pandas as pd
import utils


TIME = datetime(2017, 4, 1).timestamp()

README = """This .zip archive is part of the FMA, a dataset for music analysis.
Code & data: https://github.com/mdeff/fma
Paper: https://arxiv.org/abs/1612.01840

Each .mp3 is licensed by its artist.

The content's integrity can be verified with sha1sum -c checksums.
"""


def download_metadata():

    fma = utils.FreeMusicArchive(os.environ.get('FMA_KEY'))

    max_tid = int(fma.get_recent_tracks()[0][0])
    print('Largest track id: {}'.format(max_tid))

    not_found = {}

    id_range = trange(max_tid, desc='tracks')
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


def create_zips(dst_dir):

    def get_filepaths(subset):
        filepaths = []
        tids = tracks.index[tracks['set', 'subset'] <= subset]
        for tid in tids:
            filepaths.append(utils.get_audio_path('', tid))
        return filepaths

    def get_checksums(base_dir, filepaths):
        """Checksums are assumed to be stored in order for efficiency."""
        checksums = []
        with open(os.path.join(dst_dir, base_dir, 'checksums')) as f:
            for filepath in filepaths:
                exist = False
                for line in f:
                    if filepath == line[42:-1]:
                        exist = True
                        break
                if not exist:
                    raise ValueError('checksum not found: {}'.format(filepath))
                checksums.append(line)
        return checksums

    def create_zip(zip_filename, base_dir, filepaths):

        # Audio: all compressions are the same.
        # CSV: stored > deflated > BZIP2 > LZMA.
        # LZMA is close to BZIP2 and too recent to be widely available (unzip).
        compression = zipfile.ZIP_BZIP2

        zip_filepath = os.path.join(dst_dir, zip_filename)
        with zipfile.ZipFile(zip_filepath, 'x', compression) as zf:

            def info(name):
                name = os.path.join(zip_filename[:-4], name)
                info = zipfile.ZipInfo(name, (2017, 4, 1, 0, 0, 0))
                info.external_attr = 0o444 << 16 | 0o2 << 30
                return info

            zf.writestr(info('README.txt'), README, compression)

            checksums = get_checksums(base_dir, filepaths)
            zf.writestr(info('checksums'), ''.join(checksums), compression)

            for filepath in tqdm(filepaths):
                src = os.path.join(dst_dir, base_dir, filepath)
                dst = os.path.join(zip_filename[:-4], filepath)
                zf.write(src, dst)

        os.chmod(zip_filepath, 0o444)
        os.utime(zip_filepath, (TIME, TIME))

    METADATA = [
        'not_found.pickle',
        'raw_genres.csv', 'raw_albums.csv',
        'raw_artists.csv', 'raw_tracks.csv',
        'tracks.csv', 'genres.csv',
        'raw_echonest.csv', 'echonest.csv', 'features.csv',
    ]
    create_zip('fma_metadata.zip', 'fma_metadata', METADATA)

    tracks = utils.load('tracks.csv')
    create_zip('fma_small.zip', 'fma_large', get_filepaths('small'))
    create_zip('fma_medium.zip', 'fma_large', get_filepaths('medium'))
    create_zip('fma_large.zip', 'fma_large', get_filepaths('large'))
    create_zip('fma_full.zip', 'fma_full', get_filepaths('large'))


if __name__ == "__main__":
    if sys.argv[1] == 'metadata':
        download_metadata()
    elif sys.argv[1] == 'data':
        download_data(sys.argv[2])
    elif sys.argv[1] == 'clips':
        trim_audio(sys.argv[2])
    elif sys.argv[1] == 'normalize':
        normalize_permissions_times(sys.argv[2])
    elif sys.argv[1] == 'zips':
        create_zips(sys.argv[2])
