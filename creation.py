#!/usr/bin/env python3

# FMA: A Dataset For Music Analysis
# Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson, EPFL LTS2.

import os
import sys
import shutil
import pickle
import zipfile
import subprocess as sp
import multiprocessing
from datetime import datetime

from tqdm import tqdm, trange
import pandas as pd
import librosa
import mutagen

import utils


TIME = datetime(2017, 4, 1).timestamp()

README = """This .zip archive is part of the FMA, a dataset for music analysis.
Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson, EPFL LTS2.

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


def _extract_metadata(tid):
    """Extract metadata from one audio file."""

    metadata = pd.Series(name=tid)

    try:
        path = utils.get_audio_path(os.environ.get('AUDIO_DIR'), tid)
        f = mutagen.File(path)
        x, sr = librosa.load(path, sr=None, mono=False)
        assert f.info.channels == (x.shape[0] if x.ndim > 1 else 1)
        assert f.info.sample_rate == sr

        mode = {
            mutagen.mp3.BitrateMode.CBR: 'CBR',
            mutagen.mp3.BitrateMode.VBR: 'VBR',
            mutagen.mp3.BitrateMode.ABR: 'ABR',
            mutagen.mp3.BitrateMode.UNKNOWN: 'UNKNOWN',
        }

        metadata['bit_rate'] = f.info.bitrate
        metadata['mode'] = mode[f.info.bitrate_mode]
        metadata['channels'] = f.info.channels
        metadata['sample_rate'] = f.info.sample_rate
        metadata['samples'] = x.shape[-1]

    except Exception as e:
        print('{}: {}'.format(tid, repr(e)))
        metadata['bit_rate'] = 0
        metadata['mode'] = 'UNKNOWN'
        metadata['channels'] = 0
        metadata['sample_rate'] = 0
        metadata['samples'] = 0

    return metadata


def extract_mp3_metadata():
    """
    Fill metadata about the audio, e.g. the bit and sample rates.

    It extracts metadata from the mp3 and creates an mp3_metadata.csv table.
    """

    # More than usable CPUs to be CPU bound, not I/O bound. Beware memory.
    nb_workers = int(1.5 * len(os.sched_getaffinity(0)))
    print('Working with {} processes.'.format(nb_workers))

    tids = pd.read_csv('raw_tracks.csv', index_col=0).index

    metadata = pd.DataFrame(index=tids)
    # Prevent the columns of being of type float because of NaNs.
    metadata['channels'] = 0
    metadata['mode'] = 'UNKNOWN'
    metadata['bit_rate'] = 0
    metadata['sample_rate'] = 0
    metadata['samples'] = 0

    pool = multiprocessing.Pool(nb_workers)
    it = pool.imap_unordered(_extract_metadata, tids)

    for _, row in enumerate(tqdm(it, total=len(tids))):
        metadata.loc[row.name] = row

    metadata.to_csv('mp3_metadata.csv')


def trim_audio(dst_dir):

    dst_dir = os.path.abspath(dst_dir)
    fma_full = os.path.join(dst_dir, 'fma_full')
    fma_large = os.path.join(dst_dir, 'fma_large')
    tracks = pd.read_csv('mp3_metadata.csv', index_col=0)
    _create_subdirs(fma_large, tracks)

    not_found = pickle.load(open('not_found.pickle', 'rb'))
    not_found['clips'] = []

    for tid, track in tqdm(tracks.iterrows(), total=len(tracks)):
        duration = track['samples'] / track['sample_rate']
        src = utils.get_audio_path(fma_full, tid)
        dst = utils.get_audio_path(fma_large, tid)
        if tid in not_found['audio']:
            continue
        elif os.path.exists(dst):
            continue
        elif duration <= 30:
            shutil.copyfile(src, dst)
        else:
            start = int(duration // 2 - 15)
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
    elif sys.argv[1] == 'mp3_metadata':
        extract_mp3_metadata()
    elif sys.argv[1] == 'clips':
        trim_audio(sys.argv[2])
    elif sys.argv[1] == 'normalize':
        normalize_permissions_times(sys.argv[2])
    elif sys.argv[1] == 'zips':
        create_zips(sys.argv[2])
