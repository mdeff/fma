#!/usr/bin/env python3

# FMA: A Dataset For Music Analysis
# Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson, EPFL LTS2.

import os
import shutil
import pickle
import zipfile
import subprocess as sp
import multiprocessing
from datetime import datetime
import argparse
from functools import partial

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


def download_metadata(args):

    fma = utils.FreeMusicArchive(os.environ.get('FMA_KEY'))

    if args.tid_max is None:
        args.tid_max = int(fma.get_recent_tracks()[0][0])

    message = 'Collecting metadata from track ID {} to {}.'
    print(message.format(args.tid_min, args.tid_max))

    not_found = {}

    id_range = trange(args.tid_min, args.tid_max, desc='tracks')
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
        if dataset != 'genres':
            print('{}: {} collected, {} not found'.format(
                dataset, len(eval(dataset)), len(not_found[dataset])))

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


def download_data(args):

    dst_dir = os.path.join(os.path.abspath(args.path), 'fma_full')
    tracks = pd.read_csv('raw_tracks.csv', index_col=0)
    _create_subdirs(dst_dir, tracks)

    fma = utils.FreeMusicArchive(os.environ.get('FMA_KEY'))
    not_found = pickle.load(open('not_found.pickle', 'rb'))
    not_found['audio'] = []

    # Download missing tracks.
    collected = 0
    for tid in tqdm(tracks.index):
        dst = utils.get_audio_path(dst_dir, tid)
        if not os.path.exists(dst):
            try:
                fma.download_track(tracks.at[tid, 'track_file'], dst)
                collected += 1
            except:  # requests.HTTPError
                not_found['audio'].append(tid)

    pickle.dump(not_found, open('not_found.pickle', 'wb'))

    existing = len(tracks) - collected - len(not_found['audio'])
    print('audio: {} collected, {} existing, {} not found'.format(
        collected, existing, len(not_found['audio'])))


def _extract_metadata(tid, path):
    """Extract metadata from one audio file."""

    metadata = pd.Series(name=tid)

    try:
        path = utils.get_audio_path(path, tid)
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
        metadata['mode'] = 'ERROR'
        metadata['channels'] = 0
        metadata['sample_rate'] = 0
        metadata['samples'] = 0

    return metadata


def extract_mp3_metadata(args):
    """
    Fill metadata about the audio, e.g. the bit and sample rates.

    It extracts metadata from the mp3 and creates an mp3_metadata.csv table.
    """

    # More than usable CPUs to be CPU bound, not I/O bound. Beware memory.
    nb_workers = int(1.5 * len(os.sched_getaffinity(0)))
    print('Working with {} processes.'.format(nb_workers))

    path = os.path.join(args.path, 'fma_full')
    tids = utils.get_tids_from_directory(path)

    metadata = pd.DataFrame(index=tids)
    metadata.index.name = 'track_id'
    # Prevent the columns of being of type float because of NaNs.
    metadata['channels'] = 0
    metadata['mode'] = 'UNKNOWN'
    metadata['bit_rate'] = 0
    metadata['sample_rate'] = 0
    metadata['samples'] = 0

    pool = multiprocessing.Pool(nb_workers)
    extract = partial(_extract_metadata, path=path)
    it = pool.imap_unordered(extract, tids)

    for row in tqdm(it, total=len(tids)):
        metadata.loc[row.name] = row

    not_found = pickle.load(open('not_found.pickle', 'rb'))
    tids = list(metadata[metadata['mode'] == 'ERROR'].index)
    not_found['mp3_metadata'] = tids
    pickle.dump(not_found, open('not_found.pickle', 'wb'))

    metadata.drop(tids, inplace=True)
    metadata.sort_index(axis=0, inplace=True)
    metadata.sort_index(axis=1, inplace=True)
    metadata.to_csv('mp3_metadata.csv')


def trim_audio(args):

    path = os.path.abspath(args.path)
    fma_full = os.path.join(path, 'fma_full')
    fma_large = os.path.join(path, 'fma_large')
    tracks = pd.read_csv('mp3_metadata.csv', index_col=0)
    _create_subdirs(fma_large, tracks)

    not_found = pickle.load(open('not_found.pickle', 'rb'))
    not_found['clips'] = []

    for tid, track in tqdm(tracks.iterrows(), total=len(tracks)):
        duration = track['samples'] / track['sample_rate']
        src = utils.get_audio_path(fma_full, tid)
        dst = utils.get_audio_path(fma_large, tid)
        if os.path.exists(dst):
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


def normalize_permissions_times(args):
    path = os.path.abspath(args.path)
    for dirpath, dirnames, filenames in tqdm(os.walk(path)):
        for name in filenames:
            dst = os.path.join(dirpath, name)
            os.chmod(dst, 0o444)
            os.utime(dst, (TIME, TIME))
        for name in dirnames:
            dst = os.path.join(dirpath, name)
            os.chmod(dst, 0o555)
            os.utime(dst, (TIME, TIME))


def create_zips(args):

    def get_filepaths(subset):
        filepaths = []
        tids = tracks.index[tracks['set', 'subset'] <= subset]
        for tid in tids:
            filepaths.append(utils.get_audio_path('', tid))
        return filepaths

    def get_checksums(base_dir, filepaths):
        """Checksums are assumed to be stored in order for efficiency."""
        checksums = []
        with open(os.path.join(args.path, base_dir, 'checksums')) as f:
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

        zip_filepath = os.path.join(args.path, zip_filename)
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
                src = os.path.join(args.path, base_dir, filepath)
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
    desc = 'Collect and process data to create the Free Music Archive (FMA) dataset.'
    parser = argparse.ArgumentParser(description=desc)
    subparsers = parser.add_subparsers(title='subcommands')

    path = 'Path to the folder where the audio of the FMA subsets is stored.'

    desc = ('Query the API of the FMA and store the collected metadata in '
            'raw_tracks.csv, raw_albums.csv, raw_artists.csv, and '
            'raw_genres.csv. The files are created in the current directory.')
    subparser = subparsers.add_parser('metadata', description=desc)
    subparser.add_argument('--min', dest='tid_min', type=int, default=0,
                           help='smallest track ID to consider')
    subparser.add_argument('--max', dest='tid_max', type=int, default=None,
                           help='largest track ID to consider')
    subparser.set_defaults(func=download_metadata)

    desc = 'Download the mp3 audio of each track.'
    subparser = subparsers.add_parser('data', description=desc)
    subparser.add_argument('path', type=str, help=path)
    subparser.set_defaults(func=download_data)

    desc = 'Extract technical metadata, such as duration, from the audio.'
    subparser = subparsers.add_parser('mp3_metadata', description=desc)
    subparser.add_argument('path', type=str, help=path)
    subparser.set_defaults(func=extract_mp3_metadata)

    desc = 'Extract 30s clips from the downloaded full-length audio.'
    subparser = subparsers.add_parser('clips', description=desc)
    subparser.add_argument('path', type=str, help=path)
    subparser.set_defaults(func=trim_audio)

    desc = 'Normalize the file permissions and times.'
    subparser = subparsers.add_parser('normalize', description=desc)
    subparser.add_argument('path', type=str, help=path)
    subparser.set_defaults(func=normalize_permissions_times)

    desc = 'Create the datasets as ZIP archives.'
    subparser = subparsers.add_parser('zips', description=desc)
    subparser.add_argument('path', type=str, help=path)
    subparser.set_defaults(func=create_zips)

    args = parser.parse_args()
    args.func(args)
