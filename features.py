#!/usr/bin/env python3

# FMA: A Dataset For Music Analysis
# Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson, EPFL LTS2.

# All features are extracted using [librosa](https://github.com/librosa/librosa).
# Alternatives:
# * [MARSYAS](https://github.com/marsyas/marsyas) (C++ with Python bindings)
# * [RP extract](http://www.ifs.tuwien.ac.at/mir/downloads.html) (Matlab, Java, Python)
# * [jMIR jAudio](http://jmir.sourceforge.net) (Java)
# * [MIRtoolbox](https://www.jyu.fi/hum/laitokset/musiikki/en/research/coe/materials/mirtoolbox) (Matlab)

import os
import multiprocessing
import numpy as np
from scipy import stats
import pandas as pd
import librosa
from tqdm import tqdm
import utils


def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rmse=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


def compute_features(tid):

    features = pd.Series(index=columns(), dtype=np.float32, name=tid)

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    filepath = utils.get_audio_path(os.environ.get('AUDIO_DIR'), tid)
    x, sr = librosa.load(filepath, sr=None, mono=True)  # if sr --> kaiser_fast
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
                             n_bins=7*12, tuning=None))
    assert stft.shape[1] == cqt.shape[1] == int(np.floor(len(x) / 512 + 1))
    assert stft.shape[0] == 1 + 2048 // 2
    assert cqt.shape[0] == 7 * 12

    c = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
    feature_stats('chroma_stft', c)
    c = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cqt', c)
    c = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cens', c)
    t = librosa.feature.tonnetz(chroma=c)
    feature_stats('tonnetz', t)

    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
    m = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    feature_stats('mfcc', m)

    rmse = librosa.feature.rmse(S=stft)
    feature_stats('rmse', rmse)
    zcr = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
    feature_stats('zcr', zcr)

    s = librosa.feature.spectral_centroid(S=stft)
    feature_stats('spectral_centroid', s)
    s = librosa.feature.spectral_bandwidth(S=stft)
    feature_stats('spectral_bandwidth', s)
    s = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    feature_stats('spectral_contrast', s)
    s = librosa.feature.spectral_rolloff(S=stft)
    feature_stats('spectral_rolloff', s)

    return features


def main():
    tracks = utils.load('tracks.csv')

    # More than number of usable CPUs to be CPU bound, not I/O bound.
    nb_worker = 2 * len(os.sched_getaffinity(0))
    print('Working with {} processes.'.format(nb_worker))

    tids = tracks.index
    pool = multiprocessing.Pool(nb_worker)
    it = pool.imap_unordered(compute_features, tids)

    features = pd.DataFrame(index=tids, columns=columns(), dtype=np.float32)
    for row in tqdm(it, total=len(tids)):
        features.loc[row.name] = row

    NDIGITS = 10
    save(features, NDIGITS)
    test(features, NDIGITS)


def save(features, ndigits):

    # Should be done already, just to be sure.
    features.sort_index(axis=0, inplace=True)
    features.sort_index(axis=1, inplace=True)

    features.to_csv('features.csv', float_format='%.{}e'.format(ndigits))


def test(features, ndigits):

    assert not features.isnull().values.any()

    tmp = utils.load('features.csv')
    np.testing.assert_allclose(tmp.values, features.values, rtol=10**-ndigits)


if __name__ == "__main__":
    main()
