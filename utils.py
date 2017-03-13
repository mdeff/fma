import dotenv
import numpy as np
import ctypes
import multiprocessing
import multiprocessing.sharedctypes as sharedctypes
import os.path

FMA_METADATA = ['artist', 'title', 'genres', 'play_count']
TOP_GENRE = 'top_genre'
SPLIT = 'train'

ECHONEST_METADATA = ['release', 'artist_location', 'artist_name', 'album_date',
                     'album_name']

ECHONEST_AUDIO_FEATURES = ['acousticness', 'danceability', 'energy',
                           'instrumentalness', 'liveness', 'speechiness',
                           'tempo', 'valence']
ECHONEST_TEMPORAL_FEATURES = 'temporal_echonest_features'

ECHONEST_SOCIAL_FEATURES = ['artist_discovery', 'artist_familiarity',
                            'artist_hotttnesss', 'song_hotttnesss',
                            'song_currency']
ECHONEST_RANKS = ['artist_discovery_rank', 'artist_familiarity_rank',
                  'artist_hotttnesss_rank', 'song_currency_rank',
                  'song_hotttnesss_rank']

# Number of samples per 30s audio clip.
# TODO: fix dataset to be constant.
NB_AUDIO_SAMPLES = 1321967
SAMPLING_RATE = 44100

# Load the environment from the .env file.
dotenv.load_dotenv(dotenv.find_dotenv())


def build_path(df, data_dir):
    def path(index):
        genre = df.iloc[index]['top_genre']
        tid = df.iloc[index].name
        return os.path.join(data_dir, genre, str(tid) + '.mp3')
    return path


class Loader:
    def load(path):
        raise NotImplemented()


class RawAudioLoader(Loader):
    def __init__(self, sampling_rate=SAMPLING_RATE):
        self.sampling_rate = sampling_rate
        self.shape = (NB_AUDIO_SAMPLES * sampling_rate // SAMPLING_RATE, )
    def load(self, filename):
        return self._load(filename)[:self.shape[0]]


class LibrosaLoader(RawAudioLoader):
    def _load(self, filename):
        import librosa
        sr = self.sampling_rate if self.sampling_rate != SAMPLING_RATE else None
        x, sr = librosa.load(filename, sr=sr)
        return x


class AudioreadLoader(RawAudioLoader):
    def _load(self, filename):
        import audioread
        a = audioread.audio_open(filename)
        a.read_data()


class PydubLoader(RawAudioLoader):
    def _load(self, filename):
        from pydub import AudioSegment
        song = AudioSegment.from_file(filename)
        song = song.set_channels(1)
        x = song.get_array_of_samples()
        # print(filename) if song.channels != 2 else None
        return np.array(x)


class FfmpegLoader(RawAudioLoader):
    def _load(self, filename):
        """Fastest and less CPU intensive loading method."""
        import subprocess as sp
        command = ['ffmpeg',
                   '-i', filename,
                   '-f', 's16le',
                   '-acodec', 'pcm_s16le',
                   '-ac', '1']  # channels: 2 for stereo, 1 for mono
        if self.sampling_rate != SAMPLING_RATE:
            command.extend(['-ar', str(self.sampling_rate)])
        command.append('-')
        # 30s at 44.1 kHz ~= 1.3e6
        proc = sp.run(command, stdout=sp.PIPE, bufsize=10**7)
        return np.fromstring(proc.stdout, dtype="int16")


def build_sample_loader(path, Y, loader):

    class SampleLoader:

        def __init__(self, ids, batch_size=4):
            self.lock1 = multiprocessing.Lock()
            self.lock2 = multiprocessing.Lock()
            self.batch_foremost = sharedctypes.RawValue(ctypes.c_int, 0)
            self.batch_rearmost = sharedctypes.RawValue(ctypes.c_int, -1)
            self.condition = multiprocessing.Condition(lock=self.lock2)

            data = sharedctypes.RawArray(ctypes.c_int, ids.data)
            self.ids = np.ctypeslib.as_array(data)

            self.batch_size = batch_size
            self.loader = loader
            self.X = np.empty((self.batch_size, *loader.shape))
            self.Y = np.empty((self.batch_size, Y.shape[1]), dtype=np.int)

        def __iter__(self):
            return self

        def __next__(self):

            with self.lock1:
                if self.batch_foremost.value == 0:
                    np.random.shuffle(self.ids)

                batch_current = self.batch_foremost.value
                if self.batch_foremost.value + self.batch_size < self.ids.size:
                    batch_size = self.batch_size
                    self.batch_foremost.value += self.batch_size
                else:
                    batch_size = self.ids.size - self.batch_foremost.value
                    self.batch_foremost.value = 0

                # print(self.ids, self.batch_foremost.value, batch_current, self.ids[batch_current], batch_size)
                # print('queue', self.ids[batch_current], batch_size)
                indices = np.array(self.ids[batch_current:batch_current+batch_size])

            for i, idx in enumerate(indices):
                self.X[i] = self.loader.load(path(idx))
                self.Y[i] = Y[idx]

            with self.lock2:
                while (batch_current - self.batch_rearmost.value) % self.ids.size > self.batch_size:
                    # print('wait', indices[0], batch_current, self.batch_rearmost.value)
                    self.condition.wait()
                self.condition.notify_all()
                # print('yield', indices[0], batch_current, self.batch_rearmost.value)
                self.batch_rearmost.value = batch_current

                return self.X[:batch_size], self.Y[:batch_size]

    return SampleLoader
