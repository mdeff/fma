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


def path(df, index):
    genre = df.iloc[index]['top_genre']
    tid = df.iloc[index].name
    return os.path.join(genre, str(tid) + '.mp3')
