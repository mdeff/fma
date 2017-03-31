import os
import pickle
from tqdm import tqdm, trange
import utils


def main():

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

    for dataset in 'tracks', 'albums', 'artists':
        eval(dataset).sort_index(axis=0, inplace=True)
        eval(dataset).sort_index(axis=1, inplace=True)
        eval(dataset).to_csv(dataset + '_raw.csv')

    pickle.dump(not_found, open('not_found.pickle', 'wb'))


if __name__ == "__main__":
    main()
