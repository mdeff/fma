# FMA: A Dataset For Music Analysis

[Michaël Defferrard](http://deff.ch), [Kirell Benzi](http://kirellbenzi.com/),
[Pierre Vandergheynst](https://people.epfl.ch/pierre.vandergheynst),
[Xavier Bresson](http://research.ntu.edu.sg/expertise/academicprofile/Pages/StaffProfile.aspx?ST_EMAILID=XBRESSON),
[EPFL LTS2](https://lts2.epfl.ch).

[paper]:     https://arxiv.org/abs/1612.01840
[FMA]:       https://freemusicarchive.org

The dataset is a dump of the [Free Music Archive (FMA)][FMA], an interactive
library of high-quality, legal audio downloads. Below the abstract from the
[paper].
> We introduce the Free Music Archive (FMA), an open and easily accessible
> dataset which can be used to evaluate several tasks in music information
> retrieval (MIR), a field concerned with browsing, searching, and organizing
> large music collections. The community's growing interest in feature and
> end-to-end learning is however restrained by the limited availability of
> large audio datasets. By releasing the FMA, we hope to foster research which
> will improve the state-of-the-art and hopefully surpass the performance
> ceiling observed in e.g. genre recognition (MGR). The data is made of 106,574
> tracks, 16,341 artists, 14,854 albums, arranged in a hierarchical taxonomy of
> 161 genres, for a total of 343 days of audio and 917 GiB, all under
> permissive Creative Commons licenses. It features metadata like song title,
> album, artist and genres; user data like play counts, favorites, and
> comments; free-form text like description, biography, and tags; together with
> full-length, high-quality audio, and some pre-computed features. We propose
> a train/validation/test split and three subsets: a genre-balanced set of
> 8,000 tracks from 8 major genres, a genre-unbalanced set of 25,000 tracks
> from 16 genres, and a 98 GiB version with clips trimmed to 30s. This paper
> describes the dataset and how it was created, proposes some tasks like music
> classification and annotation or recommendation, and evaluates some baselines
> for MGR. Code, data, and usage examples are available at
> <https://github.com/mdeff/fma>.

This is a **pre-publication release**. As such, this repository as well as the
paper and data are subject to change. Stay tuned!

## Data

All metadata and features for all tracks are distributed in
**[fma_metadata.zip]** (342 MiB). The below tables can be used with [pandas] or
any other data analysis tool. See the [paper] or the [usage] notebook for
a description.
* `tracks.csv`: per track metadata such as ID, title, artist, genres, tags and
  play counts, for all 106,574 tracks.
* `genres.csv`: all 163 genre IDs with their name and parent (used to infer the
  genre hierarchy and top-level genres).
* `features.csv`: common features extracted with [librosa].
* `echonest.csv`: audio features provided by [Echonest] (now [Spotify]) for
  a subset of 13,129 tracks.

[pandas]:   http://pandas.pydata.org/
[librosa]:  https://librosa.github.io/librosa/
[spotify]:  https://www.spotify.com/
[echonest]: http://the.echonest.com/

Then, you got various sizes of MP3-encoded audio data:

1. **[fma_small.zip]**: 8,000 tracks of 30s, 8 balanced genres (GTZAN-like) (7.2 GiB)
2. **[fma_medium.zip]**: 25,000 tracks of 30s, 16 unbalanced genres (22 GiB)
3. **[fma_large.zip]**: 106,574 tracks of 30s, 161 unbalanced genres (93 GiB)
4. **[fma_full.zip]**: 106,574 untrimmed tracks, 161 unbalanced genres (879 GiB)

[fma_metadata.zip]: https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
[fma_small.zip]:    https://os.unil.cloud.switch.ch/fma/fma_small.zip
[fma_medium.zip]:   https://os.unil.cloud.switch.ch/fma/fma_medium.zip
[fma_large.zip]:    https://os.unil.cloud.switch.ch/fma/fma_large.zip
[fma_full.zip]:     https://os.unil.cloud.switch.ch/fma/fma_full.zip

## Code

The following notebooks and scripts, stored in this repository, have been
developed for the dataset.

1. [usage]: shows how to load the datasets and develop, train and test your own
   models with it.
2. [analysis]: exploration of the metadata, data and features.
3. [baselines]: baseline models for genre recognition, both from audio and
   features.
4. [features]: features extraction from the audio (used to create
   `features.csv`).
5. [webapi]: query the web API of the [FMA]. Can be used to update the dataset.
6. [creation]: creation of the dataset (used to create `tracks.csv` and
   `genres.csv`).

[usage]:     https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/usage.ipynb
[analysis]:  https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/analysis.ipynb
[baselines]: https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/baselines.ipynb
[features]:  features.py
[webapi]:    https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/webapi.ipynb
[creation]:  https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/creation.ipynb

## Installation

1. Download some data and verify its integrity.
	```sh
	echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -
	echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip"    | sha1sum -c -
	echo "c67b69ea232021025fca9231fc1c7c1a063ab50b  fma_medium.zip"   | sha1sum -c -
	echo "497109f4dd721066b5ce5e5f250ec604dc78939e  fma_large.zip"    | sha1sum -c -
	echo "0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab  fma_full.zip"     | sha1sum -c -
	```

2. Optionally, use [pyenv] to install Python 3.6 and create a [virtual
   environment][pyenv-virt].
	```sh
	pyenv install 3.6.0
	pyenv virtualenv 3.6.0 fma
	pyenv activate fma
	```

3. Clone the repository.
	```sh
	git clone https://github.com/mdeff/fma.git
	cd fma
	```

4. Install the Python dependencies from `requirements.txt`. Depending on your
   usage, you may need to install [ffmpeg] or [graphviz]. Install [CUDA] if you
   want to train neural networks on GPUs (see
   [Tensorflow's instructions](https://www.tensorflow.org/install/)).
	```sh
	make install
	```

5. Fill in the configuration.
	```sh
	cat .env
	AUDIO_DIR=/path/to/audio
	FMA_KEY=IFIUSETHEAPI
	```

6. Open Jupyter or run a notebook.
	```sh
	jupyter-notebook
	make baselines.ipynb
	```

[pyenv]:      https://github.com/pyenv/pyenv
[pyenv-virt]: https://github.com/pyenv/pyenv-virtualenv
[ffmpeg]:     https://ffmpeg.org/download.html
[graphviz]:   http://www.graphviz.org/
[CUDA]:       https://en.wikipedia.org/wiki/CUDA

## History

* 2017-05-09 pre-publication release
	* paper: [arXiv:1612.01840v2](https://arxiv.org/abs/1612.01840v2)
	* code: [git tag rc1](https://github.com/mdeff/fma/releases/tag/rc1)
	* `fma_metadata.zip` sha1: `f0df49ffe5f2a6008d7dc83c6915b31835dfe733`
	* `fma_small.zip`    sha1: `ade154f733639d52e35e32f5593efe5be76c6d70`
	* `fma_medium.zip`   sha1: `c67b69ea232021025fca9231fc1c7c1a063ab50b`
	* `fma_large.zip`    sha1: `497109f4dd721066b5ce5e5f250ec604dc78939e`
	* `fma_full.zip`     sha1: `0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab`

* 2016-12-06 beta release
	* paper: [arXiv:1612.01840v1](https://arxiv.org/abs/1612.01840v1)
	* code: [git tag beta](https://github.com/mdeff/fma/releases/tag/beta)
	* `fma_small.zip`  sha1: `e731a5d56a5625f7b7f770923ee32922374e2cbf`
	* `fma_medium.zip` sha1: `fe23d6f2a400821ed1271ded6bcd530b7a8ea551`

## Contributing

Please open an issue or a pull request if you want to contribute. Let's try to
keep this repository the central place around the dataset! Links to resources
related to the dataset are welcome. I hope the community will like it and that
we can keep it lively by evolving it toward people's needs.

## License & co

* Please cite our [paper] if you use our code or data.
* The code in this repository is released under the terms of the
  [MIT license](LICENSE.txt).
* The metadata is released under the terms of the
  [Creative Commons Attribution 4.0 International License (CC BY 4.0)][ccby40].
* We do not hold the copyright on the audio and distribute it under the terms
  of the license chosen by the artist.
* The dataset is meant for research purposes.
* We are grateful to SWITCH and EPFL for hosting the dataset within the context
  of the [SCALE-UP] project, funded in part by the swissuniversities [SUC P-2
  program].

[ccby40]: https://creativecommons.org/licenses/by/4.0
[SCALE-UP]: https://projects.switch.ch/scale-up/
[SUC P-2 program]: https://www.swissuniversities.ch/isci
