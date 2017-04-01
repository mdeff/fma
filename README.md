# FMA: A Dataset For Music Analysis

[Michaël Defferrard](http://deff.ch), [Kirell Benzi](http://kirellbenzi.com/),
[Pierre Vandergheynst](https://people.epfl.ch/pierre.vandergheynst),
[Xavier Bresson](http://research.ntu.edu.sg/expertise/academicprofile/Pages/StaffProfile.aspx?ST_EMAILID=XBRESSON),
[EPFL LTS2](https://lts2.epfl.ch).

[paper]:     https://arxiv.org/abs/1612.01840
[FMA]:       https://freemusicarchive.org
[WFMU]:      https://wfmu.org
[Wikipedia]: https://en.wikipedia.org/wiki/Free_Music_Archive

Note that this is a **beta release** and that this repository as well as the
paper and data are subject to change. Stay tuned!

## Data

The dataset is a dump of the [Free Music Archive (FMA)][FMA], an interactive
library of high-quality, legal audio downloads. Please see our [paper] for
a description of how the data was collected and cleaned as well as an analysis
and some baselines.

You got various sizes of MP3-encoded audio data:

1. [fma_small.zip]: 4,000 tracks of 30 seconds, 10 balanced genres (GTZAN-like)
   (~3.4 GiB)
2. [fma_medium.zip]: 14,511 tracks of 30 seconds, 20 unbalanced genres
   (~12.2 GiB)
3. [fma_large.zip]: 77,643 tracks of 30 seconds, 68 unbalanced genres (~90 GiB)
   (available soon)
4. [fma_full.zip]: 77,643 untrimmed tracks, 164 unbalanced genres (~900 GiB)
   (subject to distribution constraints)

[fma_small.zip]:  https://os.unil.cloud.switch.ch/fma/fma_small.zip
[fma_medium.zip]: https://os.unil.cloud.switch.ch/fma/fma_medium.zip

All the below metadata and features are tables which can be imported as [pandas
dataframes][pandas], or used with any other data analysis tool. See the [paper]
or the [usage] notebook for an exhaustive description.

* [fma_metadata.zip]: all metadata for all tracks (~7 MiB)
	* `tracks.json`: per track metadata such as ID, title, artist, genres and
	  play counts, for all 110,000 tracks.
	* `genres.json`: all 164 genre IDs with their name and parent (used to
	  infer the genre hierarchy and top-level genres).
* [fma_features.zip]: all features for all tracks (~400 MiB)
	* `features.json`: common features extracted with [librosa].
	* `spotify.json`: audio features provided by [Spotify], formerly
	  [Echonest]. Cover all tracks distributed in `fma_small.zip` and
	  `fma_medium.zip` as well as some others.

[pandas]:   http://pandas.pydata.org/
[librosa]:  https://librosa.github.io/librosa/
[spotify]:  https://www.spotify.com/
[echonest]: http://the.echonest.com/

## Code

The following notebooks have been used to create and evaluate the dataset. They
should be useful to users.

1. [usage]: how to load the datasets and develop, train and test your own
   models with it.
2. [analysis]: some exploration of the metadata, data and features.
3. [baselines]: baseline models for genre recognition, both from audio and
   features.
4. [features]: features extraction from the audio (used to create
   `features.json`).
5. [webapi]: query the web API of the [FMA]. Can be used to update the dataset
   or gather further information.
6. [creation]: creation of the dataset (used to create `tracks.json` and
   `genres.json`).

[usage]:     https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/usage.ipynb
[analysis]:  https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/analysis.ipynb
[baselines]: https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/baselines.ipynb
[features]:  https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/features.ipynb
[webapi]:    https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/webapi.ipynb
[creation]:  https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/creation.ipynb

## Installation

1. Download some data and verify its integrity.
	```sh
	echo "e731a5d56a5625f7b7f770923ee32922374e2cbf  fma_small.zip" | sha1sum -c -
	echo "fe23d6f2a400821ed1271ded6bcd530b7a8ea551  fma_medium.zip" | sha1sum -c -
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
   want to train neural networks on GPUs. See
   [Tensorflow's instructions](https://www.tensorflow.org/install/).
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

* 2016-12-06 beta release
	* paper: [arXiv:1612.01840v1](https://arxiv.org/abs/1612.01840v1)
	* code: [git tag beta](https://github.com/mdeff/fma/releases/tag/beta)
	* `fma_small.zip`  sha1: `e731a5d56a5625f7b7f770923ee32922374e2cbf`
	* `fma_medium.zip` sha1: `fe23d6f2a400821ed1271ded6bcd530b7a8ea551`

## License & co

* Please cite our [paper] if you use our code or data.
* The code in this repository is released under the terms of the [MIT license](LICENSE.txt).
* The meta-data is released under the terms of the
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
