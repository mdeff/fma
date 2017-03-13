# FMA: A Dataset For Music Analysis

[Kirell Benzi](http://kirellbenzi.com/), [Michaël Defferrard](http://deff.ch),
[Pierre Vandergheynst](https://people.epfl.ch/pierre.vandergheynst),
[Xavier Bresson](http://research.ntu.edu.sg/expertise/academicprofile/Pages/StaffProfile.aspx?ST_EMAILID=XBRESSON),
[EPFL LTS2](https://lts2.epfl.ch).

[paper]: https://arxiv.org/abs/1612.01840>
[FMA]:   https://freemusicarchive

Note that this is a **beta release** and that this repository as well as the
paper and data are subject to change. Stay tuned!

## Data

The dataset is a dump of the [Free Music Archive (FMA)][FMA]. Please see the
[paper] for a description of how the data was collected and cleaned as well as
an analysis and some baselines.

You got various sizes of MP3-encoded audio data:

1. [fma_small.zip]: 4,000 tracks of 30 seconds, 10 balanced genres (GTZAN-like)
   (~3.4 GiB)
2. [fma_medium.zip]: 14,511 tracks of 30 seconds, 20 unbalanced genres
   (~12.2 GiB)
3. [fma_large.zip]: 77,643 tracks of 30 seconds, 68 unbalanced genres (~90 GiB)
   (available soon)
4. [fma_huge.zip]: 77,643 untrimmed tracks, 68 unbalanced genres (~900 GiB)
   (subject to distribution constraints)

[fma_small.zip]:  https://os.unil.cloud.switch.ch/fma/fma_small.zip
[fma_medium.zip]: https://os.unil.cloud.switch.ch/fma/fma_medium.zip

As meta-data, you got the following in this repository:
* `tracks.json`: a table (to be imported as a [pandas dataframe]) which
  contains meta-data about each track such as the ID, the title, the artist or
  the genres. See the [usage] notebook for an exhaustive list.
* `genres.json`: all the xxx available genres, used to infer the genre
  hierarchy and top-level genres.
* `features.json`: common features extracted with [librosa].
* `spotify.json`: audio features provided by [Spotify], formerly [Echonest].
  Cover all tracks distributed in [fma_small.zip] and [fma_medium.zip] as well
  as some others.

[pandas dataframe]: http://pandas.pydata.org/
[librosa]:  https://librosa.github.io/librosa/
[spotify]:  https://www.spotify.com/
[echonest]: http://the.echonest.com/

## Code

As a user of the dataset, you're probably most interested by those notebooks:

1. [usage]: how to load the datasets and train your own models.
2. [webapi]: query the web API of the [FMA] to update the dataset or gather
   further information about tracks, albums or artists.

If you're curious you may check those notebooks, which most results appear in
the paper:

1. [analysis]: some exploration of the data.
2. [baselines]: baseline models for various tasks.

For the most curious, these were used to create the dataset:

1. [generation]: generation of the dataset.
2. [features]: generation of features from the raw audio, i.e. `features.json`.

[usage]:      https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/usage.ipynb
[webapi]:     https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/webapi.ipynb
[analysis]:   https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/analysis.ipynb
[baselines]:  https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/baselines.ipynb
[generation]: https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/fma_generation.ipynb
[features]:   https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/features.ipynb

## Installation

1. Optionally, use [pyenv] to install Python 3.6 and create a [virtual
   environment][pyenv-virt].
	```sh
	pyenv install 3.6.0
	pyenv virtualenv 3.6.0 fma
	pyenv activate fma
	```

2. Clone the repository.
	```sh
	git clone https://github.com/mdeff/fma.git
	cd fma
	```

3. Install the Python dependencies from `requirements.txt`. Depending on your
   usage, you may need to install [ffmpeg] or [graphviz].
	```sh
	make install
	```

4. Optionnaly, install [CUDA] to train neural networks on GPUs. See
   [Tensorflow's instructions](https://www.tensorflow.org/install/).

5. Fill in the configuration.
	```sh
	cat .env
	DATA_DIR=/path/to/fma_small
	```

5. Open Jupyter or run a notebook.
	```sh
	jupyter-notebook
	make fma_baselines.ipynb
	```

[pyenv]:      https://github.com/pyenv/pyenv
[pyenv-virt]: https://github.com/pyenv/pyenv-virtualenv
[ffmpeg]:     https://ffmpeg.org/download.html
[graphviz]:   http://www.graphviz.org/
[CUDA]:       https://en.wikipedia.org/wiki/CUDA

## License & co

* Please cite our [paper] if you use our code or data.
* The code is released under the terms of the [MIT license](LICENSE.txt).
* The dataset is meant for research purposes only.
* We are grateful to SWITCH and EPFL for hosting the dataset within the context
  of the [SCALE-UP] project, funded in part by the swissuniversities [SUC P-2
  program].

[SCALE-UP]: https://projects.switch.ch/scale-up/
[SUC P-2 program]: https://www.swissuniversities.ch/isci
