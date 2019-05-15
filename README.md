# FMA: A Dataset For Music Analysis

[Michaël Defferrard](http://deff.ch), [Kirell Benzi](http://kirellbenzi.com),
[Pierre Vandergheynst](https://people.epfl.ch/pierre.vandergheynst),
[Xavier Bresson](http://www.ntu.edu.sg/home/xbresson),
[EPFL LTS2](https://lts2.epfl.ch).

[paper]:     https://arxiv.org/abs/1612.01840
[FMA]:       https://freemusicarchive.org

The dataset is a dump of the [Free Music Archive (FMA)][FMA], an interactive
library of high-quality, legal audio downloads. Below the abstract from the
[paper].
> We introduce the Free Music Archive (FMA), an open and easily accessible
> dataset suitable for evaluating several tasks in MIR, a field concerned with
> browsing, searching, and organizing large music collections. The community's
> growing interest in feature and end-to-end learning is however restrained by
> the limited availability of large audio datasets. The FMA aims to overcome
> this hurdle by providing 917 GiB and 343 days of Creative Commons-licensed
> audio from 106,574 tracks from 16,341 artists and 14,854 albums, arranged in
> a hierarchical taxonomy of 161 genres. It provides full-length and
> high-quality audio, pre-computed features, together with track- and
> user-level metadata, tags, and free-form text such as biographies. We here
> describe the dataset and how it was created, propose a train/validation/test
> split and three subsets, discuss some suitable MIR tasks, and evaluate some
> baselines for genre recognition. Code, data, and usage examples are available
> at <https://github.com/mdeff/fma>.

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

## Usage

1. Download some data, verify its integrity, and uncompress the archives.
	```sh
	curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
	curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
	curl -O https://os.unil.cloud.switch.ch/fma/fma_medium.zip
	curl -O https://os.unil.cloud.switch.ch/fma/fma_large.zip
	curl -O https://os.unil.cloud.switch.ch/fma/fma_full.zip

	echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -
	echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip"    | sha1sum -c -
	echo "c67b69ea232021025fca9231fc1c7c1a063ab50b  fma_medium.zip"   | sha1sum -c -
	echo "497109f4dd721066b5ce5e5f250ec604dc78939e  fma_large.zip"    | sha1sum -c -
	echo "0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab  fma_full.zip"     | sha1sum -c -

	unzip fma_metadata.zip
	unzip fma_small.zip
	unzip fma_medium.zip
	unzip fma_large.zip
	unzip fma_full.zip
	```

	If you get any error while decompressing the archives (especially with the
	Windows and macOS system unzippers), please try [7zip]. That is probably an
	[unsupported compression issue](https://github.com/mdeff/fma/issues/5).

1. Optionally, use [pyenv] to install Python 3.6 and create a [virtual
   environment][pyenv-virt].
	```sh
	pyenv install 3.6.0
	pyenv virtualenv 3.6.0 fma
	pyenv activate fma
	```

1. Clone the repository.
	```sh
	git clone https://github.com/mdeff/fma.git
	cd fma
	```

1. Checkout the revision matching the data you downloaded (e.g., `beta`, `rc1`,
   `v1`). See the [history](#history) of the dataset.
	```sh
	git checkout rc1
	```

1. Install the Python dependencies from `requirements.txt`. Depending on your
   usage, you may need to install [ffmpeg] or [graphviz]. Install [CUDA] if you
   want to train neural networks on GPUs (see
   [Tensorflow's instructions](https://www.tensorflow.org/install/)).
	```sh
	make install
	```

1. Fill in the configuration.
	```sh
	cat .env
	AUDIO_DIR=/path/to/audio
	FMA_KEY=IFIUSETHEAPI
	```

1. Open Jupyter or run a notebook.
	```sh
	jupyter-notebook
	make baselines.ipynb
	```

[7zip]:       http://www.7-zip.org
[pyenv]:      https://github.com/pyenv/pyenv
[pyenv-virt]: https://github.com/pyenv/pyenv-virtualenv
[ffmpeg]:     https://ffmpeg.org/download.html
[graphviz]:   http://www.graphviz.org/
[CUDA]:       https://en.wikipedia.org/wiki/CUDA

## Coverage

* [Using CNNs and RNNs for Music Genre Recognition][tds2], Towards Data Science, 2018-12-13.
* [Over 1.5 TB’s of Labeled Audio Datasets][tds1], Towards Data Science, 2018-11-13.
* [Genre recognition challenge][crowdai_challenge] at the [web conference], Lyon, 2018-04.
* [25 Open Datasets for Deep Learning Every Data Scientist Must Work With][vidhya], Analytics Vidhya, 2018-03-29.
* [Slides][djd] presented at the [Data Jam days](http://datajamdays.org), Lausanne, 2017-11-24.
* [Poster][poster] presented at [ISMIR 2017](https://ismir2017.smcnus.org), China, 2017-10-24.
* [Slides][osip] for the [Open Science in Practice](https://osip2017.epfl.ch) summer school at EPFL, 2017-09-29.
* [A Music Information Retrieval Dataset, Made With FMA][fma2], freemusicarchive.org, 2017-05-22.
* [Pre-publication release announced][tw2], twitter.com, 2017-05-09.
* [FMA: A Dataset For Music Analysis][tfblog], tensorflow.blog, 2017-03-14.
* [Beta release discussed][tw1], twitter.com, 2017-02-08.
* [FMA Data Set for Researchers Released][fma1], freemusicarchive.org, 2016-12-15.

[tw1]:  https://twitter.com/YadFaeq/status/829406463286063104
[tw2]:  https://twitter.com/m_deff/status/861985446116589569
[fma1]: http://freemusicarchive.org/member/cheyenne_h/blog/FMA_Dataset_for_Researchers
[fma2]: http://freemusicarchive.org/member/cheyenne_h/blog/A_Music_Information_Retrieval_Dataset_Made_With_FMA
[tfblog]: https://tensorflow.blog/2017/03/14/fma-a-dataset-for-music-analysis
[osip]: https://doi.org/10.5281/zenodo.999353
[poster]: https://doi.org/10.5281/zenodo.1035847
[djd]: https://doi.org/10.5281/zenodo.1066119
[crowdai_challenge]: https://www.crowdai.org/challenges/www-2018-challenge-learning-to-recognize-musical-genre
[web conference]: https://www2018.thewebconf.org/program/challenges-track/
[vidhya]: https://www.analyticsvidhya.com/blog/2018/03/comprehensive-collection-deep-learning-datasets/
[tds1]: https://towardsdatascience.com/a-data-lakes-worth-of-audio-datasets-b45b88cd4ad
[tds2]: https://towardsdatascience.com/using-cnns-and-rnns-for-music-genre-recognition-2435fb2ed6af

Dataset lists
* <https://github.com/caesar0301/awesome-public-datasets>
* <https://archive.ics.uci.edu/ml/datasets/FMA:+A+Dataset+For+Music+Analysis>
* <http://deeplearning.net/datasets>
* <http://www.audiocontentanalysis.org/data-sets>
* <https://github.com/ismir/mir-datasets>
* <https://teachingmir.wikispaces.com/Datasets>
* <https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research>
* <https://cloudlab.atlassian.net/wiki/display/datasets/FMA:+A+Dataset+For+Music+Analysis>

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
  ```
  @inproceedings{fma_dataset,
    title = {FMA: A Dataset for Music Analysis},
    author = {Defferrard, Micha\"el and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
    booktitle = {18th International Society for Music Information Retrieval Conference},
    year = {2017},
    url = {https://arxiv.org/abs/1612.01840},
  }
  ```
* The code in this repository is released under the terms of the
  [MIT license](LICENSE.txt).
* The metadata is released under the terms of the
  [Creative Commons Attribution 4.0 International License (CC BY 4.0)][ccby40].
* We do not hold the copyright on the audio and distribute it under the terms
  of the license chosen by the artist.
* The dataset is meant for research purposes.
* We are grateful to the [Swiss Data Science Center] ([EPFL] and [ETH Zürich])
  for hosting the dataset.

[ccby40]: https://creativecommons.org/licenses/by/4.0
[Swiss Data Science Center]: https://datascience.ch/collaboration-and-partnerships
[EPFL]: http://www.epfl.ch
[ETH Zürich]: http://www.ethz.ch
