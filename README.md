# FMA: A Dataset For Music Analysis

[Michaël Defferrard](https://deff.ch),
[Kirell Benzi](https://kirellbenzi.com),
[Pierre Vandergheynst](https://people.epfl.ch/pierre.vandergheynst),
[Xavier Bresson](https://www.ntu.edu.sg/home/xbresson). \
International Society for Music Information Retrieval Conference (ISMIR), 2017.

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

* Paper: [`arXiv:1612.01840`][paper] ([latex and reviews](https://github.com/mdeff/paper-fma-ismir2017))
* Slides: [`doi:10.5281/zenodo.1066119`](https://doi.org/10.5281/zenodo.1066119)
* Poster: [`doi:10.5281/zenodo.1035847`](https://doi.org/10.5281/zenodo.1035847)

[paper]: https://arxiv.org/abs/1612.01840
[FMA]: https://freemusicarchive.org

## Data

All metadata and features for all tracks are distributed in **[`fma_metadata.zip`]** (342 MiB).
The below tables can be used with [pandas] or any other data analysis tool.
See the [paper] or the [`usage.ipynb`] notebook for a description.
* `tracks.csv`: per track metadata such as ID, title, artist, genres, tags and play counts, for all 106,574 tracks.
* `genres.csv`: all 163 genres with name and parent (used to infer the genre hierarchy and top-level genres).
* `features.csv`: common features extracted with [librosa].
* `echonest.csv`: audio features provided by [Echonest] (now [Spotify]) for a subset of 13,129 tracks.

[pandas]:   https://pandas.pydata.org/
[librosa]:  https://librosa.org/
[spotify]:  https://www.spotify.com/
[echonest]: https://web.archive.org/web/20170519050040/http://the.echonest.com/

Then, you got various sizes of MP3-encoded audio data:

1. **[`fma_small.zip`]**: 8,000 tracks of 30s, 8 balanced genres (GTZAN-like) (7.2 GiB)
2. **[`fma_medium.zip`]**: 25,000 tracks of 30s, 16 unbalanced genres (22 GiB)
3. **[`fma_large.zip`]**: 106,574 tracks of 30s, 161 unbalanced genres (93 GiB)
4. **[`fma_full.zip`]**: 106,574 untrimmed tracks, 161 unbalanced genres (879 GiB)

[`fma_metadata.zip`]: https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
[`fma_small.zip`]:    https://os.unil.cloud.switch.ch/fma/fma_small.zip
[`fma_medium.zip`]:   https://os.unil.cloud.switch.ch/fma/fma_medium.zip
[`fma_large.zip`]:    https://os.unil.cloud.switch.ch/fma/fma_large.zip
[`fma_full.zip`]:     https://os.unil.cloud.switch.ch/fma/fma_full.zip

See the [wiki](https://github.com/mdeff/fma/wiki) (or [#41](https://github.com/mdeff/fma/issues/41)) for **known issues (errata)**.

## Code

The following notebooks, scripts, and modules have been developed for the dataset.

1. [`usage.ipynb`]: shows how to load the datasets and develop, train, and test your own models with it.
2. [`analysis.ipynb`]: exploration of the metadata, data, and features.
   Creates the [figures](https://github.com/mdeff/fma/tree/outputs/figures) used in the paper.
3. [`baselines.ipynb`]: baseline models for genre recognition, both from audio and features.
4. [`features.py`]: features extraction from the audio (used to create `features.csv`).
5. [`webapi.ipynb`]: query the web API of the [FMA]. Can be used to update the dataset.
6. [`creation.ipynb`]: creation of the dataset (used to create `tracks.csv` and `genres.csv`).
7. [`creation.py`]: creation of the dataset (long-running data collection and processing).
8. [`utils.py`]: helper functions and classes.

[`usage.ipynb`]:     https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/usage.ipynb
[`analysis.ipynb`]:  https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/analysis.ipynb
[`baselines.ipynb`]: https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/baselines.ipynb
[`features.py`]:     features.py
[`webapi.ipynb`]:    https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/webapi.ipynb
[`creation.ipynb`]:  https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/creation.ipynb
[`creation.py`]:     creation.py
[`utils.py`]:        utils.py

## Usage

[![Binder](https://static.mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mdeff/fma/outputs?urlpath=lab/tree/usage.ipynb)
&nbsp; Click the binder badge to play with the code and data from your browser without installing anything.

1. Clone the repository.
    ```sh
    git clone https://github.com/mdeff/fma.git
    cd fma
    ```

1. <details><summary>Create a Python 3.6 environment.</summary>

    ```sh
    # with https://conda.io
    conda create -n fma python=3.6
    conda activate fma

    # with https://github.com/pyenv/pyenv
    pyenv install 3.6.0
    pyenv virtualenv 3.6.0 fma
    pyenv activate fma

    # with https://pipenv.pypa.io
    pipenv --python 3.6
    pipenv shell

    # with https://docs.python.org/3/tutorial/venv.html
    python3.6 -m venv ./env
    source ./env/bin/activate
    ```
    </details>

1. Install dependencies.
    ```sh
    pip install --upgrade pip setuptools wheel
    pip install numpy==1.12.1  # workaround resampy's bogus setup.py
    pip install -r requirements.txt
    ```
    Note: you may need to install [ffmpeg](https://ffmpeg.org/download.html) or [graphviz](https://www.graphviz.org) depending on your usage.\
    Note: install [CUDA](https://en.wikipedia.org/wiki/CUDA) to train neural networks on GPUs (see [Tensorflow's instructions](https://www.tensorflow.org/install/)).

1. Download some data, verify its integrity, and uncompress the archives.
    ```sh
    cd data

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

    cd ..
    ```

    Note: try [7zip](https://www.7-zip.org) if decompression errors.
    It might be an [unsupported compression issue](https://github.com/mdeff/fma/issues/5).

1. Fill a `.env` configuration file (at repository's root) with the following content.
    ```
    AUDIO_DIR=./data/fma_small/  # the path to a decompressed fma_*.zip
    FMA_KEY=MYKEY  # only if you want to query the freemusicarchive.org API
    ```

1. Open Jupyter or run a notebook.
    ```sh
    jupyter notebook
    make usage.ipynb
    ```

## Impact, coverage, and resources

<details><summary>100+ research papers</summary>

Full list on [Google Scholar](https://scholar.google.com/scholar?cites=13646959466952873682,13785796238335741238,7544459641098681164,5736399534855095976).
Some picks below.

* [Zero-shot Learning for Audio-based Music Classification and Tagging](https://arxiv.org/abs/1907.02670)
* [One deep music representation to rule them all? A comparative analysis of different representation learning strategies](https://doi.org/10.1007/s00521-019-04076-1)
* [Deep Learning for Audio-Based Music Classification and Tagging: Teaching Computers to Distinguish Rock from Bach](https://sci-hub.tw/10.1109/MSP.2018.2874383)
* [Learning Discrete Structures for Graph Neural Networks](https://arxiv.org/abs/1903.11960)
* [A context encoder for audio inpainting](https://arxiv.org/abs/1810.12138)
* [OpenMIC-2018: An Open Data-set for Multiple Instrument Recognition](https://archives.ismir.net/ismir2018/paper/000248.pdf)
* [Detecting Music Genre Using Extreme Gradient Boosting](https://doi.org/10.1145/3184558.3191822)
* [Transfer Learning of Artist Group Factors to Musical Genre Classification](https://doi.org/10.1145/3184558.3191823)
* [Learning to Recognize Musical Genre from Audio: Challenge Overview](https://arxiv.org/abs/1803.05337)
* [Representation Learning of Music Using Artist Labels](https://arxiv.org/abs/1710.06648)

</details>

<details><summary>2 derived works</summary>

* [OpenMIC-2018: An Open Data-set for Multiple Instrument Recognition](https://github.com/cosmir/openmic-2018)
* [ConvNet features](https://github.com/keunwoochoi/FMA_convnet_features) from [Transfer learning for music classification and regression tasks](https://arxiv.org/abs/1703.09179)

</details>

<details><summary>~10 posts</summary>

* [Music Genre Classification With TensorFlow](https://towardsdatascience.com/music-genre-classification-with-tensorflow-3de38f0d4dbb), Towards Data Science, 2020-08-11.
* [Music Genre Classification: Transformers vs Recurrent Neural Networks](https://towardsdatascience.com/music-genre-classification-transformers-vs-recurrent-neural-networks-631751a71c58), Towards Data Science, 2020-06-14.
* [Using CNNs and RNNs for Music Genre Recognition](https://towardsdatascience.com/using-cnns-and-rnns-for-music-genre-recognition-2435fb2ed6af), Towards Data Science, 2018-12-13.
* [Over 1.5 TB’s of Labeled Audio Datasets](https://towardsdatascience.com/a-data-lakes-worth-of-audio-datasets-b45b88cd4ad), Towards Data Science, 2018-11-13.
* [Discovering Descriptive Music Genres Using K-Means Clustering](https://medium.com/latinxinai/discovering-descriptive-music-genres-using-k-means-clustering-d19bdea5e443), Medium, 2018-04-09.
* [25 Open Datasets for Deep Learning Every Data Scientist Must Work With](https://www.analyticsvidhya.com/blog/2018/03/comprehensive-collection-deep-learning-datasets/), Analytics Vidhya, 2018-03-29.
* [Learning Music Genres](https://medium.com/@diegoagher/learning-music-genres-5ab1cabadfed), Medium, 2017-12-13.
* [music2vec: Generating Vector Embeddings for Genre-Classification Task](https://medium.com/@rajatheb/music2vec-generating-vector-embedding-for-genre-classification-task-411187a20820), Medium, 2017-11-28.
* [A Music Information Retrieval Dataset, Made With FMA](https://web.archive.org/web/20190907182116/http://freemusicarchive.org/member/cheyenne_h/blog/A_Music_Information_Retrieval_Dataset_Made_With_FMA), freemusicarchive.org, 2017-05-22.
* [Pre-publication release announced](https://twitter.com/m_deff/status/861985446116589569), twitter.com, 2017-05-09.
* [FMA: A Dataset For Music Analysis](https://tensorflow.blog/2017/03/14/fma-a-dataset-for-music-analysis), tensorflow.blog, 2017-03-14.
* [Beta release discussed](https://twitter.com/YadFaeq/status/829406463286063104), twitter.com, 2017-02-08.
* [FMA Data Set for Researchers Released](https://web.archive.org/web/20190826112752/http://freemusicarchive.org/member/cheyenne_h/blog/FMA_Dataset_for_Researchers), freemusicarchive.org, 2016-12-15.

</details>

<details><summary>5 events</summary>

* [Summer Workshop](https://hcdigitalscholarship.github.io/audio-files) by the [Haverford Digital Scholarship Library](https://www.haverford.edu/library/digital-scholarship), 2020-07.
* [Genre recognition challenge](https://www.crowdai.org/challenges/www-2018-challenge-learning-to-recognize-musical-genre) at the [Web Conference](https://www2018.thewebconf.org/program/challenges-track/), Lyon, 2018-04.
* [Slides](https://doi.org/10.5281/zenodo.1066119) presented at the [Data Jam days](http://datajamdays.org), Lausanne, 2017-11-24.
* [Poster](https://doi.org/10.5281/zenodo.1035847) presented at [ISMIR 2017](https://ismir2017.ismir.net), Suzhou, 2017-10-24.
* [Slides](https://doi.org/10.5281/zenodo.999353) for the [Open Science in Practice](https://osip2017.epfl.ch) summer school at EPFL, 2017-09-29.

</details>

<details><summary>~10 dataset lists</summary>

* <https://github.com/caesar0301/awesome-public-datasets>
* <https://archive.ics.uci.edu/ml/datasets/FMA:+A+Dataset+For+Music+Analysis>
* <http://deeplearning.net/datasets>
* <http://www.audiocontentanalysis.org/data-sets>
* <https://github.com/ismir/mir-datasets>
* <https://teachingmir.wikispaces.com/Datasets>
* <https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research>
* <https://loc.gov/item/2018655052>
* <https://cloudlab.atlassian.net/wiki/display/datasets/FMA:+A+Dataset+For+Music+Analysis>
* <https://www.datasetlist.com>
* <https://data-flair.training/blogs/deep-learning-project-ideas>

</details>

## Contributing

Contribute by opening an [issue](https://github.com/mdeff/fma/issues) or a [pull request](https://github.com/mdeff/fma/pulls).
Let this repository be a hub around the dataset!

## History

**2017-05-09 pre-publication release**
* paper: [arXiv:1612.01840v2](https://arxiv.org/abs/1612.01840v2)
* code: [git tag rc1](https://github.com/mdeff/fma/releases/tag/rc1)
* `fma_metadata.zip` sha1: `f0df49ffe5f2a6008d7dc83c6915b31835dfe733`
* `fma_small.zip`    sha1: `ade154f733639d52e35e32f5593efe5be76c6d70`
* `fma_medium.zip`   sha1: `c67b69ea232021025fca9231fc1c7c1a063ab50b`
* `fma_large.zip`    sha1: `497109f4dd721066b5ce5e5f250ec604dc78939e`
* `fma_full.zip`     sha1: `0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab`
* known issues: see [#41](https://github.com/mdeff/fma/issues/41)

**2016-12-06 beta release**
* paper: [arXiv:1612.01840v1](https://arxiv.org/abs/1612.01840v1)
* code: [git tag beta](https://github.com/mdeff/fma/releases/tag/beta)
* `fma_small.zip`  sha1: `e731a5d56a5625f7b7f770923ee32922374e2cbf`
* `fma_medium.zip` sha1: `fe23d6f2a400821ed1271ded6bcd530b7a8ea551`

## Acknowledgments and Licenses

We are grateful to the [Swiss Data Science Center] ([EPFL] and [ETHZ]) for hosting the dataset.

Please cite our work if you use our code or data.

```
@inproceedings{fma_dataset,
  title = {{FMA}: A Dataset for Music Analysis},
  author = {Defferrard, Micha\"el and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle = {18th International Society for Music Information Retrieval Conference (ISMIR)},
  year = {2017},
  archiveprefix = {arXiv},
  eprint = {1612.01840},
  url = {https://arxiv.org/abs/1612.01840},
}
```

```
@inproceedings{fma_challenge,
  title = {Learning to Recognize Musical Genre from Audio},
  subtitle = {Challenge Overview},
  author = {Defferrard, Micha\"el and Mohanty, Sharada P. and Carroll, Sean F. and Salath\'e, Marcel},
  booktitle = {The 2018 Web Conference Companion},
  year = {2018},
  publisher = {ACM Press},
  isbn = {9781450356404},
  doi = {10.1145/3184558.3192310},
  archiveprefix = {arXiv},
  eprint = {1803.05337},
  url = {https://arxiv.org/abs/1803.05337},
}
```

* The code in this repository is released under the [MIT license](LICENSE.txt).
* The metadata is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)][ccby40].
* We do not hold the copyright on the audio and distribute it under the license chosen by the artist.
* The dataset is meant for research purposes.

[ccby40]: https://creativecommons.org/licenses/by/4.0
[Swiss Data Science Center]: https://datascience.ch/collaboration-and-partnerships
[EPFL]: https://www.epfl.ch
[ETHZ]: https://www.ethz.ch
