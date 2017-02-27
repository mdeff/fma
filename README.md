# [FMA: A Dataset For Music Analysis][paper]

[Kirell Benzi](http://kirellbenzi.com/), [Michaël Defferrard](http://deff.ch),
[Pierre Vandergheynst](https://people.epfl.ch/pierre.vandergheynst),
[Xavier Bresson](http://research.ntu.edu.sg/expertise/academicprofile/Pages/StaffProfile.aspx?ST_EMAILID=XBRESSON),
[EPFL LTS2](https://lts2.epfl.ch).

[paper]: https://arxiv.org/abs/1612.01840>

Note that this is a **beta release** and that this repository as well as the
paper and data are subject to change. Stay tuned!

## Data

The dataset is a dump of the [Free Music Archive](https://freemusicarchive.org/).
You got various sizes:

1. [Small](https://os.unil.cloud.switch.ch/fma/fma_small.zip): 4,000 clips of
   30 seconds, 10 balanced genres (GTZAN-like) (~3.4 GiB)
2. [Medium](https://os.unil.cloud.switch.ch/fma/fma_medium.zip): 14,511 clips
   of 30 seconds, 20 unbalanced genres (~12.2 GiB)
3. Large (available soon): 77,643 clips of 30 seconds, 68 unbalanced genres
   (~90 GiB)
4. Huge (subject to distribution constraints): 77,643 untrimmed clips, 68
   unbalanced genres (~900 GiB)

Notes:

* All datasets come with MP3 audio (128 kbps, 44.1 kHz, stereo) of all clips.
* All datasets come with the following meta-data about each clip: artist,
  title, list of genres (and top genre), play count.
* Meta-data about all clips are stored in a JSON file to be loaded as a
  [pandas dataframe](http://pandas.pydata.org/).
* As additional audio meta-data, each clip of datasets 1 and 2 come with all
  [Echonest features](http://the.echonest.com/).
* Please see the [paper] for a description of how the data was collected and
  cleaned.

## Code

This repository features the following notebooks:

1. [Generation]: generation of the datasets.
2. [Analysis]: loading and basic analysis of the data.
3. [Baselines]: baseline models for various tasks.
4. [Usage]: how to load the datasets and train your own models.

[generation]: https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/fma_generation.ipynb
[analysis]: https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/fma_analysis.ipynb
[baselines]: https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/fma_baselines.ipynb
[usage]: https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/fma_usage.ipynb

## License

* Please cite our [paper] if you use our code or data.
* The code is released under the terms of the [MIT license](LICENSE.txt).
* The dataset is meant for research only.
* We are grateful to SWITCH and EPFL for hosting the dataset within the context
  of the [SCALE-UP](https://projects.switch.ch/scale-up/) project, funded in
  part by the swissuniversities
  [SUC P-2 program](https://www.swissuniversities.ch/isci).
