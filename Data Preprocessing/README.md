### Dataset preprocessing
These codes are used to preprocess the data before training or testing.

> csv-downsampler.py :
GIANT dataset has 1B records of annotated citation strings. There are 219 csv(s) in GIANT with 45M records each. This file is used to randomly choose n samples from each csv and create a downsmapled dataset.

> csv-downsampler.sh :
Bash script to take advantage of HPC for downsampling task.

> dataset-builder.ipynb :
Contains tokenizer that converts an annotated citation string into a list of word-label tuple. It provides text files that contains word-label tuple on each line. Difference between two sentense is denoted with an empty line.

> concat-text-files.ipynb :
This code takes all the text files(that are created from *dataset-builder.ipnyb*) from a directory and merge them into one.


