# Data preparation

This repository includes the code to preprocess the data from the ["Talking With Hands" dataset from Facebook Research](https://github.com/facebookresearch/TalkingWithHands32M).

## Prerequesites

You need Python >= 3.7

1. setup access to a PostgreSQL database
2. clone the [Talking With Hands](https://github.com/facebookresearch/TalkingWithHands32M) repository and extract the data (the authors provide a script for that)
3. copy `settings.yaml.sample` to `settings.yaml` and add the path your local Talking With Hands repository and the database credentials
4. install python packages specified in the [Pipfile](Pipfile) – you can do this either manually with pip or using Pipenv:
   1. install pipenv: `pip install pipenv`
   2. install dependencies: `pipenv install`
   3. start virtualenv: `pipenv shell`

## Step 1 - create and fill database

```bash
python step_01_create_and_fill_db/main.py
```

This script parses the BVH files from the Talking With Hands repository and inserts the data into the PostgreSQL database. The script uses a modified version of [this BVH library](https://github.com/20tab/bvh-python).

## Step 2 - determine corrupted takes

```bash
python step_02_determine_corrupted_takes/main.py
```

In some takes the avatar just stands in the center and shows no movement, except for minor jittering. We have communicated this issue to the authors. This script computes a score based on the avatar's movement in each take, which is then used to determine whether a take is corrupted or not. The results are exported to `step_02_determine_corrupted_takes/corrupted_take_ids.json` so scripts in step 3 can use the information to exclude these takes.

## Step 3 - create datasets

```bash
python step_03_create_datasets/main.py
```

This script determines takes for training, validation and testing and dumps the data into the final HDF5 file `talking_with_hands_4_UAIXDLAMDS.hdf5` used for the machine learning part.

## Step 4 - visualization

```bash
python step_04_visualization/main.py
```
