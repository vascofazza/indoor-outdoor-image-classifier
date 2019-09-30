# indoor-outdoor-image-classifier
A CNN-based i​mage classifier​ capable of detecting if a scene is ​indoors or outdoors.

This repository contains scripts for downloading videos corresponding to a particular or a few categories of [youtube-8m dataset](https://research.google.com/youtube8m/index.html).

##Dependencies

Dependencies for downloading youtube videos from ids

* [youtube-dl](https://github.com/rg3/youtube-dl#installation)

Dependencies for generation of frames from videos

* [ffmpeg](https://www.ffmpeg.org/download.html)

Python dependencies for the classification task

* Numpy
* Sklearn
* Tensorflow
* Keras
* tqdm
* PyYaml
* pytest

## Install Python dependencies

```
 $ pip3 install -r requirements.txt
```

## Test pretrained model

An example dataset composed by 600 YouTube videos, for a total of 60.000 video frames, is availale [here](http://insidecode.it/indoor-outdoor-data_64.zip).
The videos belong to different category, according to the [youtube-8m dataset](https://research.google.com/youtube8m/index.html).

The dataset contains instances from the following categories, labeled as follows:
```
Living_room		indoor
Bedroom			indoor
Dining_room		indoor
Garden			outdoor
Outdoor_recreation	outdoor
Hiking			outdoor
```

### Test the model on a single image
This CLI will run the pretrained model on a provided image.

```
bash classify.sh config/train_params.yml data/test/indoor_test.jpg
```

### Download and extract the train/test data
Prepare the test environment downloading the provided dataset excerpt.

```
$ cd indoor-outdoor-image-classifier
$ wget http://insidecode.it/indoor-outdoor-data_64.zip
$ unsip indoor-outdoor-data_64.zip
```

### Evaluate the model on the test split
This will test the performance of the model on the test split and run a simple unit test on two benchmark images.

```
$ bash evaluate.sh config/train_params.yml indoor-outdoor-data_64/frames
```

## Train model

It is possible to tune the model parameters defining a new configuration file (following the default one in config/train_params.yml) and train the new model using the following command:

```
bash train.sh <config-file> <image-directory-path>"
```
