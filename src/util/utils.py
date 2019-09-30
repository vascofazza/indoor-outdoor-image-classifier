import os
from keras.preprocessing import image
from tqdm import tqdm
import numpy as np

#Converts string labels ('indoor', 'outdoor') to integer indexes.
def to_numerical_label(labels, label_strings):
    label_strings = sorted(label_strings)
    mapping = {label:idx for idx, label in enumerate(label_strings)}
    return [mapping[x] for x in labels]

#Converts integer labels to string labels
def to_string_label(index, label_strings):
    label_strings = sorted(label_strings)
    return label_strings[index]

#Loads the dataset, outputs in text format
def load_dataset_plain(path):
    dataset_plain = [x.rstrip().split("\t") for x in tqdm(open(path), desc="loading dataset")]
    return zip(*dataset_plain)

#Loads an image from disk as a numpy array. Each feature is scaled between 0. and 1.
def load_image(path, resize=None):
    img = image.load_img(path, target_size=resize)
    vectorize = image.img_to_array(img)
    #normalizing
    vectorize = vectorize / 255
    return vectorize

#Loads the dataset, outputs in numpy array format.
def load_dataset_numpy(path, image_base_path, label_strings):
    paths, labels = load_dataset_plain(path)

    labels = to_numerical_label(labels, label_strings)

    images = list()
    #load images
    for path in tqdm(paths, desc="loading images"):
        path = os.path.join(image_base_path, path)
        images.append(load_image(path))
    return np.array(images), np.array(labels)

#Outputs the path of the last available checkpoint, None otherwise.
def get_last_checkpoint(dir_path):
    checkpoints_sorted = sorted(os.listdir(dir_path), key=lambda x: int(x.split(".")[0].split("_")[-1]), reverse=True)
    if len(checkpoints_sorted) > 0:
        return os.path.join(dir_path, checkpoints_sorted[0]), int(checkpoints_sorted[0].split(".")[0].split("_")[-1])
    return None, -1