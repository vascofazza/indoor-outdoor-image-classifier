import argparse
import os
from collections import Counter
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from src.util.utils import load_dataset_plain


def create_label_file(args):
    # extract the image class according to their YouTube category
    # file format: category\tlabel

    # mapping category -> label
    classes_mapping = {cat: label for cat, label in map(lambda x: x.rstrip().split("\t"), open(args.classmap))}

    # create the dataset TSV file: file_name -> label
    # the image folder structure is: ($category_video_name_folder)->(frames)->{set of frames.png}

    # reverse sort the categories to ease the file match process
    categories = sorted(classes_mapping.keys(), key=lambda x: len(x), reverse=True)

    label_counter = Counter()

    # map each frame with its label, output the result to a text file
    with open(os.path.join(args.out, 'all_labels.txt'), 'w') as f:
        for directory in tqdm(os.listdir(args.folder)):
            dir_path = os.path.join(args.folder, directory)
            if not os.path.isdir(dir_path):
                continue
            for cat in categories:
                if directory.startswith(cat):
                    label = classes_mapping[cat]
                    for frame, _, files in os.walk(dir_path):
                        if len(files) == 0:
                            continue
                        for img in files:
                            if not img.endswith(".png"):
                                continue
                            f.write(os.path.join(frame[len(args.folder)+1:], img) + "\t" + label + "\n")
                            label_counter[label] += 1
                    break

    print(label_counter)

def create_dataset(path, test_size):
    # Creating validation set
    X,Y = load_dataset_plain(path)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42, test_size=test_size, shuffle=False)
    with open(os.path.join(args.out, 'train_labels.txt'), 'w') as f:
        for instance, label in zip(X_train, y_train):
            f.write(instance+"\t"+label+"\n")

    with open(os.path.join(args.out, 'test_labels.txt'), 'w') as f:
        for instance, label in zip(X_test, y_test):
            f.write(instance+"\t"+label+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True, help='input folder containing images')
    parser.add_argument('--classmap', required=True, help='text file containing the class mapping')
    parser.add_argument('--out', required=True, help='output dataset file')
    parser.add_argument('--test_size', required=True, type=float, help='test split size (0.0 - 1.0)')
    args = parser.parse_args()

    create_label_file(args)
    create_dataset(os.path.join(args.out, 'all_labels.txt'), args.test_size)