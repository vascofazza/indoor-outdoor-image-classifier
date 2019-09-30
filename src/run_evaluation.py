from src.util.config import load_configuration
from src.util.utils import get_last_checkpoint, load_dataset_numpy
from src.model import CNN_model
import argparse
import logging
import numpy as np

#Evaluates the model on the test dataset split.
def evaluate_testset(args):
    #Load the configuration file.
    config = load_configuration(args.config)
    #Initialize the model.
    model = CNN_model.CNN(config)
    #Loads the last checkpoint.
    model.load_weights()

    logging.info("Loading test dataset")
    x_test, y_test = load_dataset_numpy(config['test_dataset'], args.data_path, label_strings=config['label_strings'])

    prediction = model.model.predict_classes(x_test, config['batch_size'])
    print("Accuracy: ", (prediction.squeeze() == y_test).astype(np.float).sum() / len(y_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='model configuration file')
    parser.add_argument('--data_path', required=True, help='path to the image dataset folder')
    args = parser.parse_args()

    evaluate_testset(args)
