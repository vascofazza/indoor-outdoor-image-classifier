from src.util.config import load_configuration
from src.util.utils import get_last_checkpoint, load_image
from src.model import CNN_model
import argparse
import logging
import numpy as np

#Predicts the class for a single input image.
def predict(args):
    #Load the config file.
    config = load_configuration(args.config)
    #Initialize the model.
    model = CNN_model.CNN(config)
    #Load the last checkpoint.
    model.load_weights()

    logging.info("Loading test dataset")

    #Image size is assumed square.
    img_size = (config['img_size'], config['img_size'])
    image = load_image(args.input, img_size)
    image = np.expand_dims(image, 0)

    #Predict the most probable class
    prediction = model.predict(image)
    print(args.input, "prediction: " + prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='model configuration file')
    parser.add_argument('--input', required=True, help='input image')
    args = parser.parse_args()

    predict(args)
