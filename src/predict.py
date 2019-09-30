from src.util.config import load_configuration
from src.util.utils import get_last_checkpoint, load_image
from src.model import CNN_model
import argparse
import logging
import numpy as np


def predict(args):
    config = load_configuration(args.config)
    model = CNN_model.CNN(config)
    model.load_weights()

    logging.info("Loading test dataset")

    img_size = (config['img_size'], config['img_size'])
    image = load_image(args.input, img_size)
    image = np.expand_dims(image, 0)

    prediction = model.predict(image)
    print(args.input, "prediction: " + prediction)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='model configuration file')
    parser.add_argument('--input', required=True, help='input image')
    args = parser.parse_args()

    predict(args)
