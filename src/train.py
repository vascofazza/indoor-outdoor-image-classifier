from src.util.config import load_configuration
from src.util.utils import get_last_checkpoint, load_dataset_numpy
from src.model import CNN_model
import argparse
import logging

class Trainer:

    def __init__(self, config, model, initial_epoch = 0):
        self.config = config
        self.model = model
        self.initial_epoch = initial_epoch

    def fit(self, data_path, num_epochs):
        logging.info("Loading training dataset")
        x_train, y_train = load_dataset_numpy(self.config['train_dataset'], data_path, label_strings=self.config['label_strings'])
        logging.info("Loading test dataset")
        x_test, y_test = load_dataset_numpy(self.config['test_dataset'], data_path, label_strings=self.config['label_strings'])

        logging.info("Starting training...")
        for epoch in range(self.initial_epoch, num_epochs):
            #['acc', 'loss', 'val_acc', 'val_loss']
            history = self.model.model.fit(x_train, y_train, self.config['batch_size'], 1, validation_data=(x_test, y_test), shuffle=self.config['shuffle'])
            # print("Validation loss: %f\nValidation accuracty: %f"%(history.history['val_loss'][0], history.history['val_acc'][0]))
            self.model.save_weights(epoch)


def train(args):
    config = load_configuration(args.config)
    model = CNN_model.CNN(config)
    last_checkpoint, last_epoch = get_last_checkpoint(config['checkpoint_dir'])
    if args.resume_training:
        if last_checkpoint is None:
            logging.warning("Error retreiving last model checkpoint -- Weights not loaded.")
        else:
            model.load_weights(last_checkpoint)
            logging.info("Weights loaded -- epoch %d." % last_epoch)

    trainer = Trainer(config, model, last_epoch+1)
    num_epochs = config['num_epochs']
    trainer.fit(args.data_path, num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='model configuration file')
    parser.add_argument('--data_path', required=True, help='path to the image dataset folder')
    parser.add_argument('--resume_training', required=False, action='store_true', default=False,
                        help='restore model last checkpoint and continue training')
    args = parser.parse_args()

    train(args)
