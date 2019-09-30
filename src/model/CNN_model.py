from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from src.util.utils import to_string_label, get_last_checkpoint
import os
import logging

#Model class.
#Implements a 3-layer CNN with RELU activation and dropout.
#Kernel size and dropout value are customizable through an YML configuration file
class CNN:
    def __init__(self, config):
        self.config = config
        self.model = self.build_model()

    def predict(self, image):
        prediction = self.model.predict_classes(image, self.config['batch_size'])[0, 0]
        return to_string_label(prediction, self.config['label_strings'])

    def save_weights(self, epoch=-1):
        path = os.path.join(self.config['checkpoint_dir'], "model_weights_epoch_%d.hdf5" % epoch)
        self.model.save_weights(path)
        logging.info("Model weights saved.")

    #loads the weights from disk. If path is not provided, it loads the last available checkpoint.
    def load_weights(self, path=None):
        if path is None:
            last_checkpoint, last_epoch = get_last_checkpoint(self.config['checkpoint_dir'])
            if last_checkpoint is None:
                logging.warning("Error retreiving last model checkpoint -- Weights not loaded.")
            else:
                self.model.load_weights(last_checkpoint)
                logging.info("Weights loaded -- epoch %d." % last_epoch)
        else:
            self.model.load_weights(path)
            logging.info("Model weights loaded.")

    def build_model(self):
        #Image size is assumed square.
        img_size = self.config['img_size']

        if K.image_data_format() == 'channels_first':
            input_shape = (3, img_size, img_size)
        else:
            input_shape = (img_size, img_size, 3)

        kernel_size = self.config['kernel_size']

        model = Sequential()
        model.add(Conv2D(32, kernel_size, input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=kernel_size))

        model.add(Conv2D(32, kernel_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(kernel_size))

        model.add(Conv2D(64, kernel_size))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(kernel_size))

        #the last CNN layer is flattened and feeded into a dense layer for the final prediction.
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(self.config['dropout']))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        #binary crossentropy for binary classification.
        model.compile(loss='binary_crossentropy',
                      optimizer=self.config['optimizer'],
                      metrics=['accuracy'])
        return model