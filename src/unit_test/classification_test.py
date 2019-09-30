import unittest

from src.model import CNN_model
from src.util.config import load_configuration
from src.util.utils import load_image
import numpy as np


class ClassificationTest(unittest.TestCase):

    def setUp(self):
        self.config = load_configuration("config/train_params.yml")
        self.model = CNN_model.CNN(self.config)
        self.model.load_weights()
        self.img_size = (self.config['img_size'], self.config['img_size'])

    def test_indoor(self):
        image = load_image("data/test/indoor_test.jpg", self.img_size)
        image = np.expand_dims(image, 0)
        self.assertEqual(self.model.predict(image), 'indoor')

    def test_outdoor(self):
        image = load_image("data/test/outdoor_test.jpg", self.img_size)
        image = np.expand_dims(image, 0)
        self.assertEqual(self.model.predict(image), 'outdoor')


if __name__ == '__main__':
    unittest.main()
