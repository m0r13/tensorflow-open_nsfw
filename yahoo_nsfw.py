#!/usr/bin/env python
import sys
import tensorflow as tf

from image_utils import create_raw_image_loader
from model import OpenNsfwModel, InputType

class YahooNSFWClassifier:
    def __init__(self, weights_path):
        self.session = tf.Session()
        self.model = OpenNsfwModel()
        self.model.build(weights_path=weights_path)
        self.session.run(tf.global_variables_initializer())

        self.fn_load_image = create_raw_image_loader()

    def classify(self, image):
        image = self.fn_load_image(image)
        predictions = self.session.run(self.model.predictions, feed_dict={self.model.input: image})
        return predictions

if __name__ == "__main__":
    from PIL import Image
    classifier = YahooNSFWClassifier("data/open_nsfw-weights.npy")
    print("NSFW score: %f" % classifier.classify(Image.open(sys.argv[1]))[0][1])
