# by Bob in 20170326
import caffe
import numpy as np
import cv2
import os
import string, random

from augumentation import pts_augumentation
from landmark5_reader import reader_alfw, reader_celeba

class ImageInputDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        self.params = eval(self.param_str)

        # store input as class variables
        self.batch_size = self.params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(self.params)

        # === reshape tops ===
        top[0].reshape(
            self.batch_size, 3, self.params['im_shape'][0], self.params['im_shape'][1])
        top[1].reshape(
            self.batch_size, 10)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, label = self.batch_loader.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = label

    def reshape(self, bottom, top):
        # === reshape tops ===
        top[0].reshape(
            self.batch_size, 3, self.params['im_shape'][0], self.params['im_shape'][1])
        top[1].reshape(
            self.batch_size, 10)

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass

class BatchLoader(object):
    def __init__(self, params, debug=False):
        self.im_shape = params['im_shape']
        self.dataset = params.get('dataset', ['alfw'])
        self.debug = debug

        self.image_generator = self.__image_generator()

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        return self.image_generator.next()

    def __image_generator(self):
        readers = []
        if 'alfw' in self.dataset:
            readers.append(reader_alfw)
        if 'celeba' in self.dataset:
            readers.append(reader_celeba)
        if len(readers) == 0:
            raise Exception("No reader set...")
        errCount = 0
        while True:
            for reader in readers:
                for im, landmark5, bbox in reader():
                    _im, _landmark5 = pts_augumentation(im, landmark5, bbox, self.im_shape[0], 5, debug=self.debug)
                    if not (np.all(_landmark5 <= 1.0) and np.all(_landmark5 >=0.0)):
                        errCount += 1
                        if errCount % 10 == 9:
                            print "bad landmark encounter count: %d"%errCount
                        continue
                    errCount = 0
                    _im = np.transpose(_im, (2, 0, 1))
                    _im = _im.astype(np.float32)
                    _im = _im/127.5-1.0
                    _label = _landmark5.reshape(10)
                    yield _im, _label

if __name__ == "__main__":
    a = BatchLoader({"im_shape": (128, 128), "dataset": ['celeba']}, debug=True)
    print a.load_next_image()
    print a.load_next_image()
    print a.load_next_image()
    print a.load_next_image()
    print a.load_next_image()
    print a.load_next_image()
    print a.load_next_image()
    print a.load_next_image()
    #a = BatchLoader({"im_shape": (40, 40), "dataset": ['alfw', 'celeba']}, debug=False)
    #while True:
    #    a.load_next_image()

