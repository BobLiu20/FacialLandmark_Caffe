# by Bob in 20170326
import caffe
import numpy as np
import cv2
import os
import string, random

from batch_reader import BatchReader

class ImageInputDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        self.params = eval(self.param_str)

        # store input as class variables
        self.batch_size = self.params['batch_size']

        # Create a batch loader to load the images.
        # we can disable reader when test
        if self.params['need_reader']:
            self.batch_reader = BatchReader(**self.params)
            self.batch_generator = self.batch_reader.batch_generator()

        # === reshape tops ===
        top[0].reshape(
            self.batch_size, 3, self.params['img_size'], self.params['img_size'])
        top[1].reshape(
            self.batch_size, 10)

    def forward(self, bottom, top):
        """
        Load data.
        """
        images, labels = self.batch_generator.next()
        top[0].data[...] = images
        top[1].data[...] = labels

    def reshape(self, bottom, top):
        # === reshape tops ===
        top[0].reshape(
            self.batch_size, 3, self.params['img_size'], self.params['img_size'])
        top[1].reshape(
            self.batch_size, 10)

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


