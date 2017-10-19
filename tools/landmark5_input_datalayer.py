# by Bob in 20170326
import numpy as np
import cv2
import caffe
import os
import string, random

from augumentation import pts_augumentation

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
        self.debug = debug

        self.image_generator = self.__image_generator()

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        return self.image_generator.next()

    def __image_generator(self):
        datasetPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset')
        lines = open("%s/trainImageList.txt"%(datasetPath)).readlines()
        while True:
            np.random.shuffle(lines)
            for line in lines:
                line = line.split()
                path = line[0]
                facePos = map(int, [line[1], line[3], line[2], line[4]]) # x1,y1,x2,y2
                landmark5 = map(float, line[5:]) # x1,y1,...,x5,y5
                #landmark5 = np.array(landmark5, dtype=np.float32).reshape(5, 2)
                #landmark5[:,0] = (landmark5[:, 0] - float(facePos[0])) / float(facePos[2]-facePos[0])
                #landmark5[:,1] = (landmark5[:, 1] - float(facePos[1])) / float(facePos[3]-facePos[1])
                im = cv2.imread(os.path.join(datasetPath, path))
                if im is None:
                    continue
                #_im = im[facePos[1]:facePos[3], facePos[0]:facePos[2]]

                bbox = [facePos[0], facePos[1], facePos[2]-facePos[0], facePos[3]-facePos[1]]
                _im, landmark5 = pts_augumentation(im, landmark5, bbox, self.im_shape[0], 5, debug=self.debug)

                #_im = cv2.resize(_im, self.im_shape)
                _im = np.transpose(_im, (2, 0, 1))
                _im = _im.astype(np.float32)
                _im = _im/127.5-1.0
                _label = landmark5.reshape(10)
                yield _im, _label

if __name__ == "__main__":
    a = BatchLoader({"im_shape": (40, 40)}, debug=True)
    print a.load_next_image()

