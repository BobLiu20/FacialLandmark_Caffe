#coding=utf-8
import sys, os
import caffe
import numpy as np
import cv2

root = os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + "/../")

# set your model in here
if len(sys.argv) != 3:
    print "Run like python test.py xxx.caffemodel deploy.prototxt"
    sys.exit()
# deploy = root + '/training/deploy.prototxt'
# caffe_model = root + '/model/_iter_100000.caffemodel'
deploy = sys.argv[2]
caffe_model = sys.argv[1]
img_folder = root + '/testing/test_images/'

net = caffe.Net(deploy, caffe_model, caffe.TEST)

for idx, im_path in enumerate(os.listdir(img_folder)):
    im_path = os.path.join(img_folder, im_path)
    im = cv2.imread(im_path)
    if im is None:
        print "Invaild image: ", im_path
        continue
    # preprocess
    im = cv2.resize(im, net.blobs['data'].data.shape[2:])
    im_ = np.transpose(im, (2, 0, 1))
    im_ = im_.astype(np.float32)
    im_ = im_/127.5-1.0
    # feet forward
    net.blobs['data'].data[0,...] = im_
    out = net.forward()
    output = net.blobs[net.blobs.keys()[-1]].data
    output = output.reshape((5, 2))
    # change to original image
    output = output * np.array([im.shape[1], im.shape[0]])# net.blobs['data'].data.shape[2:]
    output = output.astype(np.int)
    # save result
    for o in output:
        cv2.circle(im, tuple(o), 2, (0,255,0), -1)
    out_path = im_path.replace('test_images', 'test_output')
    if not os.path.isdir(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))
    cv2.imwrite(out_path, im)
    print "Save to : ", out_path
    #print output

