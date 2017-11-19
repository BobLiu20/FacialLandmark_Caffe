import numpy as np
import cv2
import os
import string, random

def reader_alfw():
    datasetPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset')
    lines = open("%s/trainImageList.txt"%(datasetPath)).readlines()
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
        bbox = [facePos[0], facePos[1], facePos[2]-facePos[0], facePos[3]-facePos[1]]
        yield im, landmark5, bbox

def reader_celeba():
    datasetPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset')
    #lines = open("%s/5pts.lst"%(datasetPath)).readlines()
    lines = open("%s/celeba_ZF_bbox_pts.txt"%(datasetPath)).readlines()
    np.random.shuffle(lines)
    for line in lines:
        sample = line.split()
        img = cv2.imread(os.path.join("%s/img_celeba"%(datasetPath), sample[0]))
        if img is None:
            print "The image is not exist: ", os.path.join("%s/img_celeba"%(datasetPath))
            continue
        data = map(int, sample[1:])
        bbox = data[0:4] # x1,y1 ,x2,y2
        bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]# only for ZF_bbox_pts
        pts = data[4:] # x1,y1,x2,y2...
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue
        yield img, pts, bbox

