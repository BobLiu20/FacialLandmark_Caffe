import os
os.environ['GLOG_minloglevel'] = '2' # warnings
import caffe
import sys
import cv2
import numpy as np

if len(sys.argv) != 3:
    print "run python main.py xxx.caffemodel deploy.prototxt"
    sys.exit()
PATH_TO_DEPLOY_TXT = sys.argv[2]
PATH_TO_WEIGHTS = sys.argv[1]

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
BENCHMARK_PATH = os.path.join(ROOT, 'benchmark')
DATA_PATH = os.path.join(ROOT, 'dataset')

def isDatasetReady():
    for n in ["lfw_5590", "net_7876", "testImageList.txt"]:
        if not os.path.exists(os.path.join(DATA_PATH, n)):
            return False
    return True

def mseNormlized(groundTruth, pred):
    delX = groundTruth[0]-groundTruth[2] 
    delY = groundTruth[1]-groundTruth[3] 
    interOc = (1e-6+(delX*delX + delY*delY))**0.5  # Euclidain distance
    diff = (pred-groundTruth)**2
    sumPairs = (diff[0::2]+diff[1::2])**0.5  # Euclidian distance 
    return (sumPairs / interOc)  # normlized 

class Predictor:
    def __preprocess(self, im):
        im_resize = cv2.resize(im, self.net.blobs[self.net.blobs.keys()[0]].data.shape[2:][::-1])
        im_resize = np.transpose(im_resize, (2, 0, 1))
        im_resize = im_resize.astype('f4')
        im_resize = im_resize/127.5-1.0
        return  im_resize

    def predict(self, im):
        im_resize = self.__preprocess(im)
        self.net.blobs[self.net.blobs.keys()[0]].data[0, ...] = im_resize
        prediction = self.net.forward()[self.net.blobs.keys()[-1]][0]
        return prediction

    def __init__(self, protoTXTPath, weightsPath):
        caffe.set_device(0)
        caffe.set_mode_gpu()
        self.net = caffe.Net(protoTXTPath, weightsPath, caffe.TEST)

if __name__ == "__main__":
    if not isDatasetReady():
        print "The dataset is not exist. Please go to dataset folder to download it."
        sys.exit()

    predictor = Predictor(PATH_TO_DEPLOY_TXT, PATH_TO_WEIGHTS)

    errorPerLandmark = np.zeros(5, dtype ='f4')
    numCount = 0
    failureCount = 0
    print "Processing:"
    for line in open(os.path.join(DATA_PATH, "testImageList.txt")):
        line = line.split()
        path = line[0]
        facePos = map(int, [line[1], line[3], line[2], line[4]]) # x1,y1,x2,y2
        landmark5 = map(float, line[5:]) # x1,y1,...,x5,y5
        landmark5 = np.array(landmark5, dtype=np.float32).reshape(5, 2)
        landmark5[:,0] = (landmark5[:, 0] - float(facePos[0])) / float(facePos[2]-facePos[0])
        landmark5[:,1] = (landmark5[:, 1] - float(facePos[1])) / float(facePos[3]-facePos[1])

        image = cv2.imread(os.path.join(DATA_PATH, path))
        if image is None:
            print "Skip ", path
            continue
        faceImage = image[facePos[1]:facePos[3], facePos[0]:facePos[2]]

        prediction = predictor.predict(faceImage)
        normlized = mseNormlized(landmark5.reshape([10]), prediction)
        errorPerLandmark += normlized
        numCount += 1
        if normlized.mean() > 0.1: # Count error above 10% as failure
            failureCount += 1
        print "\r%d"%(numCount),
        sys.stdout.flush()

    meanError = (errorPerLandmark/numCount).mean()*100.0
    print "\nResult: \nmean error: %.05f \nfailures: %.02f%%(%d/%d)"%(meanError, 
        (failureCount/float(numCount)*100.0), failureCount, numCount)

