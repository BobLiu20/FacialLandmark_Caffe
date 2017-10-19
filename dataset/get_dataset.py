import os , sys
from zipfile import ZipFile
from urllib import urlretrieve

MYPATH = os.path.dirname(os.path.abspath(__file__))

theurl='http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/train.zip'
filename = os.path.join(MYPATH, 'AFLW.zip')
if os.path.isfile(filename):
    print "AFLW.zip training data already downloaded"
else:
    print "Downloading "+theurl + " ....."
    name, hdrs = urlretrieve(theurl, filename)
    print "Finished downloading AFLW....."
    print 'Extracting zip data...'
    with ZipFile(filename) as theOpenedFile:
        theOpenedFile.extractall(MYPATH)
        theOpenedFile.close()
    print "Done extraction AFLW zip folder"

