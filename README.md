### What's this
1. Facial Landmark Detection with Caffe CNN.
2. Implementation of the two nets from different paper.
 * Net1: [Approaching human level facial landmark localization by deep learning](http://www.sciencedirect.com/science/article/pii/S0262885615001341?via%3Dihub)
 * Net2: [TCNN: Facial Landmark Detection with Tweaked Convolutional Neural Networks](http://www.openu.ac.il/home/hassner/projects/tcnn_landmarks/)

### How to prepare dataset
Just enter dataset folder and run script to get training data
 ```
 cd dataset
 python get_dataset.py
 ```

### How to training
Please run caffe command in the root of this project.
 * train net1
 ```
 caffe train --solver=training/net1/solver.prototxt --gpu=0
 ```
 * train net2
 ```
 caffe train --solver=training/net2/vanilla_adam_solver.prototxt --gpu=0
 ```

### How to testing or predict
 ```
 cd testing
 python test.py ../model/net1/_iter_100000.caffemodel ../training/net1/deploy.prototxt
 ```
 Please replace correct path of caffe model and deploy file in above command.

### How to do a benchmark
 ```
 cd benchmark
 python test.py ../model/net1/_iter_100000.caffemodel ../training/net1/deploy.prototxt
 ```
 Please replace correct path of caffe model and deploy file in above command.   
 After do that, you will get mean error of your model.

### Reference
 [VanillaCNN](https://github.com/ishay2b/VanillaCNN)

