# MAAJ 5308: FINAL PROJECT
## Semantic Map Labelling
### Team Members
Max Polzin

### Overview
This project implements a logistic regression classifer model with k-folds validation and various other scripts to evaulate the on two datasets.
The following intructions detail how to reproduce the results discussed in the acompying report by running
the provided code in Colabs or in your own local desktop environment.

## Dependencies:
This project uses ????? YOLOv3 ..

## Setup and Usage:
### Setting up the local Environment
1. Unzip the provided file called "SemanticMapLabels.zip" OR visit https://github.com/MaxPolzinCU/SemanticMapLabels and clone the repository
2. Install Python 3.8.5 and Pip 20.0.2 or greater
3. Install the Intel RealSense SDK Python wrapper with the following command: "pip install pyrealsense2"
4. Install the following libraries:
- numpy 1.20.2
- matplotlib 3.4.1
- openCV 4.5.2
5. Clone the official Darknet YOLOv3 repository found at: https://github.com/pjreddie/darknet

Follow the instructions provided in the following link to build the library: https://pjreddie.com/darknet/yolo/
- The MakeFile parameters used are all default except for as follows:
    - GPU=1
    - CUDNN=0
    - OPENCV=1
    - OPENMP=0
    - DEBUG=0
Once completed the desired weights (e.g. "yolov3-tiny.weights") should be moved into the ./model/weights folder and the compiled library "libdarknet.so" should be placed in the ./model folder. 

Alternatly the weights provided in this repository and the compiled "libdarknet.so" can be used if this code is being run on a CUDA enabled GPU.

### Excuting locally
1. Open a terminal and cd into the "SemanticMapLabels" folder.
2. Run: "python classifierDepth.py" \
This will generate a 2D plot representing the environment captured by the stereo camera with annotated labels of the detected objects.   
4. Run: "python classifierWebcam2.py" \
This will access the computers webcam, if available, and perform objection detection while also ????????

## Repo Contents:
- **classifierDepth.py**: Main script needed, allows for both object detection and mapping using the Inteal RealSense D435i stereo camera

- **classifierSingle.py**: Test object detection on a single image, feed it a input image from ./models/data

- **classifierWebcam.py**: Test object detection with a webcam

- **classifierWebcam2.py**: Test object detection with a webcam, uses a structure more similar to that of the main "classifierDepth.py" script

- **models**: Folder to contain everything related to the ML models
    - *weights*: Folder to contain pre-trained weights for YOLOv3 Network
        - yolov3-tiny.weights: ????
    - *data*: Folder to contain and labels or datasets to be used
        - dog.jpg: Test image for "classifierSingle.py"
        - coco.names: Contains all the labels from the COCO ??? database
    - *cfg*: Folder to contain all config files for networks used
        - coco.data: Config paramters for COCO ?????
    - *libdarknet.so*: Pre-compiled Darknet library using YOLOv3 and trained on ????

