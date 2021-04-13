
# REQUIRED LIBRARIES:
from ctypes import *
import math
import random
import cv2 as cv
import numpy as np
from random import randint
import pyrealsense2 as rs
import matplotlib.pyplot as plt

# SUPPORTING STRUCTS:
#Bounding box
class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

#Input image
class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

#Detection params
class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

#Frame metadata
class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

#?????????
class IplROI(Structure):
    pass

#???????
class IplTileInfo(Structure):
    pass

#???????
class IplImage(Structure):
    pass

IplImage._fields_ = [
    ('nSize', c_int),
    ('ID', c_int),
    ('nChannels', c_int),               
    ('alphaChannel', c_int),
    ('depth', c_int),
    ('colorModel', c_char * 4),
    ('channelSeq', c_char * 4),
    ('dataOrder', c_int),
    ('origin', c_int),
    ('align', c_int),
    ('width', c_int),
    ('height', c_int),
    ('roi', POINTER(IplROI)),
    ('maskROI', POINTER(IplImage)),
    ('imageId', c_void_p),
    ('tileInfo', POINTER(IplTileInfo)),
    ('imageSize', c_int),          
    ('imageData', c_char_p),
    ('widthStep', c_int),
    ('BorderMode', c_int * 4),
    ('BorderConst', c_int * 4),
    ('imageDataOrigin', c_char_p)]


class iplimage_t(Structure):
    _fields_ = [('ob_refcnt', c_ssize_t),
                ('ob_type',  py_object),
                ('a', POINTER(IplImage)),
                ('data', py_object),
                ('offset', c_size_t)]

NUM_SAMPLES_FILTER = 10

# SYSTEM SETUP:
#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("./models/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

#SUPPORTING FUNCTIONS:
#???????
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

#??????????
def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

#??????????
def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res


#Converts a input frame to a array of pixels:
def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

# def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
#     """if isinstance(image, bytes):  
#         # image is a filename 
#         # i.e. image = b'/darknet/data/dog.jpg'
#         im = load_image(image, 0, 0)
#     else:  
#         # image is an nparray
#         # i.e. image = cv2.imread('/darknet/data/dog.jpg')
#         im, image = array_to_image(image)
#         rgbgr_image(im)
#     """
#     im, image = array_to_image(image)
#     rgbgr_image(im)
#     num = c_int(0)
#     pnum = pointer(num)
#     predict_image(net, im)
#     dets = get_network_boxes(net, im.w, im.h, thresh, 
#                              hier_thresh, None, 0, pnum)
#     num = pnum[0]
#     if nms: do_nms_obj(dets, num, meta.classes, nms)

#     res = []
#     for j in range(num):
#         a = dets[j].prob[0:meta.classes]
#         if any(a):
#             ai = np.array(a).nonzero()[0]
#             for i in ai:
#                 b = dets[j].bbox
#                 res.append((meta.names[i], dets[j].prob[i], 
#                            (b.x, b.y, b.w, b.h)))

#     res = sorted(res, key=lambda x: -x[1])
#     if isinstance(image, bytes): free_image(im)
#     free_detections(dets, num)
#     return res


def predictFrames(net, meta, videoSource, thresh=.8, hier_thresh=.5, nms=.45):
    #Store the depths, x's, and y's
    depth = []
    x = []
    y = []
    labels = []
    label_num = []

    #Colours to draw on GUI with
    classes_box_colors = [(0, 0, 255), (0, 255, 0)]  #red for palmup --> stop, green for thumbsup --> go
    classes_font_colors = [(255, 255, 0), (0, 255, 255)]
    
    #Wait for a coherent pair of frames: depth and color
    frames = videoSource.poll_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame and not color_frame:
        return x, y, labels, label_num,depth, color_frame #Return null

    # # Convert images to numpy arrays
    depth_arr = np.asanyarray(depth_frame.get_data())
    color_arr = np.asanyarray(color_frame.get_data())

    #Convert the input frame to a array of pixels
    im, arr = array_to_image(color_arr) 
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]

    if (nms): do_nms_obj(dets, num, meta.classes, nms);
    
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0: #Detection threshold
                b = dets[j].bbox
                x.append(b.x)
                y.append(b.y)
                labels.append(meta.names[i].decode('utf-8'))
                label_num.append(i)

                #Calculate corners of bounding box
                x1 = int(b.x - b.w / 2.)
                y1 = int(b.y - b.h / 2.)
                x2 = int(b.x + b.w / 2.)
                y2 = int(b.y + b.h / 2.)

                #Draw bounding box on image and add label
                cv.rectangle(color_arr, (x1, y1), (x2, y2), classes_box_colors[0], 2)
                cv.putText(color_arr, meta.names[i].decode('utf-8'), (x1, y1 - 20), 1, 1, classes_font_colors[0], 2, cv.LINE_AA)

                #Get distance to bounding box centroid
                depth.append(depth_frame.get_distance(int(b.x),int(b.y)))                        
    
    cv.imshow('output', color_arr)
    cv.imshow('depth', depth_arr)
    if cv.waitKey(1) == ord('q'):
        return      
    return x,y,depth,labels,label_num,color_arr

# Get the target depth
def getTarget(pipeD435, net, meta, guiShow):
    #Network will make predictions on each frame
    x,y,depth,labels,label_num,frame = predictFrames(net, meta, pipeD435) 
    
    #Calculate angle between camera origin and the centroid of bounding box
    theta = []
    for i in range(len(x)):
        theta.append(x[i] /600*87-87.0/2) 
        if (theta[i] < 0): theta[i] += 360
        if (theta[i] > 180): theta[i] = theta[i] - 360

    detections = [x,y,depth,theta,labels,label_num]
    return(detections)

# Generate semantic map
def mapGenerator(detections, intr, frameFilter):
    plt.ion()
    plt.grid(color='r', linestyle='-', linewidth=2)
    plt.axis([-2.5,2.5,0,5])

    for i in range(0,len(detections[0])):
        #Transform x,y,depth coordinates to new xyz:
        cxy = [detections[0][i], detections[1][i]]
        depth = detections[2][i]
        point = rs.rs2_deproject_pixel_to_point(intr, cxy, depth) #Outputs xyz in cameras reference frame in 3D space

        #Display map:
        if len(point) != 0: 
            p=plt.plot(point[1], point[2],marker='o',label=detections[4][i])
            plt.text(point[1],point[2],detections[4][i])
            fig.canvas.draw()

    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv.cvtColor(img,cv.COLOR_RGB2BGR)

    #img2 = cv.createMat(640,480,frameFilter)
    #img2=np.array(frameFilter,dtype=np.uint8)
    #img2=cv.adaptiveThreshold(frameFilter,1,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,3,0)
    cv.imshow("Labelled Map",img)
    cv.imshow("Heat map",frameFilter)
    plt.clf()
    plt.ioff()

    #ADDD make it so you have "persistence" in the map, everytime a object at a specific x,y (+/- 10 pix) you
    #increment a count (x,y,depth,label,count) for that object and if count > thresh then draw it on the map, otherwise do 
    #not draw it on the map

    #ADD Make sure this plot is slowing FPS to much

if __name__ == "__main__":
    #Set to 0 to turn off GUI's:
    guiShow = 1 
    
    #Load YOLOv3:
    net = load_net("./models/cfg/yolov3.cfg".encode('utf-8'), "./models/weights/yolov3.weights".encode('utf-8'), 0)
    meta = load_meta("./models/cfg/coco.data".encode('utf-8'))
    #print(meta.names[79])
    
    #Init D435i camera pipeline:
    pipeD435 = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg = pipeD435.start(config)

    profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

    fig = plt.figure() #Figure for plotting the map

    storedDetections = [10] #Store the most recent detections
    frameFilter = np.zeros((480,640,80,NUM_SAMPLES_FILTER)) #3D numpy array that is the image size (w,h) by the number of classes 

    #While camera is on perform detection and mapping:
    counter = 0
    while pipeD435.poll_for_frames:
        counter=counter+1
        if counter >NUM_SAMPLES_FILTER-1:
            counter =0
        np.delete(frameFilter,0,axis=3)
        np.append(frameFilter,np.zeros((480,640,80))
      
        #Localize detected objects:
        detections = getTarget(pipeD435, net, meta, guiShow) #Return contains array of [x,y,depth,theta,label] for each detected object
        
        for i in range(len(detections[0])):
            frameFilter[int(detections[1][i])][int(detections[0][i])][int(detections[5][i])][counter] = 1
        newframeFilter=frameFilter.sum(axis=3) #sum across the 10 samples (4th axis of the 4-d array)
        newframeFilter=newframeFilter.sum(axis=2) #sum across the labels
        #storedDetections.append(detections)

        #Remove excess detections
        # if len(storedDetections) > 10:
        #     storedDetections.remove(0)

        #Average the detected points, remove noise


        #Create Semantic map:
        if len(detections[0]) !=0: mapGenerator(detections, intr, newframeFilter)

#ADD a def __init__ like in micamove.py and use multiple threads to handle network pred and mapping at the same time
