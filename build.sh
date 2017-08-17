g++ -DCPU_ONLY -std=c++11 Demo_Detection.cpp FasterRCNN.cpp -lcaffe -lglog -lopencv_core -lopencv_highgui -lopencv_imgproc \
-I./caffe-faster-R-CNN/include \
-I/usr/include/hdf5/serial \
-L. \
-o Demo_Detection