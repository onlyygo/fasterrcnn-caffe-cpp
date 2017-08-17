# Faster-RCNN-by-C-
implement faster-rcnn in pure C++

For linux: It is easy to make it in Linux. 

1, download caffe-faster-R-CNN from 'https://github.com/ShaoqingRen/caffe', which forked from BLVC/caffe. The unzip it to this directory.

2, compile that, and copy the 'libcaffe.so' to this directory.

3, edit the 'build.sh' to specify the 'hdf5 include path' and other 'include' and 'lib' path. 

4, do 'sh build.sh'

5, do './Demo_Detection'

In Windows, you should create a new project by IDE, like Visual Studio, and add the '*.cpp, *.h' to it. And you shoud set the 'include directory'(caffe, glog, opencv), 'lib directory'(caffe, glog, opencv) and add the 'caffe.dll,glog*.dll, 
opencv_core*.dll, opencv_highgui*.dll, opencv_imgproc*.dll' link into project.

Now I have finished the 'Test-version' and it is only support 9 anchors setting, if you want to modify the number of anchors, please edit the ANCHORS variable in 'config.h'. And the 'Train-version' have some bug yet, which in the 'Train_alpha'. If you are in interested it, please debug it. Thanks.
If you have some question, please tell me or send email to 'onlyygo@qq.com'.
