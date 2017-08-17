#include "FasterRCNN.h"

int main()
{
	string model_file = "./model/faster_rcnn_VOC2007_vgg_16layers_facedetection/proposal_test.prototxt";
	string weights_file = "./model/faster_rcnn_VOC2007_vgg_16layers_facedetection/proposal_final";
	string detection_model_file = "./model/faster_rcnn_VOC2007_vgg_16layers_facedetection/detection_test.prototxt";
	string detection_weights_file = "./model/faster_rcnn_VOC2007_vgg_16layers_facedetection/detection_final";

	//int GPUID = 0;
	//Caffe::SetDevice(GPUID);
	Caffe::set_mode(Caffe::CPU);
	
	FasterRCNN faster_rcnn;
	faster_rcnn.initRPN_NET(model_file, weights_file);
	faster_rcnn.initFastRCNN_NET(detection_model_file, detection_weights_file);
	string images[] = { "we.png"};
	for (int i = 0;i<1;i++){
		cv::Mat src = cv::imread(images[i]);
		faster_rcnn.proposal_im_detect(src);
		faster_rcnn.fast_rcnn_conv_feat_detect();
		faster_rcnn.vis_detections(src, faster_rcnn.getDetectionNum(), faster_rcnn.getDetectionBoxes(), faster_rcnn.getDetectionScores(), 0.75);
		faster_rcnn.release();
		cv::imwrite("faster-rcnn.jpg",src);
		cv::imshow("faster-rcnn",src);
		cv::waitKey(0);
	}
	return 0;
}