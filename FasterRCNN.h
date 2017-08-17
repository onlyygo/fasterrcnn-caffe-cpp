#ifndef FASTERRCNN_H_
#define FASTERRCNN_H_
#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include "boost/shared_ptr.hpp"
#include "caffe/caffe.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using namespace std;
using boost::shared_ptr;

#include "config.h"
#include "nms.h"

class FasterRCNN {
public:
	FasterRCNN();
	~FasterRCNN();

	void initRPN_NET(const string& model_file, const string& weights_file);
	void initFastRCNN_NET(const string& model_file, const string& weights_file);

	void proposal_im_detect(cv::Mat image0);
	void fast_rcnn_im_detect(cv::Mat image0);
	void fast_rcnn_conv_feat_detect();

	static void print_net(boost::shared_ptr<Net<float> > net);
	static void vis_detections(cv::Mat &image, int num_out,
			float* detection_boxes, float* detection_scores, float CONF_THRESH);

	float getFeatureWidth() {
		return feature_width_;
	}
	;
	float getFeatureHeight() {
		return feature_height_;
	}
	;
	float getDetectionNum() {
		return detection_num;
	}
	;
	float* getDetectionBoxes() {
		return detection_boxes;
	}
	;
	float* getDetectionScores() {
		return detection_scores;
	}
	;
	boost::shared_ptr<Net<float> > getRPN_NET() {
		return rpn_net;
	}
	;
	boost::shared_ptr<Net<float> > getFastRCNN_NET() {
		return fast_rcnn_net;
	}
	;
	void release();
private:
	void prep_im_for_blob(cv::Mat image0, cv::Mat& image0_resize,
			float& im_scale);
	void proposal_locate_anchors(int feature_width, int feature_height,
			float* anchors);
	void fast_rcnn_bbox_transform_inv(const float* box_deltas, float* anchors,
			float* pred_boxes, int num);
	void clip_boxes(float* pred_boxes, int im_width, int im_height, int num);
	void filter_boxes_sort(float* pred_boxes, float* pred_scores, int num,
			int &valid_num);
	void precess_bbox_pred(const float* data, int num, int channels, int width, int height, float* &pred_boxes);
	void precess_cls_prob(const float* data, int num, int channels, int width, int height, float* &pred_scores);
	boost::shared_ptr<Net<float> > rpn_net;
	boost::shared_ptr<Net<float> > fast_rcnn_net;
	Config cfg;

	float scale;
	float input_width;
	float input_height;

	float feature_width_;
	float feature_height_;

	float *proposal_boxes;
	float *proposal_scores;
	int proposal_num;

	float *detection_boxes;
	float *detection_scores;
	int detection_num;
};
#endif
