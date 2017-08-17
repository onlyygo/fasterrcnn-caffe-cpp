/*
 * FastRcnnTrain.h
 *
 *  Created on: Aug 27, 2016
 *      Author: cudahe
 */

#ifndef FASTRCNNTRAIN_H_
#define FASTRCNNTRAIN_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include "caffe/caffe.hpp"
#include "ProposalTrain.h"

using std::string;

struct DetectionConfig {

	int test_scales;
	int test_max_size;
	int batch_size;
	float fg_fraction;
	float fg_thresh;
	float bg_thresh_lo;
	float bg_thresh_hi;
	bool do_scale;
	float bbox_thresh;
};

class FastRcnnTrain {
public:
	FastRcnnTrain(const string& solver_file, const string& weights_file,
			const string& test_net_def_file, int gpu_id);
	virtual ~FastRcnnTrain();

	void fast_rcnn_train(ImageRoiDB image_roidb);
	void prep_im_for_blob(cv::Mat image0, cv::Mat& image0_resize,
			float& im_scale, DetectionConfig conf);
	void fast_rcnn_prepare_image_roidb(DetectionConfig conf,
			ImageRoiDB& image_roidb);
	void fast_rcnn_get_minibatch(DetectionConfig conf, ImageRoiDB image_roidb,
			float* &data_buf, float* &rois, float* &labels,
			float* &bbox_targets, float* &bbox_loss_weights, int& sample_num);
	void set_anchors(float* anchors_data, int num);
	void free();
	void save_mode(string name);
	void show_accuarcy();
private:
	void get_bbox_regression_labels(float* bbox_targets_data,
			vector<int> keep_inds, float* &bbox_targets,
			float* &bbox_loss_weights);
	void sample_rois(DetectionConfig conf, ImageRoiDB image_roidb,
			int fg_rois_per_image, int rois_per_image, float* &rois,
			float* &bbox_targets, float* &bbox_loss_weights, int& sample_num);
	void fast_rcnn_map_im_rois_to_feat_rois(float im_scales, float* im_rois,
			int im_rois_num, float* &feat_rois);
	void fast_rcnn_bbox_transform(float* ex_boxes, float* gt_boxes,
			float* regression_label, int fg_num);
	void boxoverlap(float* ex_rois, int m, float* gt_rois, int n,
			float* ex_gt_overlaps);
	void init_config();

//	void set_feature_size(int feature_width, int feature_height);
//	void calc_feature_size(int image_width, int image_height);
public:
	caffe::shared_ptr<caffe::Solver<float> > solver_;
private:

	DetectionConfig conf_;
	string test_net_def_file_;
	float* anchors;
	int* vertical_index;
	float* vertical_score;
	int anchors_num_;
	vector<int> fg_inds;
	vector<int> bg_inds;
	//int feature_width_;
	//int feature_height_;
};

#endif /* FASTRCNNTRAIN_H_ */
