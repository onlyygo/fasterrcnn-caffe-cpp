/*
 * ProposalTrain.h
 *
 *  Created on: Aug 27, 2016
 *      Author: cudahe
 */

#ifndef PROPOSALTRAIN_H_
#define PROPOSALTRAIN_H_

#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include "caffe/caffe.hpp"

using std::string;

struct ProposalConfig {

	int test_scales;
	int test_max_size;
	int batch_size;
	float fg_fraction;
	float bg_weight;
	bool do_scale;
	float feat_stride;
	float fg_thresh;
	float bg_thresh_lo;
	float bg_thresh_hi;
};
struct ImageRoiDB {

	cv::Mat image0;
	cv::Mat image0_resize;
	float im_scales;

	float* bbox_targets; //n*5
	int bbox_targets_num; //总行数
	float* boxes;
	int boxes_num; //gt's num
};
//单幅影像以及对应的roi

class ProposalTrain {
public:
	ProposalTrain(const string& solver_file, const string& weights_file,
			const string& test_net_def_file, int gpu_id);
	virtual ~ProposalTrain();

	void proposal_train(ImageRoiDB image_roidb);
	void prep_im_for_blob(cv::Mat image0, cv::Mat& image0_resize,
			float& im_scale, ProposalConfig conf);
	void proposal_prepare_image_roidb(ProposalConfig conf,
			ImageRoiDB &image_roidb);
	void proposal_generate_minibatch(ProposalConfig conf,
			ImageRoiDB image_roidb, float* &data_buf, float* &labels,
			float* &label_weights, float* &bbox_targets,
			float* &bbox_loss_weights);
    void save_mode(string name);
    void show_accuarcy();


private:

	void sample_rois(ProposalConfig conf, ImageRoiDB image_roidb,
			int fg_rois_per_image, int rois_per_image, float im_scales,
			float* &labels, float* &label_weights, float* &bbox_targets,
			float* &bbox_loss_weights);

	void boxoverlap(float* ex_rois, int m, float* gt_rois, int n,
			float* ex_gt_overlaps);
	void proposal_locate_anchors(int feature_width, int feature_height,
			float* anchors, ProposalConfig conf_proposal);
	void fast_rcnn_bbox_transform(float* src_rois, float* target_rois,
			float* regression_label, int fg_num);

	int init_config();
	void set_feature_size(int feature_width, int feature_height);
	void calc_feature_size(int image_width, int image_height);

public:
	caffe::shared_ptr<caffe::Solver<float> > solver_;
private:

	ProposalConfig conf_;
	string test_net_def_file_;
	int feature_width_;
	int feature_height_;
};

#endif /* PROPOSALTRAIN_H_ */
