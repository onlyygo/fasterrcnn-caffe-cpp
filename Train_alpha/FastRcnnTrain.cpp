/*
 * FastRcnnTrain.cpp
 *
 *  Created on: Aug 27, 2016
 *      Author: cudahe
 */

#include "FasterRCNN.h"
#include "FastRcnnTrain.h"
#include "boost/algorithm/string.hpp"

FastRcnnTrain::FastRcnnTrain(const string& solver_file,
		const string& weights_file, const string& test_net_def_file,
		int gpu_id) {
	// TODO Auto-generated constructor stub

	// Set device id and mode
//	if (gpu_id >= 0) {
//		cout << "Use GPU with device ID " << gpu_id<<endl;
//		Caffe::SetDevice(gpu_id);
//		Caffe::set_mode(Caffe::GPU);
//	} else {
//		cout << "Must Use GPU ";
//		exit(-1);
//	}

	caffe::shared_ptr<Solver<float> > solver(
			new caffe::SGDSolver<float>(solver_file));
	solver_ = solver;
	solver_->net()->CopyTrainedLayersFrom(weights_file);
	test_net_def_file_ = test_net_def_file;
	init_config();
}

FastRcnnTrain::~FastRcnnTrain() {
	// TODO Auto-generated destructor stub
}

static void print_float(float* bu, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			cout << bu[i * cols + j] << "    ";
		}
		cout << endl;
	}
}
void FastRcnnTrain::fast_rcnn_train(ImageRoiDB image_roidb) {

	//第零步：将图像resize，并记录resize的因子
	prep_im_for_blob(image_roidb.image0, image_roidb.image0_resize,
			image_roidb.im_scales, conf_);
	//第一步：计算featuresize
	//calc_feature_size(image_roidb.image0_resize.cols, image_roidb.image0_resize.rows);
	//compute image_roidb.bbox_targets
	fast_rcnn_prepare_image_roidb(conf_, image_roidb);
	float* data_buf;
	float* rois;	//sample_num*5
	float* labels;	//sample_num*1
	float* bbox_targets;	//sample_num*8
	float* bbox_loss_weights;	//sample_num*8
	int sample_num;
	fast_rcnn_get_minibatch(conf_, image_roidb, data_buf, rois, labels,
			bbox_targets, bbox_loss_weights, sample_num);

	//print_float(rois, sample_num, 5);
	//print_float(labels, sample_num, 1);
	//print_float(bbox_targets, sample_num, 8);
	//print_float(bbox_loss_weights, sample_num, 8);
	//training
	solver_->net()->SetPhase(TRAIN);
	solver_->net()->blob_by_name("data")->Reshape(1, 3,
			image_roidb.image0_resize.rows, image_roidb.image0_resize.cols);
	solver_->net()->blob_by_name("labels")->Reshape(sample_num, 1, 1, 1);
	solver_->net()->blob_by_name("rois")->Reshape(sample_num, 5, 1, 1);
	solver_->net()->blob_by_name("bbox_targets")->Reshape(sample_num, 8, 1, 1);
	solver_->net()->blob_by_name("bbox_loss_weights")->Reshape(sample_num, 8, 1,
			1);
	solver_->net()->Reshape();
	solver_->net()->blob_by_name("data")->set_cpu_data(data_buf);
	solver_->net()->blob_by_name("labels")->set_cpu_data(labels);
	solver_->net()->blob_by_name("rois")->set_cpu_data(rois);
	solver_->net()->blob_by_name("bbox_targets")->set_cpu_data(bbox_targets);
	solver_->net()->blob_by_name("bbox_loss_weights")->set_cpu_data(
			bbox_loss_weights);

	//FasterRCNN::print_net(solver_->net());

	solver_->Step(1);
	delete[] data_buf;
	delete[] rois;
	delete[] labels;
	delete[] bbox_targets;
	delete[] bbox_loss_weights;
	delete[] image_roidb.bbox_targets;
}

void FastRcnnTrain::prep_im_for_blob(cv::Mat image0, cv::Mat& image0_resize,
		float& im_scale, DetectionConfig conf) {

	cv::Mat image0_src;
	image0.convertTo(image0_src, CV_32FC3);

	if (conf.do_scale) {
		//ŒÆËãËõ·ÅÒò×Ó
		int im_size_min = min(image0_src.rows, image0_src.cols);
		int im_size_max = max(image0_src.rows, image0_src.cols);

		im_scale = min(float(conf.test_scales) / im_size_min,
				float(conf.test_max_size) / im_size_max);

		int new_rows = round(image0_src.rows * im_scale);
		int new_cols = round(image0_src.cols * im_scale);
		cv::resize(image0_src, image0_resize, cv::Size(new_cols, new_rows), 0,
				0, CV_INTER_LINEAR);
	} else {

		im_scale = 1;
		image0_src.copyTo(image0_resize);
	}
}
/**
 * -> anchors + gt -> overlap -> fg_inds & bg_inds -> regression_label -> image_roidb.bbox_targets
 */
void FastRcnnTrain::fast_rcnn_prepare_image_roidb(DetectionConfig conf,
		ImageRoiDB& image_roidb) {

	//计算每一个样本与每一个gt的重叠度
	float* ex_gt_overlaps = new float[anchors_num_ * image_roidb.boxes_num];
	boxoverlap(anchors, anchors_num_, image_roidb.boxes, image_roidb.boxes_num, ex_gt_overlaps);

	//计算重叠度的行最大值和列最大值，以及取得最大值的索引
	//m行1列，每个元素对应该取行的最大值的索引
	vertical_index = new int[anchors_num_]();
	//m行1列，每个元素对应该行的最大值
	vertical_score = new float[anchors_num_]();
	//1行n列，每个元素对应该列取得最大值的索引
	int* horizon_index = new int[image_roidb.boxes_num]();
	//1行n列，每个元素对应该列的最大值
	float* horizon_score = new float[image_roidb.boxes_num]();

	for (int ex_index = 0; ex_index < anchors_num_; ex_index++) {

		//将位于图外anchor的overlap全部置-1
		float ex_x1 = anchors[ex_index * 4 + 0];
		float ex_y1 = anchors[ex_index * 4 + 1];
		float ex_x2 = anchors[ex_index * 4 + 2];
		float ex_y2 = anchors[ex_index * 4 + 3];
		if (ex_x1 < 0 || ex_y1 < 0 || ex_x2 < 0 || ex_y2 < 0
				|| ex_x1 >= image_roidb.image0.cols
				|| ex_y1 >= image_roidb.image0.rows
				|| ex_x2 >= image_roidb.image0.cols
				|| ex_y2 >= image_roidb.image0.rows) {

			for (int i = 0; i < image_roidb.boxes_num; i++) {
				ex_gt_overlaps[ex_index * image_roidb.boxes_num + i] = -1;
			}
		}

		//计算单科最高分
		for (int gt_index = 0; gt_index < image_roidb.boxes_num; gt_index++) {

			float score = ex_gt_overlaps[ex_index * image_roidb.boxes_num
					+ gt_index];

			if (score > horizon_score[gt_index]) {

				horizon_score[gt_index] = score;
				horizon_index[gt_index] = ex_index;
			}

			if (score > vertical_score[ex_index]) {

				vertical_score[ex_index] = score;
				vertical_index[ex_index] = gt_index;
			}
		}
	}

	//print_float(horizon_score, 1, image_roidb.boxes_num);
	//根据anchor的重叠度，确定那些是正样本，那些是负样本
	fg_inds.clear();
	bg_inds.clear();
	for (int ex_index = 0; ex_index < anchors_num_; ex_index++) {

		if (vertical_score[ex_index]>=conf_.bbox_thresh) {

			fg_inds.push_back(ex_index);
		}
		if (vertical_score[ex_index]>=conf_.bg_thresh_lo && vertical_score[ex_index]<conf_.bg_thresh_hi) {

			bg_inds.push_back(ex_index);
		}
	}
	//将正样本anchor转化为回归形式，即转化成差值（与gt的）
	float* src_rois = new float[fg_inds.size() * 4];
	float* target_rois = new float[fg_inds.size() * 4];
	float* regression_label = new float[fg_inds.size() * 4];

	for (int i = 0; i < fg_inds.size(); i++) {

		//获得anchors中的实际索引
		int index = fg_inds[i];
		src_rois[i * 4 + 0] = anchors[index * 4 + 0];
		src_rois[i * 4 + 1] = anchors[index * 4 + 1];
		src_rois[i * 4 + 2] = anchors[index * 4 + 2];
		src_rois[i * 4 + 3] = anchors[index * 4 + 3];
		//get index anchor 对应重叠度最大的gt
		target_rois[i * 4 + 0] = image_roidb.boxes[vertical_index[index] * 4 + 0];
		target_rois[i * 4 + 1] = image_roidb.boxes[vertical_index[index] * 4 + 1];
		target_rois[i * 4 + 2] = image_roidb.boxes[vertical_index[index] * 4 + 2];
		target_rois[i * 4 + 3] = image_roidb.boxes[vertical_index[index] * 4 + 3];
	}
	//计算回归标签
	fast_rcnn_bbox_transform(src_rois, target_rois, regression_label,
			fg_inds.size());
	//得到候选的bbox_targets m*5
	float* bbox_targets = new float[anchors_num_ * 5]();
	float* sums = new float[1 * 4]();
	float* squared_sums = new float[1 * 4]();
	for (int i = 0; i < fg_inds.size(); i++) {

		int index = fg_inds[i];
		bbox_targets[index * 5 + 0] = 1;	//类别id，将来转化为多类的需更改此项
		bbox_targets[index * 5 + 1] = regression_label[i * 4 + 0];
		bbox_targets[index * 5 + 2] = regression_label[i * 4 + 1];
		bbox_targets[index * 5 + 3] = regression_label[i * 4 + 2];
		bbox_targets[index * 5 + 4] = regression_label[i * 4 + 3];

		sums[0] += regression_label[i * 4 + 0];
		sums[1] += regression_label[i * 4 + 1];
		sums[2] += regression_label[i * 4 + 2];
		sums[3] += regression_label[i * 4 + 3];
		squared_sums[0] += (regression_label[i * 4 + 0]
				* regression_label[i * 4 + 0]);
		squared_sums[1] += (regression_label[i * 4 + 1]
				* regression_label[i * 4 + 1]);
		squared_sums[2] += (regression_label[i * 4 + 2]
				* regression_label[i * 4 + 2]);
		squared_sums[3] += (regression_label[i * 4 + 3]
				* regression_label[i * 4 + 3]);
	}

	sums[0] /= fg_inds.size();
	sums[1] /= fg_inds.size();
	sums[2] /= fg_inds.size();
	sums[3] /= fg_inds.size();
	squared_sums[0] /= fg_inds.size();
	squared_sums[1] /= fg_inds.size();
	squared_sums[2] /= fg_inds.size();
	squared_sums[3] /= fg_inds.size();

	squared_sums[0] -= (sums[0] * sums[0]);
	squared_sums[1] -= (sums[1] * sums[1]);
	squared_sums[2] -= (sums[2] * sums[2]);
	squared_sums[3] -= (sums[3] * sums[3]);

	squared_sums[0] = sqrt(squared_sums[0]);
	squared_sums[1] = sqrt(squared_sums[1]);
	squared_sums[2] = sqrt(squared_sums[2]);
	squared_sums[3] = sqrt(squared_sums[3]);
	//Normalize bbox_targets
	for (int i = 0; i < fg_inds.size(); i++) {

		int index = fg_inds[i];
		bbox_targets[index * 5 + 1] -= sums[0];
		bbox_targets[index * 5 + 2] -= sums[1];
		bbox_targets[index * 5 + 3] -= sums[2];
		bbox_targets[index * 5 + 4] -= sums[3];
		bbox_targets[index * 5 + 1] /= (squared_sums[0] + 0.000001);
		bbox_targets[index * 5 + 2] /= (squared_sums[1] + 0.000001);
		bbox_targets[index * 5 + 3] /= (squared_sums[2] + 0.000001);
		bbox_targets[index * 5 + 4] /= (squared_sums[3] + 0.000001);
	}
	image_roidb.bbox_targets = bbox_targets;
	image_roidb.bbox_targets_num = anchors_num_;
	//Compute values needed for means and stds
	delete[] sums;
	delete[] squared_sums;
	delete[] src_rois;
	delete[] target_rois;
	delete[] regression_label;
	delete[] ex_gt_overlaps;
	delete[] horizon_score;
	delete[] horizon_index;
}

void FastRcnnTrain::fast_rcnn_get_minibatch(DetectionConfig conf,
		ImageRoiDB image_roidb, float* &data_buf, float* &rois, float* &labels,
		float* &bbox_targets, float* &bbox_loss_weights, int& sample_num) {

	const int num_images = 1;
	const int num_classes = 1;
	int rois_per_image = conf.batch_size / num_images;
	int fg_rois_per_image = round(rois_per_image * conf.fg_fraction);
	float* im_rois;
	sample_rois(conf, image_roidb, fg_rois_per_image, rois_per_image, im_rois,
			bbox_targets, bbox_loss_weights, sample_num);
	fast_rcnn_map_im_rois_to_feat_rois(image_roidb.im_scales, im_rois,
			sample_num, rois);
	delete[] im_rois;

	labels = new float[sample_num]();
	for (int i = 0; i < sample_num; i++) {
		labels[i] = 1;
	}
	int width_resize = image_roidb.image0_resize.cols;
	int height_resize = image_roidb.image0_resize.rows;
	data_buf = new float[height_resize * width_resize * 3];
	for (int h = 0; h < height_resize; ++h) {
		for (int w = 0; w < width_resize; ++w) {
			data_buf[(0 * height_resize + h) * width_resize + w] = float(
					image_roidb.image0_resize.at<cv::Vec3f>(cv::Point(w, h))[0])
					- float(103.9390);
			data_buf[(1 * height_resize + h) * width_resize + w] = float(
					image_roidb.image0_resize.at<cv::Vec3f>(cv::Point(w, h))[1])
					- float(116.7790);
			data_buf[(2 * height_resize + h) * width_resize + w] = float(
					image_roidb.image0_resize.at<cv::Vec3f>(cv::Point(w, h))[2])
					- float(123.6800);

		}
	}
}
/**
 * anchors + keep -> rois; image_roidb.bbox_targets + ikeep_inds -> bbox_targets + bbox_loss_weights
 */
void FastRcnnTrain::sample_rois(DetectionConfig conf, ImageRoiDB image_roidb,
		int fg_rois_per_image, int rois_per_image, float* &rois,
		float* &bbox_targets, float* &bbox_loss_weights, int& sample_num) {

	//计算要保留的索引
	random_shuffle(fg_inds.begin(), fg_inds.end());
	random_shuffle(bg_inds.begin(), bg_inds.end());
	vector<int> keep_inds;
	vector<int> labels;
	int fg_rois_per_this_image = min(fg_rois_per_image,
			(int) fg_inds.size());
	int bg_rois_per_this_image = min(rois_per_image - fg_rois_per_this_image, (int) bg_inds.size());
	for (int i = 0; i < fg_rois_per_this_image; i++) {
		keep_inds.push_back(fg_inds[i]);
		labels.push_back(1);
	}
	for (int i = 0; i < bg_rois_per_this_image; i++) {
		keep_inds.push_back(bg_inds[i]);
		labels.push_back(0);
	}
	rois = new float[keep_inds.size() * 4]();
	for (int i = 0; i < keep_inds.size(); i++) {

		int index = keep_inds[i];
		rois[i * 4 + 0] = anchors[index * 4 + 0];
		rois[i * 4 + 1] = anchors[index * 4 + 1];
		rois[i * 4 + 2] = anchors[index * 4 + 2];
		rois[i * 4 + 3] = anchors[index * 4 + 3];
	}
	get_bbox_regression_labels(image_roidb.bbox_targets, keep_inds,
			bbox_targets, bbox_loss_weights);
	sample_num = keep_inds.size();
}

void FastRcnnTrain::fast_rcnn_map_im_rois_to_feat_rois(float im_scales,
		float* im_rois, int im_rois_num, float* &feat_rois) {

	feat_rois = new float[im_rois_num * 5]();
	for (int i = 0; i < im_rois_num; i++) {

		feat_rois[i * 5 + 0] = 0;
		feat_rois[i * 5 + 1] = round(im_rois[i * 4 + 0] * im_scales);
		feat_rois[i * 5 + 2] = round(im_rois[i * 4 + 1] * im_scales);
		feat_rois[i * 5 + 3] = round(im_rois[i * 4 + 2] * im_scales);
		feat_rois[i * 5 + 4] = round(im_rois[i * 4 + 3] * im_scales);
	}
}
/**
 *
 */
void FastRcnnTrain::get_bbox_regression_labels(float* bbox_targets_data,
		vector<int> keep_inds, float* &bbox_targets,
		float* &bbox_loss_weights) {

	const int num_classes = 1;
	bbox_targets = new float[keep_inds.size() * 4 * (num_classes + 1)]();
	bbox_loss_weights = new float[keep_inds.size() * 4 * (num_classes + 1)]();
	for (int i = 0; i < keep_inds.size(); i++) {

		int index = keep_inds[i];
		if(bbox_targets_data[index * 5 + 0] <= 0) continue;
		bbox_targets[i * 4 * (num_classes + 1) + 4] = bbox_targets_data[index
				* 5 + 1];
		bbox_targets[i * 4 * (num_classes + 1) + 5] = bbox_targets_data[index
				* 5 + 2];
		bbox_targets[i * 4 * (num_classes + 1) + 6] = bbox_targets_data[index
				* 5 + 3];
		bbox_targets[i * 4 * (num_classes + 1) + 7] = bbox_targets_data[index
				* 5 + 4];
		bbox_loss_weights[i * 4 * (num_classes + 1) + 4] = 1;
		bbox_loss_weights[i * 4 * (num_classes + 1) + 5] = 1;
		bbox_loss_weights[i * 4 * (num_classes + 1) + 6] = 1;
		bbox_loss_weights[i * 4 * (num_classes + 1) + 7] = 1;
	}
}

void FastRcnnTrain::init_config() {

	conf_.test_scales = 600;
	conf_.test_max_size = 1000;
	conf_.batch_size = 128;
	conf_.fg_fraction = 0.25;
	conf_.fg_thresh = 0.5;
	conf_.bg_thresh_lo = 0.1;
	conf_.bg_thresh_hi = 0.5;
	conf_.do_scale = true;
	conf_.bbox_thresh = 0.5;
}

void FastRcnnTrain::boxoverlap(float* ex_rois, int m, float* gt_rois, int n,
		float* ex_gt_overlaps) {

	for (int ex_index = 0; ex_index < m; ex_index++) {

		float ex_x1 = ex_rois[ex_index * 4 + 0];
		float ex_y1 = ex_rois[ex_index * 4 + 1];
		float ex_x2 = ex_rois[ex_index * 4 + 2];
		float ex_y2 = ex_rois[ex_index * 4 + 3];
		for (int gt_index = 0; gt_index < n; gt_index++) {

			float gt_x1 = gt_rois[gt_index * 4 + 0];
			float gt_y1 = gt_rois[gt_index * 4 + 1];
			float gt_x2 = gt_rois[gt_index * 4 + 2];
			float gt_y2 = gt_rois[gt_index * 4 + 3];

			float x1 = max(ex_x1, gt_x1);
			float y1 = max(ex_y1, gt_y1);
			float x2 = min(ex_x2, gt_x2);
			float y2 = min(ex_y2, gt_y2);
			float w = x2 - x1 + 1;
			float h = y2 - y1 + 1;
			float inter = w * h;
			float ex_area = (ex_x2 - ex_x1 + 1) * (ex_y2 - ex_y1 + 1);
			float gt_area = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1);
			float s = 0;
			if (w > 0 && h > 0) {
				s = inter / (ex_area + gt_area - inter);
			}
			ex_gt_overlaps[ex_index * n + gt_index] = s;
		}
	}
}

void FastRcnnTrain::fast_rcnn_bbox_transform(float* ex_boxes, float* gt_boxes,
		float* regression_label, int fg_num) {

	for (int i = 0; i < fg_num; i++) {

		float ex_widths = ex_boxes[i * 4 + 2] - ex_boxes[i * 4 + 0] + 1;
		float ex_heights = ex_boxes[i * 4 + 3] - ex_boxes[i * 4 + 1] + 1;
		float ex_ctr_x = ex_boxes[i * 4 + 0] + 0.5 * (ex_widths - 1);
		float ex_ctr_y = ex_boxes[i * 4 + 1] + 0.5 * (ex_heights - 1);

		float gt_widths = gt_boxes[i * 4 + 2] - gt_boxes[i * 4 + 0] + 1;
		float gt_heights = gt_boxes[i * 4 + 3] - gt_boxes[i * 4 + 1] + 1;
		float gt_ctr_x = gt_boxes[i * 4 + 0] + 0.5 * (gt_widths - 1);
		float gt_ctr_y = gt_boxes[i * 4 + 1] + 0.5 * (gt_heights - 1);

		float eps = 0.000001;
		float targets_dx = (gt_ctr_x - ex_ctr_x) / (ex_widths + eps);
		float targets_dy = (gt_ctr_y - ex_ctr_y) / (ex_heights + eps);
		float targets_dw = log(gt_widths / (ex_widths + eps));
		float targets_dh = log(gt_heights / (ex_heights + eps));

		regression_label[i * 4 + 0] = targets_dx;
		regression_label[i * 4 + 1] = targets_dy;
		regression_label[i * 4 + 2] = targets_dw;
		regression_label[i * 4 + 3] = targets_dh;
	}
}

void FastRcnnTrain::set_anchors(float* anchors_data, int num){

	anchors = new float[num*4];
	for(int i=0;i<num*4;i++){

		anchors[i] = anchors_data[i];
	}
	anchors_num_ = num;
}

void FastRcnnTrain::free(){

	anchors_num_ = 0;
	delete[] anchors;
	delete[] vertical_index;
	delete[] vertical_score;
}

void FastRcnnTrain::save_mode(string name){

	caffe::NetParameter net_param;
	solver_->net()->ToProto(&net_param, false);
	caffe::WriteProtoToBinaryFile(net_param, name);
}

void FastRcnnTrain::show_accuarcy(){

	const float* accuarcy_data = solver_->net()->blob_by_name(
				"accuarcy")->cpu_data();
		int accuarcy_height =
				solver_->net()->blob_by_name("accuarcy")->height();
		int accuarcy_width =
				solver_->net()->blob_by_name("accuarcy")->width();
		int accuarcy_channels = solver_->net()->blob_by_name(
				"accuarcy")->channels();
		int accuarcy_num =
				solver_->net()->blob_by_name("accuarcy")->num();
	float sum0 = 0;
	for(int i=0; i<accuarcy_height*accuarcy_width*accuarcy_channels*accuarcy_num; i++){
		sum0 += accuarcy_data[i];
	}
	cout<<"accuarcy = "<<sum0<<endl;
}
//void FastRcnnTrain::set_feature_size(int feature_width, int feature_height){
//
//	//feature_width_ = feature_width;
//	//feature_height_ = feature_height;
//}
//
//void FastRcnnTrain::calc_feature_size(int image_width, int image_height){
//
//	FasterRCNN faster_rcnn(0,true);
//	faster_rcnn.initProposalDetectionModel("");
//	faster_rcnn.initRPN_NET(conf_.test_net_def_file, "");
//	cv::Mat src(image_height, image_width, CV_8UC3, cv::Scalar(128, 128, 128));
//	faster_rcnn.proposal_im_detect(src);
//	//feature_height_ = faster_rcnn.getFeatureHeight();
//	//feature_width_ = faster_rcnn.getFeatureWidth();;
//	faster_rcnn.release();
//}
