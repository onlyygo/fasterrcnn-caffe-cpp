/*
 * ProposalTrain.cpp
 *
 *  Created on: Aug 27, 2016
 *      Author: cudahe
 */

#include "ProposalTrain.h"
#include "FasterRCNN.h"
using namespace std;

static void print_float(float* bu, int rows, int cols) {

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {

			cout << bu[i * cols + j] << "    ";
		}
		cout << endl;
	}
}

ProposalTrain::ProposalTrain(const string& solver_file,
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

	caffe::shared_ptr<caffe::Solver<float> > solver(
			new caffe::SGDSolver<float>(solver_file));
	solver_ = solver;
	solver_->net()->CopyTrainedLayersFrom(weights_file);
	test_net_def_file_ = test_net_def_file;
	init_config();
	feature_height_ = 0;
	feature_width_ = 0;
}

ProposalTrain::~ProposalTrain() {
	// TODO Auto-generated destructor stub
}

void ProposalTrain::proposal_train(ImageRoiDB image_roidb) {

	//第零步：将图像resize，并记录resize的因子
	prep_im_for_blob(image_roidb.image0, image_roidb.image0_resize,
			image_roidb.im_scales, conf_);
	//第一步：计算featuresize
	calc_feature_size(image_roidb.image0_resize.cols,
			image_roidb.image0_resize.rows);
	//第二步：计算bbox_targets，这是一个m行5列的矩阵，m为特征图size×9，第一列代表目标的类别，正数为类的id，负数为负样本，0为不参与样本
	proposal_prepare_image_roidb(conf_, image_roidb);
	//input_blobs = {im_blob, labels_blob, labels_weights_blob, bbox_targets_blob, bbox_loss_blob};
	float *data_buf;
	float* labels; //M*1
	float* labels_weights; //M*1
	float* bbox_targets; //M*4
	float* bbox_loss_weights; //M*4
	proposal_generate_minibatch(conf_, image_roidb, data_buf, labels,
			labels_weights, bbox_targets, bbox_loss_weights); // new5 need delete
	//training
	solver_->net()->SetPhase(TRAIN);
	solver_->net()->blob_by_name("data")->Reshape(1, 3,
			image_roidb.image0_resize.rows, image_roidb.image0_resize.cols);
	solver_->net()->blob_by_name("labels")->Reshape(1, 9, feature_height_,
			feature_width_);
	solver_->net()->blob_by_name("labels_weights")->Reshape(1, 9,
			feature_height_, feature_width_);
	solver_->net()->blob_by_name("bbox_targets")->Reshape(1, 36,
			feature_height_, feature_width_);
	solver_->net()->blob_by_name("bbox_loss_weights")->Reshape(1, 36,
			feature_height_, feature_width_);
	solver_->net()->Reshape();
	solver_->net()->blob_by_name("data")->set_cpu_data(data_buf);
	solver_->net()->blob_by_name("labels")->set_cpu_data(labels);
	solver_->net()->blob_by_name("labels_weights")->set_cpu_data(
			labels_weights);
	solver_->net()->blob_by_name("bbox_targets")->set_cpu_data(bbox_targets);
	solver_->net()->blob_by_name("bbox_loss_weights")->set_cpu_data(
			bbox_loss_weights);
	solver_->Step(2);

	delete[] data_buf;
	delete[] labels_weights;
	delete[] labels;
	delete[] bbox_targets;
	delete[] bbox_loss_weights;
	delete[] image_roidb.bbox_targets;
}

void ProposalTrain::prep_im_for_blob(cv::Mat image0, cv::Mat& image0_resize,
		float& im_scale, ProposalConfig conf) {

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

void ProposalTrain::proposal_prepare_image_roidb(ProposalConfig conf,
		ImageRoiDB &image_roidb) {

	const int num_images = 1;
	int feature_width = feature_width_;
	int feature_height = feature_height_;
	int m = feature_width * feature_height * 9;
	//计算所有候选样本anchors
	float* anchors = new float[feature_width * feature_height * 9 * 4];
	proposal_locate_anchors(feature_width, feature_height, anchors, conf);
	//计算每一个样本与每一个gt的重叠度
	float* ex_gt_overlaps = new float[m * image_roidb.boxes_num];
	//对boxes做resize
	float* scale_boxes = new float[image_roidb.boxes_num * 4]();
	for (int i = 0; i < image_roidb.boxes_num * 4; i++) {

		scale_boxes[i] = image_roidb.boxes[i] * image_roidb.im_scales;
	}
	boxoverlap(anchors, m, scale_boxes, image_roidb.boxes_num, ex_gt_overlaps);

	//计算重叠度的行最大值和列最大值，以及取得最大值的索引
	//m行1列，每个元素对应该取行的最大值的索引
	int* vertical_index = new int[m]();
	//m行1列，每个元素对应该行的最大值
	float* vertical_score = new float[m]();
	//1行n列，每个元素对应该列取得最大值的索引
	int* horizon_index = new int[image_roidb.boxes_num]();
	//1行n列，每个元素对应该列的最大值
	float* horizon_score = new float[image_roidb.boxes_num]();

	for (int ex_index = 0; ex_index < m; ex_index++) {

		//将位于图外anchor的overlap全部置-1
		float ex_x1 = anchors[ex_index * 4 + 0];
		float ex_y1 = anchors[ex_index * 4 + 1];
		float ex_x2 = anchors[ex_index * 4 + 2];
		float ex_y2 = anchors[ex_index * 4 + 3];
		if (ex_x1 < 0 || ex_y1 < 0 || ex_x2 < 0 || ex_y2 < 0
				|| ex_x1 >= image_roidb.image0_resize.cols
				|| ex_y1 >= image_roidb.image0_resize.rows
				|| ex_x2 >= image_roidb.image0_resize.cols
				|| ex_y2 >= image_roidb.image0_resize.rows) {

			for(int i=0;i<image_roidb.boxes_num;i++){
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
	vector<int> fg_inds;
	vector<int> bg_inds;
	for (int ex_index = 0; ex_index < m; ex_index++) {

		bool is_fg = false;
		for(int i=0;i<image_roidb.boxes_num;i++){
			if(ex_gt_overlaps[ex_index * image_roidb.boxes_num + i] >= conf.fg_thresh \
					|| ex_gt_overlaps[ex_index * image_roidb.boxes_num + i] == horizon_score[i]){
				is_fg = true;
				break;
			}
		}
		if(is_fg){

			fg_inds.push_back(ex_index);
			continue;
		}
		int bg_count = 0;
		for(int i=0;i<image_roidb.boxes_num;i++){
			if(ex_gt_overlaps[ex_index * image_roidb.boxes_num + i] >= conf.bg_thresh_lo \
					&& ex_gt_overlaps[ex_index * image_roidb.boxes_num + i] < conf.bg_thresh_hi){
				bg_count++;
			}
		}
		if(bg_count == image_roidb.boxes_num){
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
		target_rois[i * 4 + 0] = scale_boxes[vertical_index[index] * 4 + 0];
		target_rois[i * 4 + 1] = scale_boxes[vertical_index[index] * 4 + 1];
		target_rois[i * 4 + 2] = scale_boxes[vertical_index[index] * 4 + 2];
		target_rois[i * 4 + 3] = scale_boxes[vertical_index[index] * 4 + 3];
	}
	//计算回归标签
	fast_rcnn_bbox_transform(src_rois, target_rois, regression_label,
			fg_inds.size());
	//得到候选的bbox_targets m*5
	float* bbox_targets = new float[m * 5]();
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
	for (int i = 0; i < bg_inds.size(); i++) {

		int index = bg_inds[i];
		bbox_targets[index * 5 + 0] = -1;
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
	image_roidb.bbox_targets_num = m;
	//Compute values needed for means and stds
	delete[] scale_boxes;
	delete[] sums;
	delete[] squared_sums;
	delete[] src_rois;
	delete[] target_rois;
	delete[] regression_label;
	delete[] anchors;
	delete[] ex_gt_overlaps;
	delete[] vertical_score;
	delete[] horizon_score;
	delete[] vertical_index;
	delete[] horizon_index;
}

void ProposalTrain::proposal_generate_minibatch(ProposalConfig conf,
		ImageRoiDB image_roidb, float* &data_buf, float* &labels,
		float* &labels_weights, float* &bbox_targets,
		float* &bbox_loss_weights) {

	int num_images = 1;
	int rois_per_image = conf.batch_size / num_images;
	int fg_rois_per_image = round(rois_per_image * conf.fg_fraction);
	//正负样本采样 : sample_rois
	sample_rois(conf, image_roidb, fg_rois_per_image, rois_per_image,
			image_roidb.im_scales, labels, labels_weights, bbox_targets,
			bbox_loss_weights);
	//reshape
	int width_resize = image_roidb.image0_resize.cols;
	int height_resize = image_roidb.image0_resize.rows;
	//通过map找到，这样尺寸输入对应的特征图尺寸
	//int feature_width = conf.output_width_map[im_blob.cols];
	//int feature_height = conf.output_height_map[im_blob.rows];
	//labels logic shape[9, feature_height, feature_width] 内存排列不变
	//labels_weights logic shape[9, feature_height, feature_width] 内存排列不变
	//bbox_targets 内存排列不变
	//bbox_loss 内存排列不变
	//图像内存修正
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

void ProposalTrain::proposal_locate_anchors(int feature_width,
		int feature_height, float* anchors, ProposalConfig conf_proposal) {

	const int SIZE = feature_width * feature_height;

	for (int c = 0; c < SIZE; c++) {

		for (int r = 0; r < ANCHOR_NUM; r++) {

			for (int n = 0; n < ANCHOR_DIM; n++) {

				float temp = 0;
				if (n % 2 == 0) {

					temp = (c - 1) / feature_height * conf_proposal.feat_stride;
				} else {

					//c = 0 36 ...=0
					//c = 1 37 ...=16

					//c = 35 71...=35*16=560
					temp = (c % feature_height) * conf_proposal.feat_stride;
				}
				temp += ANCHORS[r][n];
				anchors[c * ANCHOR_NUM * ANCHOR_DIM + r * ANCHOR_DIM + n] = temp;
			}
		}
	}
}

void ProposalTrain::boxoverlap(float* ex_rois, int m, float* gt_rois, int n,
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

void ProposalTrain::fast_rcnn_bbox_transform(float* ex_boxes, float* gt_boxes,
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

void ProposalTrain::sample_rois(ProposalConfig conf, ImageRoiDB image_roidb,
		int fg_rois_per_image, int rois_per_image, float im_scales,
		float* &labels, float* &labels_weights, float* &bbox_targets,
		float* &bbox_loss_weights) {

	int bbox_targets_num = image_roidb.bbox_targets_num;

	labels = new float[bbox_targets_num]();		//init for zero and for return
	labels_weights = new float[bbox_targets_num]();	//init for zero and for return
	bbox_targets = new float[bbox_targets_num * 4]();			//m*4,for return
	bbox_loss_weights = new float[bbox_targets_num * 4]();		//m*4,for return

	vector<int> fg_inds, bg_inds;
	//init
	for (int i = 0; i < bbox_targets_num; i++) {

		float* bb_buffer = &(image_roidb.bbox_targets[i * 5 + 0]);		//m*5

		bbox_targets[i * 4 + 0] = bb_buffer[1];
		bbox_targets[i * 4 + 1] = bb_buffer[2];
		bbox_targets[i * 4 + 2] = bb_buffer[3];
		bbox_targets[i * 4 + 3] = bb_buffer[4];

		labels[i] = 0;
		labels_weights[i] = 0;
		bbox_loss_weights[i * 4 + 0] = 0;
		bbox_loss_weights[i * 4 + 1] = 0;
		bbox_loss_weights[i * 4 + 2] = 0;
		bbox_loss_weights[i * 4 + 3] = 0;

		if (bb_buffer[0] > 0) {					//正样本

			fg_inds.push_back(i);
		} else if (bb_buffer[0] < 0) {					//负样本

			bg_inds.push_back(i);
		}
	}
	//执行正负样本选择
	int fg_num = min(fg_rois_per_image, (int) fg_inds.size());
	int bg_num = min(rois_per_image - fg_num, (int) bg_inds.size());
	//随机打算顺序
	random_shuffle(fg_inds.begin(), fg_inds.end());
	random_shuffle(bg_inds.begin(), bg_inds.end());

	for (int i = 0; i < fg_num; i++) {

		int index = fg_inds[i];
		float* bb_buffer = &(image_roidb.bbox_targets[index * 5 + 0]);	//m*5
		labels[index] = bb_buffer[0];
		labels_weights[index] = 1;
		bbox_loss_weights[index * 4 + 0] = 1;
		bbox_loss_weights[index * 4 + 1] = 1;
		bbox_loss_weights[index * 4 + 2] = 1;
		bbox_loss_weights[index * 4 + 3] = 1;
	}
	for (int i = 0; i < bg_num; i++) {

		int index = bg_inds[i];
		labels_weights[index] = conf.bg_weight;
	}
}

int ProposalTrain::init_config() {

	conf_.test_scales = 600;
	conf_.test_max_size = 1000;
	conf_.batch_size = 256;
	conf_.fg_fraction = 0.5;
	conf_.bg_weight = 1;
	conf_.do_scale = true;
	conf_.feat_stride = 16;
	conf_.fg_thresh = 0.7;
	conf_.bg_thresh_lo = 0;
	conf_.bg_thresh_hi = 0.3;
}

void ProposalTrain::set_feature_size(int feature_width, int feature_height) {

	feature_height_ = feature_height;
	feature_width_ = feature_width;
}

void ProposalTrain::calc_feature_size(int image_width, int image_height) {

	FasterRCNN faster_rcnn(1, true);
	faster_rcnn.initProposalDetectionModel("");
	faster_rcnn.initRPN_NET(test_net_def_file_, "");
	cv::Mat src(image_height, image_width, CV_8UC3, cv::Scalar(128, 128, 128));
	faster_rcnn.proposal_im_detect(src);
	feature_height_ = faster_rcnn.getFeatureHeight();
	feature_width_ = faster_rcnn.getFeatureWidth();
	faster_rcnn.release();
}

void ProposalTrain::save_mode(string name){

	caffe::NetParameter net_param;
	solver_->net()->ToProto(&net_param, false);
	caffe::WriteProtoToBinaryFile(net_param, name);
}

void ProposalTrain::show_accuarcy(){

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
