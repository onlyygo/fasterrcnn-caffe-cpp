#include "FasterRCNN.h"

FasterRCNN::FasterRCNN() {

	proposal_boxes = NULL;
	proposal_scores = NULL;
	detection_boxes = NULL;
	detection_scores = NULL;
}

FasterRCNN::~FasterRCNN() {
	this->release();
}

void FasterRCNN::release() {

	if (proposal_boxes != NULL) {

		delete[] proposal_boxes;
		proposal_boxes = NULL;
	}

	if (proposal_scores != NULL) {

		delete[] proposal_scores;
		proposal_scores = NULL;
	}

	if (detection_boxes != NULL) {

		delete[] detection_boxes;
		detection_boxes = NULL;
	}

	if (detection_scores != NULL) {

		delete[] detection_scores;
		detection_scores = NULL;
	}

}

void FasterRCNN::initRPN_NET(const string& model_file,
		const string& weights_file) {

	rpn_net = boost::shared_ptr<Net<float> >(
			new Net<float>(model_file, caffe::TEST));
	if (weights_file.size())
		rpn_net->CopyTrainedLayersFrom(weights_file);
}

void FasterRCNN::initFastRCNN_NET(const string& model_file,
		const string& weights_file) {

	fast_rcnn_net = boost::shared_ptr<Net<float> >(
			new Net<float>(model_file, caffe::TEST));
	fast_rcnn_net->CopyTrainedLayersFrom(weights_file);
}

void FasterRCNN::proposal_im_detect(cv::Mat image0) {

	const int BOX_DIMS = 4;

	input_width = image0.cols;
	input_height = image0.rows;
	//�����ڴ�Ǽ�
	float *data_buf = NULL;
	float* box_deltas = NULL;
	float* anchors = NULL;
	float* pred_boxes = NULL;
	float* pred_scores = NULL;
	float* boxes_nms = NULL;
	float* scores_nms = NULL;
	float* temp = NULL;
	//numϵ��
	int num;
	int valid_num;
	int top_num;
	int pick_num;
	//precess����ȥ��ֵ��resize
	cv::Mat image0_resize;
	float im_scale;
	prep_im_for_blob(image0, image0_resize, im_scale);
	scale = im_scale;
	//׼���������
	int height_resize = image0_resize.rows;
	int width_resize = image0_resize.cols;
	data_buf = new float[height_resize * width_resize * 3];
	for (int h = 0; h < height_resize; ++h) {
		for (int w = 0; w < width_resize; ++w) {
			data_buf[(0 * height_resize + h) * width_resize + w] = float(
					image0_resize.at<cv::Vec3f>(cv::Point(w, h))[0])
					- float(103.9390);
			data_buf[(1 * height_resize + h) * width_resize + w] = float(
					image0_resize.at<cv::Vec3f>(cv::Point(w, h))[1])
					- float(116.7790);
			data_buf[(2 * height_resize + h) * width_resize + w] = float(
					image0_resize.at<cv::Vec3f>(cv::Point(w, h))[2])
					- float(123.6800);
		}
	}

	//�������
	rpn_net->blob_by_name(cfg.data_layer)->Reshape(1, 3, height_resize, width_resize);
	rpn_net->Reshape();
	rpn_net->blob_by_name(cfg.data_layer)->set_cpu_data(data_buf);
	rpn_net->ForwardFrom(0);
	//�õ�������
	const float* proposal_bbox_pred_data = rpn_net->blob_by_name(
			cfg.proposal_bbox_layer)->cpu_data();
	int proposal_bbox_pred_height =
			rpn_net->blob_by_name(cfg.proposal_bbox_layer)->height();
	int proposal_bbox_pred_width =
			rpn_net->blob_by_name(cfg.proposal_bbox_layer)->width();
	int proposal_bbox_pred_channels = rpn_net->blob_by_name(
			cfg.proposal_bbox_layer)->channels();
	int proposal_bbox_pred_num =
			rpn_net->blob_by_name(cfg.proposal_bbox_layer)->num();

	int feature_width = proposal_bbox_pred_width;
	int feature_height = proposal_bbox_pred_height;
	feature_width_ = feature_width;
	feature_height_ = feature_height;
	//����num
	num = ANCHOR_NUM * feature_width * feature_height;
	//����������proposal_bbox_pred_data
	box_deltas = new float[num * BOX_DIMS];
	int index = 0;
	for (int width_index = 0; width_index < proposal_bbox_pred_width;
			width_index++) {

		for (int height_index = 0; height_index < proposal_bbox_pred_height;
				height_index++) {

			for (int channels_index = 0;
					channels_index < proposal_bbox_pred_channels;
					channels_index++) {

				//w h c
				box_deltas[index++] =
						proposal_bbox_pred_data[channels_index
								* proposal_bbox_pred_width
								* proposal_bbox_pred_height
								+ height_index * proposal_bbox_pred_width
								+ width_index];
			}
		}
	}
	//����anchors
	const int SIZE = feature_width * feature_height;
	anchors = new float[ANCHOR_NUM * SIZE * ANCHOR_DIM];	//(9, 2268, 4)
	proposal_locate_anchors(feature_width, feature_height, anchors);
	//����pred_boxes
	pred_boxes = new float[num * BOX_DIMS];
	fast_rcnn_bbox_transform_inv(box_deltas, anchors, pred_boxes, num);
	//// scale back
	//for (int i = 0; i<num * BOX_DIMS; i++){

	//	pred_boxes[i] /= im_scale;
	//}

	clip_boxes(pred_boxes, width_resize, height_resize, num);

	//�����һ�����
	const float* proposal_cls_prob_data = rpn_net->blob_by_name(
			cfg.proposal_cls_layer)->cpu_data();
	int proposal_cls_prob_height =
			rpn_net->blob_by_name(cfg.proposal_cls_layer)->height();
	int proposal_cls_prob_width =
			rpn_net->blob_by_name(cfg.proposal_cls_layer)->width();
	int proposal_cls_prob_channels =
			rpn_net->blob_by_name(cfg.proposal_cls_layer)->channels();
	int proposal_cls_prob_num =
			rpn_net->blob_by_name(cfg.proposal_cls_layer)->num();
	//����������proposal_cls_prob
	//step1
	temp = new float[proposal_cls_prob_height * proposal_cls_prob_width];//63��36��9����
	for (int i = 0; i < proposal_cls_prob_height * proposal_cls_prob_width;
			i++) {

		temp[i] = proposal_cls_prob_data[proposal_cls_prob_height
				* proposal_cls_prob_width + i];
	}
	pred_scores = new float[proposal_cls_prob_height * proposal_cls_prob_width];
	index = 0;
	int width_reshape = proposal_cls_prob_width;
	int height_reshape = feature_height;
	int channels_reshape = proposal_cls_prob_height / height_reshape;
	for (int width_index = 0; width_index < width_reshape; width_index++) {

		for (int height_index = 0; height_index < height_reshape;
				height_index++) {

			for (int channels_index = 0; channels_index < channels_reshape;
					channels_index++) {

				//w h c
				pred_scores[index++] = temp[channels_index * height_reshape
						* width_reshape + height_index * width_reshape
						+ width_index];
			}
		}
	}
	//ȥ����Ч��box��Ȼ������
	filter_boxes_sort(pred_boxes, pred_scores, num, valid_num);

	//ȡ��ǰtop�����box
	if (cfg.per_nms_topN > 0)
		top_num = min(valid_num, cfg.per_nms_topN);

	//��box���ظ���ȥ��
	nms<float>(pred_boxes, pred_scores, top_num, cfg.nms_overlap_thresint,
			pick_num, boxes_nms, scores_nms);

	//ȡ��ʣ���ǰtop��

	if (cfg.after_nms_topN > 0)
		proposal_num = min(pick_num, cfg.after_nms_topN);

	proposal_boxes = new float[proposal_num * BOX_DIMS];
	proposal_scores = new float[proposal_num];
	for (int i = 0; i < proposal_num; i++) {

		proposal_boxes[i * BOX_DIMS + 0] = boxes_nms[i * BOX_DIMS + 0];
		proposal_boxes[i * BOX_DIMS + 1] = boxes_nms[i * BOX_DIMS + 1];
		proposal_boxes[i * BOX_DIMS + 2] = boxes_nms[i * BOX_DIMS + 2];
		proposal_boxes[i * BOX_DIMS + 3] = boxes_nms[i * BOX_DIMS + 3];
		proposal_scores[i] = scores_nms[i];
	}

	//�ͷ���ʱ�ڴ�
	delete[] temp;
	delete[] boxes_nms;
	delete[] scores_nms;
	delete[] pred_scores;
	delete[] pred_boxes;
	delete[] anchors;
	delete[] box_deltas;
	delete[] data_buf;
}

void FasterRCNN::prep_im_for_blob(cv::Mat image0, cv::Mat& image0_resize,
		float& im_scale) {

	cv::Mat image0_src;
	image0.convertTo(image0_src, CV_32FC3);

	if (this->cfg.do_scale) {
		//������������
		int im_size_min = min(image0_src.rows, image0_src.cols);
		int im_size_max = max(image0_src.rows, image0_src.cols);

		im_scale = min(float(cfg.test_scales) / im_size_min,
				float(cfg.test_max_size) / im_size_max);

		int new_rows = round(image0_src.rows * im_scale);
		int new_cols = round(image0_src.cols * im_scale);
		cv::resize(image0_src, image0_resize, cv::Size(new_cols, new_rows), 0,
				0, CV_INTER_LINEAR);
	} else {

		im_scale = 1;
		image0_src.copyTo(image0_resize);
	}
}

void FasterRCNN::proposal_locate_anchors(int feature_width, int feature_height,
		float* anchors) {

	const int SIZE = feature_width * feature_height;

	for (int c = 0; c < SIZE; c++) {

		for (int r = 0; r < ANCHOR_NUM; r++) {

			for (int n = 0; n < ANCHOR_DIM; n++) {

				float temp = 0;
				if (n % 2 == 0) {

					temp = (c - 1) / feature_height * cfg.feat_stride;
				} else {

					//c = 0 36 ...=0
					//c = 1 37 ...=16

					//c = 35 71...=35*16=560
					temp = (c % feature_height) * cfg.feat_stride;
				}
				temp += ANCHORS[r][n];
				anchors[c * ANCHOR_NUM * ANCHOR_DIM + r * ANCHOR_DIM + n] = temp;
			}
		}
	}
}

void FasterRCNN::fast_rcnn_bbox_transform_inv(const float* box_deltas,
		float* anchors, float* pred_boxes, int num) {
	const int BOX_DIMS = 4;
	for (int i = 0; i < num; i++) {

		float src_w = anchors[i * BOX_DIMS + 2] - anchors[i * BOX_DIMS + 0] + 1;
		float src_h = anchors[i * BOX_DIMS + 3] - anchors[i * BOX_DIMS + 1] + 1;
		float src_ctr_x = float(anchors[i * BOX_DIMS + 0] + 0.5 * (src_w - 1));
		float src_ctr_y = float(anchors[i * BOX_DIMS + 1] + 0.5 * (src_h - 1));

		float dst_ctr_x = float(box_deltas[i * BOX_DIMS + 0]);
		float dst_ctr_y = float(box_deltas[i * BOX_DIMS + 1]);
		float dst_scl_x = float(box_deltas[i * BOX_DIMS + 2]);
		float dst_scl_y = float(box_deltas[i * BOX_DIMS + 3]);

		float pred_ctr_x = dst_ctr_x * src_w + src_ctr_x;
		float pred_ctr_y = dst_ctr_y * src_h + src_ctr_y;
		float pred_w = exp(dst_scl_x) * src_w;
		float pred_h = exp(dst_scl_y) * src_h;
		pred_boxes[i * BOX_DIMS + 0] = pred_ctr_x - 0.5 * (pred_w - 1);
		pred_boxes[i * BOX_DIMS + 1] = pred_ctr_y - 0.5 * (pred_h - 1);
		pred_boxes[i * BOX_DIMS + 2] = pred_ctr_x + 0.5 * (pred_w - 1);
		pred_boxes[i * BOX_DIMS + 3] = pred_ctr_y + 0.5 * (pred_h - 1);
	}
}

void FasterRCNN::clip_boxes(float* pred_boxes, int im_width, int im_height,
		int num) {

	const int BOX_DIMS = 4;
	for (int i = 0; i < num; i++) {

		pred_boxes[i * BOX_DIMS + 0] = max(
				min(pred_boxes[i * BOX_DIMS + 0], (float) im_width), 0.f);
		pred_boxes[i * BOX_DIMS + 1] = max(
				min(pred_boxes[i * BOX_DIMS + 1], (float) im_height), 0.f);
		pred_boxes[i * BOX_DIMS + 2] = max(
				min(pred_boxes[i * BOX_DIMS + 2], (float) im_width), 0.f);
		pred_boxes[i * BOX_DIMS + 3] = max(
				min(pred_boxes[i * BOX_DIMS + 3], (float) im_height), 0.f);
	}
}

void FasterRCNN::filter_boxes_sort(float* pred_boxes, float* pred_scores,
		int num, int &valid_num) {

	const int BOX_DIMS = 4;
	valid_num = num;
	for (int i = 0; i < num; i++) {

		int widths = pred_boxes[i * BOX_DIMS + 2] - pred_boxes[i * BOX_DIMS + 0]
				+ 1;
		int heights = pred_boxes[i * BOX_DIMS + 3]
				- pred_boxes[i * BOX_DIMS + 1] + 1;
		if (widths < cfg.test_min_box_size
				|| heights < cfg.test_min_box_size
				|| pred_scores[i] == 0) {

			pred_scores[i] = 0;
			valid_num--;
		}
	}
	//sort by pred_scores
	//��С����
	for (int findMin = 0; findMin < num; findMin++) {

		for (int i = 0; i < num - 1 - findMin; i++) {

			float pro = pred_scores[i];
			float next = pred_scores[i + 1];
			if (pro < next) {

				pred_scores[i + 1] = pro;
				pred_scores[i] = next;

				//pred_boxes 
				for (int j = 0; j < BOX_DIMS; j++) {
					int temp = pred_boxes[i * BOX_DIMS + j];
					pred_boxes[i * BOX_DIMS + j] = pred_boxes[(i + 1) * BOX_DIMS
							+ j];
					pred_boxes[(i + 1) * BOX_DIMS + j] = temp;
				}
			}
		}
	}
}

void FasterRCNN::fast_rcnn_im_detect(cv::Mat image0) {

	const int BOX_DIMS = 4;
	//�ڴ�����Ǽ�
	float* data_buf = NULL;
	float* rois = NULL;
	float* pred_boxes = NULL;
	float* pred_scores = NULL;

	//׼��������� - ͼ�����
	cv::Mat image0_resize;
	float im_scale;
	prep_im_for_blob(image0, image0_resize, im_scale);
	scale = im_scale;
	//im_list_to_blob����Ӱ����Ҫ

	int height_resize = image0_resize.rows;
	int width_resize = image0_resize.cols;
	data_buf = new float[height_resize * width_resize * 3];
	for (int h = 0; h < height_resize; ++h) {
		for (int w = 0; w < width_resize; ++w) {
			data_buf[(0 * height_resize + h) * width_resize + w] = float(
					image0_resize.at<cv::Vec3f>(cv::Point(w, h))[0])
					- float(103.9390);
			data_buf[(1 * height_resize + h) * width_resize + w] = float(
					image0_resize.at<cv::Vec3f>(cv::Point(w, h))[1])
					- float(116.7790);
			data_buf[(2 * height_resize + h) * width_resize + w] = float(
					image0_resize.at<cv::Vec3f>(cv::Point(w, h))[2])
					- float(123.6800);

		}
	}
	const int ROI_DIMS = (BOX_DIMS + 1);
	rois = new float[proposal_num * ROI_DIMS];

	//ROI��ݵ���
	for (int i = 0; i < proposal_num; i++) {

		//����һλ
		rois[i * ROI_DIMS + 4] = proposal_boxes[i * BOX_DIMS + 3];
		rois[i * ROI_DIMS + 3] = proposal_boxes[i * BOX_DIMS + 2];
		rois[i * ROI_DIMS + 2] = proposal_boxes[i * BOX_DIMS + 1];
		rois[i * ROI_DIMS + 1] = proposal_boxes[i * BOX_DIMS + 0];
		rois[i * ROI_DIMS + 0] = 0;
	}

	//�������������
	fast_rcnn_net->blob_by_name(cfg.data_layer)->Reshape(1, image0_resize.channels(),
			height_resize, width_resize);
	fast_rcnn_net->blob_by_name(cfg.rois_layer)->Reshape(proposal_num, 5, 1, 1);
	fast_rcnn_net->Reshape();

	fast_rcnn_net->blob_by_name(cfg.data_layer)->set_cpu_data(data_buf);
	fast_rcnn_net->blob_by_name(cfg.rois_layer)->set_cpu_data(rois);
	fast_rcnn_net->ForwardFrom(0);

	const float* bbox_pred_data =
			fast_rcnn_net->blob_by_name(cfg.detection_bbox_layer)->cpu_data();//�ȼ���8��300��
	int bbox_pred_height = fast_rcnn_net->blob_by_name(cfg.detection_bbox_layer)->height();
	int bbox_pred_width = fast_rcnn_net->blob_by_name(cfg.detection_bbox_layer)->width();
	int bbox_pred_channels =
			fast_rcnn_net->blob_by_name(cfg.detection_bbox_layer)->channels();
	int bbox_pred_num = fast_rcnn_net->blob_by_name(cfg.detection_bbox_layer)->num();
	//bbox_pred_data�����ȡ������
	precess_bbox_pred(bbox_pred_data, bbox_pred_num, bbox_pred_channels, bbox_pred_width, bbox_pred_height, pred_boxes);

	const float* cls_prob_data =
			fast_rcnn_net->blob_by_name(cfg.detection_cls_layer)->cpu_data();//�ȼ���2��300��
	int cls_prob_height = fast_rcnn_net->blob_by_name(cfg.detection_cls_layer)->height();
	int cls_prob_width = fast_rcnn_net->blob_by_name(cfg.detection_cls_layer)->width();
	int cls_prob_channels = fast_rcnn_net->blob_by_name(cfg.detection_cls_layer)->channels();
	int cls_prob_num = fast_rcnn_net->blob_by_name(cfg.detection_cls_layer)->num();
	precess_cls_prob(cls_prob_data, cls_prob_num, cls_prob_channels, cls_prob_width, cls_prob_width, pred_scores);

	//���
	nms<float>(pred_boxes, pred_scores, bbox_pred_num,
			cfg.nms_overlap_thresint2, detection_num, detection_boxes,
			detection_scores);

	//�ͷ��ڴ�
	delete[] data_buf;
	delete[] rois;
	delete[] pred_boxes;
	delete[] pred_scores;
}

void FasterRCNN::fast_rcnn_conv_feat_detect() {

	const int BOX_DIMS = 4;

	//�ڴ�����Ǽ�
	float* rois = NULL;
	float* data_buf = NULL;

	float* pred_boxes = NULL;
	float* pred_scores = NULL;

	////׼��������� - data���
	const float* conv_feat_blob_data = rpn_net->blob_by_name(
			cfg.shared_layer)->cpu_data();
	int conv_feat_blob_data_height = rpn_net->blob_by_name(
			cfg.shared_layer)->height();
	int conv_feat_blob_data_width = rpn_net->blob_by_name(
			cfg.shared_layer)->width();
	int conv_feat_blob_data_channels = rpn_net->blob_by_name(
			cfg.shared_layer)->channels();
	int conv_feat_blob_data_num = rpn_net->blob_by_name(
			cfg.shared_layer)->num();
	data_buf = new float[conv_feat_blob_data_height * conv_feat_blob_data_width
			* conv_feat_blob_data_channels * conv_feat_blob_data_num];
	memcpy(data_buf, conv_feat_blob_data,
			sizeof(float) * conv_feat_blob_data_height
					* conv_feat_blob_data_width * conv_feat_blob_data_channels
					* conv_feat_blob_data_num);

	const int ROI_DIMS = (BOX_DIMS + 1);
	rois = new float[proposal_num * ROI_DIMS];

	//ROI��ݵ���
	for (int i = 0; i < proposal_num; i++) {

		//����һλ
		rois[i * ROI_DIMS + 4] = proposal_boxes[i * BOX_DIMS + 3];
		rois[i * ROI_DIMS + 3] = proposal_boxes[i * BOX_DIMS + 2];
		rois[i * ROI_DIMS + 2] = proposal_boxes[i * BOX_DIMS + 1];
		rois[i * ROI_DIMS + 1] = proposal_boxes[i * BOX_DIMS + 0];
		rois[i * ROI_DIMS + 0] = 0;
	}

	//�������������
	fast_rcnn_net->blob_by_name(cfg.data_layer)->Reshape(conv_feat_blob_data_num,
			conv_feat_blob_data_channels, conv_feat_blob_data_height,
			conv_feat_blob_data_width);
	fast_rcnn_net->blob_by_name(cfg.rois_layer)->Reshape(proposal_num, 5, 1, 1);
	fast_rcnn_net->Reshape();

	fast_rcnn_net->blob_by_name(cfg.data_layer)->set_cpu_data(data_buf);
	fast_rcnn_net->blob_by_name(cfg.rois_layer)->set_cpu_data(rois);
	fast_rcnn_net->ForwardFrom(0);

	//ȡ�����
	const float* bbox_pred_data =
			fast_rcnn_net->blob_by_name(cfg.detection_bbox_layer)->cpu_data();//�ȼ���8��300��
	int bbox_pred_height = fast_rcnn_net->blob_by_name(cfg.detection_bbox_layer)->height();
	int bbox_pred_width = fast_rcnn_net->blob_by_name(cfg.detection_bbox_layer)->width();
	int bbox_pred_channels =
			fast_rcnn_net->blob_by_name(cfg.detection_bbox_layer)->channels();
	int bbox_pred_num = fast_rcnn_net->blob_by_name(cfg.detection_bbox_layer)->num();
	//bbox_pred_data�����ȡ������
	precess_bbox_pred(bbox_pred_data, bbox_pred_num, bbox_pred_channels, bbox_pred_width, bbox_pred_height, pred_boxes);

	const float* cls_prob_data =
			fast_rcnn_net->blob_by_name(cfg.detection_cls_layer)->cpu_data();//�ȼ���2��300��
	int cls_prob_height = fast_rcnn_net->blob_by_name(cfg.detection_cls_layer)->height();
	int cls_prob_width = fast_rcnn_net->blob_by_name(cfg.detection_cls_layer)->width();
	int cls_prob_channels = fast_rcnn_net->blob_by_name(cfg.detection_cls_layer)->channels();
	int cls_prob_num = fast_rcnn_net->blob_by_name(cfg.detection_cls_layer)->num();
	precess_cls_prob(cls_prob_data, cls_prob_num, cls_prob_channels, cls_prob_width, cls_prob_width, pred_scores);

	//nms�����
	nms<float>(pred_boxes, pred_scores, bbox_pred_num,
			cfg.nms_overlap_thresint2, detection_num, detection_boxes,
			detection_scores);

	//�ͷ��ڴ�
	delete[] data_buf;
	delete[] rois;
	delete[] pred_boxes;
	delete[] pred_scores;

}

void FasterRCNN::precess_bbox_pred(const float* data, int num, int channels, int width, int height, float* &pred_boxes){

	float* pred_transforms = new float[num*ANCHOR_DIM];//release later
	pred_boxes = new float[num*ANCHOR_DIM];//for return

	int anchor_length = channels * width * height;
	assert(anchor_length == cfg.bbox_pred_num_output);
	for (int row = 0; row < num; row++) {

		pred_transforms[row*ANCHOR_DIM+0] = data[row * anchor_length + ANCHOR_DIM*cfg.bbox_pred_index+0];
		pred_transforms[row*ANCHOR_DIM+1] = data[row * anchor_length + ANCHOR_DIM*cfg.bbox_pred_index+1];
		pred_transforms[row*ANCHOR_DIM+2] = data[row * anchor_length + ANCHOR_DIM*cfg.bbox_pred_index+2];
		pred_transforms[row*ANCHOR_DIM+3] = data[row * anchor_length + ANCHOR_DIM*cfg.bbox_pred_index+3];
	}
	fast_rcnn_bbox_transform_inv(pred_transforms, proposal_boxes,
			pred_boxes, num);
	// scale back
	for (int i = 0; i < num * ANCHOR_DIM; i++) {

		pred_boxes[i] /= scale;
	}
	clip_boxes(pred_boxes, input_width, input_height, num);

	delete[] pred_transforms;
}

void FasterRCNN::precess_cls_prob(const float* data, int num, int channels, int width, int height, float* &pred_scores){

	pred_scores = new float[num];//for return

	int anchor_length = channels * width * height;
	assert(anchor_length == cfg.cls_score_num_output);
	for (int row = 0; row < num; row++) {

		pred_scores[row] = data[row * anchor_length + cfg.cls_score_index];
	}
}


void FasterRCNN::print_net(boost::shared_ptr<Net<float> > net) {

	cout << "************************************************" << endl;
	cout << "********************" << net->name() << "**********************"
			<< endl;
	cout << "************************************************" << endl;
	vector<string> blob_names = net->blob_names();
	for (int i = 0; i < blob_names.size(); i++) {

		int _height = net->blob_by_name(blob_names[i])->height();
		int _width = net->blob_by_name(blob_names[i])->width();
		int _channels = net->blob_by_name(blob_names[i])->channels();
		int _num = net->blob_by_name(blob_names[i])->num();
		cout << blob_names[i] << " : " << "_height = " << _height
				<< " _width = " << _width << " _channels = " << _channels
				<< " _num = " << _num << endl;
	}
}

void FasterRCNN::vis_detections(cv::Mat &image, int num_out,
		float* detection_boxes, float* detection_scores, float CONF_THRESH) {
	const int BOX_DIMS = 4;
	for (int i = 0; i < num_out; i++) {

		if (detection_scores[i] > CONF_THRESH) {
			cv::rectangle(image,
					cv::Point(detection_boxes[i * BOX_DIMS + 0],
							detection_boxes[i * BOX_DIMS + 1]),
					cv::Point(detection_boxes[i * BOX_DIMS + 2],
							detection_boxes[i * BOX_DIMS + 3]),
					cv::Scalar(255, 0, 0));
			char text[32];
			sprintf(text, "%.3f", detection_scores[i]);
			cv::putText(image, text,
					cv::Point(detection_boxes[i * BOX_DIMS + 0],
							detection_boxes[i * BOX_DIMS + 1]),
					CV_FONT_HERSHEY_DUPLEX, 1.0f, cv::Scalar(255, 0, 0));
		}
	}
}
