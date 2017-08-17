#ifndef CONFIG_H_
#define CONFIG_H_

const int ANCHOR_NUM = 9;
const int ANCHOR_DIM = 4;

const int ANCHORS[ANCHOR_NUM][ANCHOR_DIM] = {  { -83, -39, 100, 56 },

		{ -175, -87, 192, 104 }, { -359, -183, 376, 200 }, { -55, -55, 72, 72 },
		{ -119, -119, 136, 136 }, { -247, -247, 264, 264 },
		{ -35, -79, 52, 96 }, { -79, -167, 96, 184 }, { -167, -343, 184, 360 } };

struct Config {

	int per_nms_topN = 6000;
	float nms_overlap_thresint = 0.7;
	float nms_overlap_thresint2 = 0.25;
	int after_nms_topN = 300;
	bool do_scale = true;

	string data_layer = "data";
	string rois_layer = "rois";
	string proposal_cls_layer = "proposal_cls_prob";
	string proposal_bbox_layer = "proposal_bbox_pred";
	string detection_bbox_layer = "bbox_pred";
	string detection_cls_layer = "cls_prob";
	string shared_layer = "conv5_3";
	int class_num = 1;
	int cls_score_num_output = class_num+1;
    int bbox_pred_num_output = (class_num+1)*ANCHOR_DIM;
	int cls_score_index = 1;
    int bbox_pred_index = 1;

	int test_scales = 800;
	int test_max_size = 1000;
	int feat_stride = 16;
	int test_min_box_size = 16;
};

#endif
