//============================================================================
// Name        : PreparingTrainingData.cpp
// Author      : onlyygo
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "boost/algorithm/string.hpp"
#include <algorithm>
using namespace std;
#include "ProposalTrain.h"
#include "FastRcnnTrain.h"

int main() {

	google::InitGoogleLogging("CNN");
	FLAGS_stderrthreshold = google::ERROR;
	caffe::Caffe::SetDevice(1);
	caffe::Caffe::set_mode(caffe::Caffe::GPU);

	string image_path =
			"./datasets/VOCdevkit_lihe/VOC2007/JPEGImages/";
	string info_file = "./info.txt";
	ifstream in(info_file);
	vector<string> lines;
	while (!in.eof()) {
		string line;
		std::getline(in, line);
		if (line.size() == 0 || line.at(0) == '#')
			continue;
		line = line.substr(0, line.size() - 1);
		lines.push_back(line);
	}
	in.close();
	/////////////////ProposalTrain
	{
		cout << "!!!Proposal Training start!!!" << endl; // prints !!!Hello World!!!
		ProposalTrain pt(
				"./models/rpn_prototxts/vgg_16layers_conv3_1/solver_60k80k.prototxt",
				"./models/pre_trained_models/vgg_16layers/vgg16.caffemodel",
				"./models/rpn_prototxts/vgg_16layers_conv3_1/test.prototxt",
				0);

		int max_iter = pt.solver_->max_iter();
		int iter = pt.solver_->iter();
		while (iter < max_iter) {

			random_shuffle(lines.begin(), lines.end());
			string line = lines[0];
			cout << "info: " << line << endl;
			vector<string> ss;
			boost::split(ss, line, boost::is_any_of("\t"));
			string image_name = image_path + ss[0];
			int boxes_num = atoi(ss[1].c_str());

			cv::Mat img0 = cv::imread(image_name);
			ImageRoiDB image_roidb;
			image_roidb.image0 = img0;
			image_roidb.boxes_num = boxes_num;
			image_roidb.im_scales = -1;
			image_roidb.bbox_targets = NULL;
			image_roidb.bbox_targets_num = -1;
			image_roidb.boxes = new float[image_roidb.boxes_num * 4];
			for (int i = 0; i < image_roidb.boxes_num; i++) {

				string box_str = ss[2 + i].substr(1, ss[2 + i].size() - 2);
				vector<string> p1p2s;
				boost::split(p1p2s, box_str, boost::is_any_of(","));
				int x1 = max(atoi(p1p2s[0].c_str()), 0);
				int y1 = max(atoi(p1p2s[1].c_str()), 0);
				int x2 = max(atoi(p1p2s[2].c_str()), 0);
				int y2 = max(atoi(p1p2s[3].c_str()), 0);
				image_roidb.boxes[i * 4 + 0] = x1;
				image_roidb.boxes[i * 4 + 1] = y1;
				image_roidb.boxes[i * 4 + 2] = x2;
				image_roidb.boxes[i * 4 + 3] = y2;
			}
			pt.proposal_train(image_roidb);
			pt.show_accuarcy();
			delete[] image_roidb.boxes;

			iter = pt.solver_->iter();
		}
		pt.save_mode("proposal_vgg16.caffemode");
	}
	/////////////////FastRcnnTrain
	{
		cout << "!!!Fast-RCNN Training start!!!" << endl; // prints !!!Hello World!!!
		FastRcnnTrain ft(
				"./models/fast_rcnn_prototxts/vgg_16layers_conv3_1/solver_30k40k.prototxt",
				"./models/pre_trained_models/vgg_16layers/vgg16.caffemodel",
				"./models/fast_rcnn_prototxts/vgg_16layers_conv3_1/test.prototxt",
				0);

		int max_iter = ft.solver_->max_iter();
		int iter = ft.solver_->iter();
		while (iter < max_iter) {

			random_shuffle(lines.begin(), lines.end());
			string line = lines[0];
			cout << "info: " << line << endl;
			vector<string> ss;
			boost::split(ss, line, boost::is_any_of("\t"));
			string image_name = image_path + ss[0];
			int boxes_num = atoi(ss[1].c_str());

			cv::Mat img0 = cv::imread(image_name);
			ImageRoiDB image_roidb;
			image_roidb.image0 = img0;
			image_roidb.boxes_num = boxes_num;
			image_roidb.im_scales = -1;
			image_roidb.bbox_targets = NULL;
			image_roidb.bbox_targets_num = -1;
			image_roidb.boxes = new float[image_roidb.boxes_num * 4];
			for (int i = 0; i < image_roidb.boxes_num; i++) {

				string box_str = ss[2 + i].substr(1, ss[2 + i].size() - 2);
				vector<string> p1p2s;
				boost::split(p1p2s, box_str, boost::is_any_of(","));
				int x1 = max(atoi(p1p2s[0].c_str()), 0);
				int y1 = max(atoi(p1p2s[1].c_str()), 0);
				int x2 = max(atoi(p1p2s[2].c_str()), 0);
				int y2 = max(atoi(p1p2s[3].c_str()), 0);
				image_roidb.boxes[i * 4 + 0] = x1;
				image_roidb.boxes[i * 4 + 1] = y1;
				image_roidb.boxes[i * 4 + 2] = x2;
				image_roidb.boxes[i * 4 + 3] = y2;
			}
			ft.set_anchors(image_roidb.boxes, image_roidb.boxes_num);
			ft.fast_rcnn_train(image_roidb);
			ft.show_accuarcy();
			ft.free();
			delete[] image_roidb.boxes;

			iter = ft.solver_->iter();
		}
		ft.save_mode("fast_vgg16.caffemode");
	}
	google::ShutdownGoogleLogging();
	return 0;
}
