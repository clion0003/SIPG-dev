#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>
#include <iostream>
#include "../socketcpp/tcpClient.h"
#include "../detect/MyTypes.h"

using namespace std;

typedef struct east_bndbox EastBbox;

class EastFilter
{
public:
	EastFilter(string imgPath, cv::Rect deeplabBbox);
	void drawBboxes(string outputPath);
	void filter(void);
private:
	vector<cv::Rect> bboxes;
	vector<vector<cv::Rect>> clusters;
	vector<EastBbox> eastBboxes;
	cv::Rect deeplabBbox;
	cv::Mat srcImg;

	// ãÐÖµ
	double mserHeightSmall, mserHeightHuge, mserWidthHuge;
private:
	cv::Rect eastbox2rect(EastBbox bbox);
	void deeplabFilter(void);
	void singleFilter(void);
	void clusterFilter(void);
	bool isClose(cv::Rect r1, cv::Rect r2);
};
