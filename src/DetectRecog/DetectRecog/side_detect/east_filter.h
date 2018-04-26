#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>
#include <iostream>
#include "../socketcpp/tcpClient.h"
#include "../detect/MyTypes.h"
#include "side_utils.h"
#include "side_config.h"

typedef struct east_bndbox EastBbox;

class EastFilter
{
public:
	EastFilter(string imgPath, cv::Rect deeplabBbox, std::vector<EastBbox> eastBboxes);
	EastFilter(string imgPath, cv::Rect deeplabBbox);
	void drawBboxes(string outputPath, std::vector<cv::Rect> bboxes);
	void drawBboxes(string outputPath, std::vector<std::vector<cv::Rect>> clus);
	void filter(void);

public:
	std::vector<std::vector<cv::Rect>> clus;
	string debugSavePrefix;

private:
	std::vector<cv::Rect> bboxes;
	std::vector<std::vector<cv::Rect>> clusters;
	std::vector<EastBbox> eastBboxes;
	cv::Rect deeplabBbox;
	cv::Mat srcImg;

	// ãÐÖµ
	double eastHeightSmall, eastHeightHuge, eastWidthHuge;

private:
	cv::Rect eastbox2rect(EastBbox bbox);
	void deeplabFilter(void);
	void singleFilter(void);
	void clusterFilter(void);
	void cluster(int disThres, std::vector<cv::Rect> bboxes, std::vector<std::vector<cv::Rect>>& clus, int mode);
	bool isClusterable(cv::Rect r1, cv::Rect r2, int disThres, int mode);

	void findMainRegion(std::vector<cv::Rect>& mainRegion);
	void buildResult(std::vector<cv::Rect> mainRegion);
};
