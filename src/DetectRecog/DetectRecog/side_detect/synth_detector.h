#ifndef __SYNTH_DETECTOR_H
#define __SYNTN_DETECTOR_H

#include "mser_filter.h"
#include "east_filter.h"

class SynthDetector
{
public:
	SynthDetector(string imgPath, string tmpSavePath);
	string getStr(void);
private:
	string imgPath;
	string tmpSavePath;
	cv::Mat srcImg;
	string resStr;
	vector<vector<cv::Rect>> finalRes;
	int resMode;
private:
	void getMserRes(vector<vector<cv::Rect>> &mserRes, cv::Rect deeplabBbox);
	void getEastRes(vector<vector<cv::Rect>> &eastRes, cv::Rect deeplabBbox);
	int synthRes(vector<vector<cv::Rect>> &mserRes, vector<vector<cv::Rect>> &eastRes);
};

#endif
