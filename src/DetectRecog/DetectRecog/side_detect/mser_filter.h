#ifndef _MSER_FILTER_H
#define _MSER_FILTER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>
#include "side_utils.h"
#include "side_config.h"
#include <iostream>

using namespace std;

//最终的结果是返回vector<vector<Rect>>类型，第一个里是公司号，第二个是箱号，第三个是箱型
//如果是行模式，每一个vector<Rect>的长度最长为1，列模式则允许更长

class MserFilter
{
public:
	MserFilter(cv::Mat srcImg, cv::Rect deeplabBbox);
	void drawMser(string outputPath);
	void drawBboxes(string outputPath, vector<cv::Rect> bboxes);
	void drawClus(string outputPath, vector<vector<cv::Rect>> clus);
	void drawOnSrcImg(string outputPath, vector<vector<cv::Rect>> clus);
	int filter(void);
	int judgeSide(void);
	virtual ~MserFilter();
public:
	vector<vector<cv::Rect>> rowClus;	// 最后rowClus[0], [1], [2]分别为公司号, 箱号, 箱型的值
	vector<vector<cv::Rect>> colClus;	// 最后colClus[0], [1], [2]分别为公司号, 箱号, 箱型的值

	string debugSavePrefix;	// for debug
	cv::Rect mainRegion;

private:
	cv::Mat srcImg;
	cv::Rect deeplabBbox;
	vector<vector<cv::Point>> msers;
	vector<cv::Rect> bboxes;

	// 阈值
	double mserHeightSmall, mserHeightHuge, mserWidthHuge;

private:
	//一些供调用的功能函数
	bool isClose(cv::Rect r1, cv::Rect r2, int disThres, int mode);			//用于聚类时判断是否属于一类
	void deeplabFilter(void);
	void singleFilter(void);
	void clusFilter(void);
	int clusProcess(int mode);
	void cluster(int disThres, vector<cv::Rect> bboxes, vector<vector<cv::Rect>>& clus, int mode);

	// 行处理相关函数
	void findMainRow(vector<cv::Rect>& mainRowClus);
	void getByMainRow(vector<cv::Rect> mainRowClus);
	int buildRowResult(void);
	vector<cv::Rect> getBetterRow(vector<cv::Rect> mainClus, vector<cv::Rect> oldClus, vector<cv::Rect> newClus);
	void rmRowOutlier(vector<cv::Rect>& clus);
	void mergeRowSmallBbox(vector<cv::Rect>& cluster);

	// 列处理相关函数
	void findMainCol(vector<cv::Rect>& mainColClus, int &mainCharH, int &mainCharW);
	void getByMainCol(vector<cv::Rect> mainColClus, int mainCharH, int mainCharW);
	int buildColResult(void);
	void colReDiv(vector<cv::Rect>& clus);
	void colSelfPatch(vector<cv::Rect>& clus, int mainCharH, int mainCharW, int mode);
	void midPatch(vector<vector<cv::Rect>> &clus, int mainCharH, int mainCharW);
	void sidePatch(vector<vector<cv::Rect>> &clus, int mainCharH, int mainCharW);
	void adjustBbox(vector<vector<cv::Rect>> &clus, int mainCharH, int mainCharW);
    void patchByBorder(vector<cv::Rect>& bboxes, int mainCharH, int border, int targetLen);
	void rmColOutlier(vector<cv::Rect>& clus, int mainCharH, int mainCharW, int mode);
	void simpleClus(vector<vector<cv::Rect>> &tmpClus, vector<cv::Rect> bboxes, double disThre);
	vector<cv::Rect> getBetterCT(vector<cv::Rect> mainClus, vector<cv::Rect> oldClus, vector<cv::Rect> newClus);
	int getContainerIdIdx(vector<vector<cv::Rect>>& tmpClus, int mainCharH);
};

#endif
