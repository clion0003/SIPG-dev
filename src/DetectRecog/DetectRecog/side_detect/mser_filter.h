#ifndef _MSER_FILTER_H
#define _MSER_FILTER_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <cmath>
#include "side_utils.h"
#include <iostream>

using namespace std;

#define WRONG 0
#define ROW_MODE 1
#define ROW_LOOSE_MODE 2	//�ڶ����о���ʱ�ſ���һЩ����
#define COL_MODE 3
#define COL_LOOSE_MODE 4

class MserFilter
{
public:
	MserFilter(cv::Mat srcImg, cv::Rect deeplabBbox);
	void drawMser(string outputPath);
	void drawBboxes(string outputPath, vector<cv::Rect> bboxes);
	void drawClus(string outputPath, vector<vector<cv::Rect>> clus);
	void drawResult(string outputPath, vector<cv::Rect> result);
	int filter(void);
	int judgeSide(void);
	virtual ~MserFilter();
public:
	vector<vector<cv::Rect>> rowClus;	// ���rowClus[0], [1], [2]�ֱ�Ϊ��˾��, ���, ���͵�ֵ
	vector<vector<cv::Rect>> colClus;	// ���colClus[0], [1], [2]�ֱ�Ϊ��˾��, ���, ���͵�ֵ

	string savePath;	// for debug
	cv::Rect mainRegion;

private:
	cv::Mat srcImg;
	cv::Rect deeplabBbox;
	vector<vector<cv::Point>> msers;
	vector<cv::Rect> bboxes;

	// ��ֵ
	double mserHeightSmall, mserHeightHuge, mserWidthHuge;

private:
	//һЩ�����õĹ��ܺ���
	bool isClose(cv::Rect r1, cv::Rect r2, int disThres, int mode);			//���ھ���ʱ�ж��Ƿ�����һ��
	void deeplabFilter(void);
	void singleFilter(void);
	void clusFilter(void);
	int clusProcess(int mode);
	void cluster(int disThres, vector<cv::Rect> bboxes, vector<vector<cv::Rect>>& clus, int mode);

	// �д�����غ���
	void findMainRow(vector<cv::Rect>& mainRowClus);
	void getByMainRow(vector<cv::Rect> mainRowClus);
	int buildRowResult(void);
	vector<cv::Rect> getBetterRow(vector<cv::Rect> mainClus, vector<cv::Rect> oldClus, vector<cv::Rect> newClus);
	void rmRowOutlier(vector<cv::Rect>& clus);
	void mergeRowSmallBbox(vector<cv::Rect>& cluster);

	// �д�����غ���
	void findMainCol(vector<cv::Rect>& mainColClus);
	void getByMainCol(vector<cv::Rect> mainColClus);
	int buildColResult(void);
	void colReDiv(vector<cv::Rect>& clus);
	void colSelfPatch(vector<cv::Rect>& clus, int mainCharH, int mainCharW);
    void patchByBorder(vector<cv::Rect>& bboxes, int mainCharH, int border, int targetLen);
	void rmColOutlier(vector<cv::Rect>& clus, int mainCharH, int mainCharW);
	void simpleClus(vector<vector<cv::Rect>> &tmpClus, vector<cv::Rect> bboxes, double disThre);
	vector<cv::Rect> getBetterCol(vector<cv::Rect> mainClus, vector<cv::Rect> oldClus, vector<cv::Rect> newClus);
};

#endif
