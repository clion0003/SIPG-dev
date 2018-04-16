#ifndef __SIDE_UTILS_H
#define __SIDE_UTILS_H

#include <string>
#include <vector>
#include <iostream>
#include <cstring>
#include <io.h>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

using namespace std;

vector<string> getDirFileNames(string dirPath);
double heightSD(vector<cv::Rect> r);
double yCenterSD(vector<cv::Rect> r);
bool sideSortByX(cv::Rect r1, cv::Rect r2);
bool sideSortByY(cv::Rect r1, cv::Rect r2);
bool clusSortByY(vector<cv::Rect> c1, vector<cv::Rect> c2);
bool sortByLen(vector<cv::Rect> v1, vector<cv::Rect> v2);
bool sortByH(cv::Rect r1, cv::Rect r2);
bool sortByW(cv::Rect r1, cv::Rect r2);
cv::Rect getRowClusBbox(vector<cv::Rect> cluster);
cv::Rect getColClusBbox(vector<cv::Rect> cluster);
int getClusAvgH(vector<cv::Rect> cluster, int deOutlier);
int getClusAvgW(vector<cv::Rect> cluster, int deOutlier);
int getClusTypicalH(vector<cv::Rect> cluster);
int getClusTypicalW(vector<cv::Rect> cluster);
void clearDir(string path);
int getMaxClusIdx(vector<vector<cv::Rect>> clus);
int getMinYDis(cv::Rect r1, cv::Rect r2);

#endif