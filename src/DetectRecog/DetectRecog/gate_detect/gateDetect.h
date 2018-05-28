#ifndef __GATE_DETECT_H
#define __GATE_DETECT_H

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching.hpp>

using std::string;
using std::vector;
using cv::Mat;
using cv::Rect;

const string TMP_SAVE_PATH = "D:\\SideTmpDir\\";

class gateDetect {
public:
    gateDetect(string filePath, bool isFront = true);

    ~gateDetect();

    void detect();

private:
    bool isFront;

    string src_path;
    string org_path;

    Mat src; // src image;
    Mat org; // deeplab image;
    Mat gray; // gray deeplab image;
    
    int delta = 5;
    int min_area = 50;
    int max_area = 500;
    double max_variation = 0.6;
    vector<Rect> mserBox;
    vector<Rect> getMserBox(int delta, int min_area, int max_area, double max_variation);
    void SingleFilter();
    void showImg(vector<Rect>& boxes, string name = "show");
    void showImg(vector<Rect>& boxes, cv::Vec4f para, string name = "show");
    bool verticalCheck(vector<Rect>& a, bool isFirst = true);

    void verticalMerge(vector<Rect>& a);

    void verticalFilter();
    vector<Rect> ret;
    cv::Vec4f res_paras;
    void fillblank(vector<Rect>& a);
    int avg_width = 0;
    int avg_height = 0;

};


#endif // !__GATE_DETECT_H
