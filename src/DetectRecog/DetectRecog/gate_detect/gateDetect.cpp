#include "gateDetect.h"

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching.hpp>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include "gateUtils.h"
#include "../socketcpp/tcpClient.h"
#include "../side_detect/side_utils.h"

enum HELP_DEBUG_LEVEL{NO_DEBUG, DEBUG_LOW, DEBUG_MIDDLE, DEBUG_HIGH};
enum HELP_SHOW_LEVEL{NO_SHOW, SHOW_ONLY_RESULT, SHOW_SOME_TMP_IMAGE, SHOW_ALL_TMP_IMAGE, SHOW_FOR_DEBUG};

#define DEBUG_LEVEL 0
#define SHOW_LEVEL 1

using namespace std;
using namespace cv;

string debug_path = "G:\\qianhou2\\database\\101_f.jpg";

gateDetect::gateDetect(string filePath, bool isFront){

#if SHOW_LEVEL > 3
    filePath = debug_path;
#endif 

    this->isFront = isFront;
    src_path = filePath;

    src = imread(src_path);

    org_path = src_path;
    org_path.replace(org_path.find(".jpg"), 4, "_dl.jpg");

    initSocket();

    Rect deeplab_box;
    deeplab_request(src_path, deeplab_box);

    org = src(deeplab_box);

    cvtColor(org, gray, CV_BGR2GRAY);
}

gateDetect::~gateDetect() {
    DeleteFile(org_path.c_str());
}

void gateDetect::detect()
{
    while (delta >= 1) {
        mserBox = getMserBox(delta, min_area, max_area, max_variation);
        if (mserBox.size() >= 50) break;
        delta--;
    }

    verticalFilter();

    vector<Rect> final_result = ret;

    int s = ret.size();

    if (s < 11) {
        mserBox = getMserBox(1, 30, max_area, max_variation);
        verticalFilter();
        if (ret.size() == 11) final_result = ret;
    }
    else if (s > 11) {
        while (ret.size() > 11 && delta <= 20) {
            delta += 5;
            mserBox = getMserBox(delta, min_area, max_area, max_variation);
        }
        if (ret.size() == 11) final_result = ret;
    }

    swap(final_result, ret);
    clearDir(TMP_SAVE_PATH);

    for (int i = 0; i < ret.size(); i++) {
        imwrite(TMP_SAVE_PATH + to_string(i) + ".jpg", org(ret[i]));
    }

    if (ret.size()) {
        initSocket();
        string res;
        alex_request(TMP_SAVE_PATH, res);
        cout << "result: " << res << endl;
        showImg(ret, "final result");
    }
    else {
        showImg(ret, "find nothing");
    }
}

// generate vector rects for detection by mser;
vector<Rect> gateDetect::getMserBox(int delta, int min_area, int max_area, double max_variation) {
    // mser;
    Ptr<MSER> ptr = MSER::create(delta, min_area, max_area, max_variation);
    vector<vector<Point>> mserContours;
    vector<Rect> mserBox;
    ptr->detectRegions(gray, mserContours, mserBox);

    // discard rects by height-width rate;
    vector<Rect> tmpBox;
    for (auto box : mserBox) {
        double rate = box.height*1.0 / box.width;
        if (rate < 10 && rate >= 1.3) {
            tmpBox.push_back(box);
        }
    }
    swap(tmpBox, mserBox);

    // merge almost same rects;
    vector<Rect> res;
    sort(mserBox.begin(), mserBox.end(), utils::sortByArea);
    for (auto a : mserBox) {
        bool find = false;
        for (auto &b : res) {
            Rect c = a & b;
            if (c.area() > 0.9 * a.area() && c.area() > 0.9 * b.area()) {
                b = a | b;
                find = true;
                break;
            }
        }
        if (!find) res.push_back(a);
    }

#if DEBUG_LEVEL
    cout << "mser config: [" << delta << ", " << min_area << ", " << max_area << ", " << max_variation << "] => " << res.size() << endl;
#endif

#if SHOW_LEVEL > 1
    showImg(mserBox, "mser");
#endif
    return res;
}

void gateDetect::SingleFilter() {

}

// show img + rects;
void gateDetect::showImg(vector<Rect>& boxes, string name)
{
    name = src_path + " -- [" + name + "]";
    Mat tmp = org.clone();
    for (auto box : boxes) {
        cv::rectangle(tmp, box, cv::Scalar(0, 0, 255));
    }
    imshow(name, tmp);
    waitKey();
    destroyWindow(name);
}

// show img + rects + line;
void gateDetect::showImg(vector<Rect>& boxes, Vec4f paras, string name){
    name = src_path + " -- [" + name + "]";
    Mat tmp = org.clone();
    for (auto box : boxes) {
        cv::rectangle(tmp, box, cv::Scalar(0, 0, 255));
    }
    Point2f p0(paras[2], paras[3]);
    double y1 = p0.y + paras[1] * 500;
    double x1 = p0.x + paras[0] * 500;
    double y2 = p0.y - paras[1] * 500;
    double x2 = p0.x - paras[0] * 500;
    line(tmp, Point2f(x1, y1), Point2f(x2, y2), cv::Scalar(0, 255, 0));
    imshow(name, tmp);
    waitKey();
    destroyWindow(name);
}

// some checks of vertical numbers;
bool gateDetect::verticalCheck(vector<Rect>& a, bool isFirst) {

    if (isFirst && a.size() < 10) return false;

    // compute average height;
    int sum = 0;
    int last = 0;
    int avg = 10000;
    while (true) {
        int num = 0;
        sum = 0;
        for (auto it : a) {
            if (it.height < 1.5*avg) {
                sum += it.height;
                num++;
            }
        }
        avg = sum * 1.0 / num;
        if (num == last) break;
        last = num;
    }
    if (last < 8) return false;

    // discards rects higher than 1.5 times avg height;
    vector<Rect> tmp;
    for (auto it : a) if (it.height < 1.5*avg) tmp.push_back(it);
    swap(tmp, a);

#if DEBUG_LEVEL
    cout << "check " << (isFirst ? "first" : "second") << endl;
    cout << "height filter size:" << a.size() << " avg height:" << avg << endl;
#endif

    // sweep line algorithm from top to bottom;
    sort(a.begin(), a.end(), utils::sortByY);
    int end = -1;
    tmp.clear();
    vector<vector<Rect>> res;
    for (auto it : a) {
        if (it.y > end) {
            if (tmp.size()) res.push_back(tmp);
            end = it.y + it.height + avg;
            tmp.clear();
            tmp.push_back(it);
        }
        else {
            tmp.push_back(it);
            end = max(end, it.y + it.height + avg);
        }
    }
    if (tmp.size()) res.push_back(tmp);

    // check by total height;
    vector<int> height;
    vector<pair<int, int>> lh;
    sum = 0;
    for (auto it : res) {
        int low = INT_MAX;
        int high = INT_MIN;
        for (auto v : it) {
            high = max(high, v.y + v.height);
            low = min(low, v.y);
        }
        height.push_back(high - low + 1);
        lh.push_back({ low, high });
        sum += height.back();
    }
    if (sum < 9 * avg) return false;

    // choose highest cluster as container number;
    int mh = max_element(height.begin(), height.end()) - height.begin();
    tmp.clear();
    tmp.insert(tmp.end(), res[mh].begin(), res[mh].end());

#if DEBUG_LEVEL > 1
    cout << "cluster size: " << res.size() << " " << mh << endl;
#endif 

    // propose to add the check bit;
    if (mh + 1 < height.size() && res[mh + 1][0].y - lh[mh].second <2 * avg) {
        tmp.insert(tmp.end(), res[mh + 1].begin(), res[mh + 1].end());
    }

    // propose to add company number;
    if (mh > 0 && height[mh - 1] > 3.6*avg) {
        tmp.insert(tmp.end(), res[mh - 1].begin(), res[mh - 1].end());
    }
    swap(a, tmp);

#if SHOW_LEVEL > 2
    if (isFirst) showImg(a, "check first");
    else showImg(a, "check second");
#endif

    // check twice;
    if (isFirst) return verticalCheck(a, false);
    return true;
}


// deal with some special cases of overlap rects;
void gateDetect::verticalMerge(vector<Rect>& a) {
    // case1: delete the redundant rect overlap with two rects like:
    /*
          A
          U        
    */
    sort(a.begin(), a.end(), utils::sortByY);
    vector<Rect> tmp;
    vector<int> bad;
    bool flag = false;
    for (int i = 0; i < a.size(); i++) {
        if (i&&i + 1 < a.size() && utils::isOverlap(a[i], a[i - 1]) && utils::isOverlap(a[i], a[i + 1]) && a[i].width < a[i - 1].width&&a[i].width < a[i + 1].width) {
            flag = true;
            continue;
        }
        else {
            tmp.push_back(a[i]);
        }
    }
    swap(tmp, a);

#if SHOW_LEVEL > 2
    if (flag) showImg(a, "merge first");
#endif

    // case2: merge two rects who have similar vertical postion;
    tmp.clear();
    flag = false;
    for (int i = 0; i < a.size(); i++) {
        int dh = i + 1 < a.size() ? a[i].y + a[i].height - a[i + 1].y : 0;
        if (i + 1 < a.size() && utils::isOverlap(a[i], a[i + 1]) && (i == 0 || !utils::isOverlap(a[i - 1], a[i])) && (dh > 0.8*a[i].height || dh > 0.8*a[i + 1].height)) {
            tmp.push_back(a[i] | a[i + 1]);
            flag = true;
            i++;
        }
        else {
            tmp.push_back(a[i]);
        }
    }
    swap(tmp, a);

#if SHOW_LEVEL > 2
    if (flag) showImg(a, "merge second");
#endif

}


// detect vertical numbers;
void gateDetect::verticalFilter() {

    // step1: line sweep algorithm from left to right;
    sort(mserBox.begin(), mserBox.end(), utils::sortByX);
    vector<vector<Rect>> res;
    vector<Rect> tmp;

    int end = -1;
    for (auto it : mserBox) {
        if (it.x < 0.3*org.cols || abs(org.cols - it.x - it.width)<20) continue;
        if (it.x > end) {
            if (verticalCheck(tmp)) res.push_back(tmp);
            end = it.x + it.width;
            tmp.clear();
            tmp.push_back(it);
        }
        else {
            tmp.push_back(it);
            end = max(end, it.x + it.width);
        }
    }
    if (verticalCheck(tmp)) res.push_back(tmp);


    // step2: linear fitting;
    ret.clear();
    int maxnum = 0;
    Vec4f res_paras;
    for (auto& rects : res) {
        Vec4f paras;
        while (true) {

            // linear fitting using center points of rects;
            int num = rects.size();
            vector<Point> vpt;
            for (auto it : rects) {
                vpt.push_back(Point(it.x + it.width / 2, it.y + it.height / 2));
            }
            fitLine(vpt, paras, DIST_L1, 0, 1e-2, 1e-2);
            double dy = paras[1];
            double dx = paras[0];
            double x0 = paras[2];
            double y0 = paras[3];
            tmp.clear();

            // discard rects far away from linear line;
            for (auto it : rects) {
                double x = it.x + it.width / 2;
                double y = it.y + it.height / 2;
                double dist = abs(dy * x - dx * y + dx * y0 - dy * x0);
                if (dist <= it.width / 2) {
                    tmp.push_back(it);
                }
            }
            rects.clear();
            for (auto it : tmp) {
                rects.push_back(it);
            }
            if (num == rects.size()) break;
        }

        // merge complete overlap rects;
        tmp.clear();
        std::sort(rects.begin(), rects.end(), utils::sortByArea);
        for (int i = 0; i < rects.size(); i++) {
            bool flag = true;
            for (int j = i + 1; j < rects.size(); j++) {
                Rect a = rects[i];
                Rect b = rects[j];
                Rect c = a & b;
                if (c == a) {
                    flag = false;
                }
            }
            if (flag) tmp.push_back(rects[i]);
        }
        rects.clear();
        for (auto it : tmp) {
            rects.push_back(it);
        }

        // deal with some special cases of overlap rects;
        verticalMerge(rects);

        // choose the most likely cluster as result;
        // paras[1] determines the sin of angle;
        if (rects.size() > maxnum&&paras[1]<-0.97 || paras[1]>0.99) {
            maxnum = rects.size();
            ret = rects;
            res_paras = paras;
        }
    }

    // step3: scale width;
    if (!ret.size()) return;
    avg_width = 0;
    avg_height = 0;
    for (auto it : ret) {
        avg_width += it.width;
        avg_height += it.height;
    }
    avg_width /= ret.size();
    avg_height /= ret.size();
    avg_width += 1;
    avg_height += 1;
    for (auto &it : ret) {
        if (it.width < avg_width) {
            int mid = it.x + it.width / 2;
            it.width = avg_width;
            it.x = mid - it.width / 2;
        }
    }

    fillblank(ret);

#if SHOW_LEVEL > 1
    showImg(ret, res_paras, "vertical result");
#endif

}

void gateDetect::fillblank(vector<Rect>& a){
    int len = a.size();
    if (len >= 11) return;
    sort(a.begin(), a.end(), utils::sortByY);
    vector<Rect> center;
    for (int i = 4; i < len - 1; i++) {
        int dh = a[i + 1].y - (a[i].y + a[i].height);
        if(dh > 0.8*avg_height){
            double rate = dh*1.0 / avg_height;
            if (rate > 0.9&&rate < 1.2) {
                int cx = (a[i].x + a[i].width / 2 + a[i + 1].x + a[i + 1].width / 2) / 2;
                int cy = (a[i].y + a[i].height / 2 + a[i + 1].y + a[i + 1].height / 2) / 2;
                Rect tmp(cx - avg_width / 2, cy - avg_height / 2, avg_width, avg_height);
                center.push_back(tmp);
            }
        }
    }

    if (center.size() + a.size() <= 11) {
        for (auto it : center) {
            a.push_back(it);
        }
    }
#if SHOW_LEVEL > 1
    if(center.size()) showImg(center, "fill blank");
#endif
}





