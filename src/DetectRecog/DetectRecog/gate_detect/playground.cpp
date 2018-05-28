#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem.hpp>
#include "..\socketcpp\tcpClient.h"
using namespace std;
using namespace cv;
#define DEBUGSHOW 3
Mat org, gray, show;
vector<Rect> mserBbox;
vector<Rect> dp;
vector<Rect> history;
string testpath = "G:\\qianhou2\\database\\553_f.jpg";
string badtest = "531,501, 725,33 ,30, 354, 626, 12,713,37,7,24,20,239...67,";
int lvl = 5;
int result = -1;
int minarea = 50;
double maxd = 0.6;
namespace {
    string savefilepath = "C:\\Users\\archlab\\Desktop\\out4.txt";
    string savefilepath2 = "C:\\Users\\archlab\\Desktop\\out5.txt";
    int s0 = 0;
    int s1 = 0;
    int s2 = 0;

    bool sortByArea(const Rect& lhs, const Rect& rhs) {
        return lhs.area() < rhs.area();
    }

    bool sortByX(const Rect& a, const Rect& b) {
        return a.x < b.x;
    }

    bool sortByY(const Rect& a, const Rect& b) {
        return a.y < b.y;
    }

    void geometryFilter(vector<Rect>& input_boxes, vector<Rect>& geometry_filter, int strict_mode) {
        for (auto rect : input_boxes) {
            if (rect.height > rect.width && rect.height *1.0 / rect.width < 10 && abs(org.cols - rect.x - rect.width)>20) {
                if (strict_mode == 1 && rect.height *1.0 / rect.width > 1.3)
                    geometry_filter.push_back(rect);
                else
                    geometry_filter.push_back(rect);
            }
        }

        sort(geometry_filter.begin(), geometry_filter.end(), sortByArea);
        vector<Rect> res;
        for (auto a : geometry_filter) {
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

        std::swap(res, geometry_filter);
        
    }
    void mser() {
        
        vector<Rect> geometry_filter;
        for (int i = lvl; i >= 1; i--) {
            lvl = i;
            Ptr<MSER> ptr = MSER::create(i, minarea, 500, maxd);
            cout << "config:" << i << " " << minarea << " " << maxd << endl;
            vector<vector<Point>> mserContours;
            geometry_filter.clear();
            mserBbox.clear();
            ptr->detectRegions(gray, mserContours, mserBbox);
            geometryFilter(mserBbox, geometry_filter, 1);
            cout << "mser size:" << i << " : " << geometry_filter.size() << endl;
            if (lvl>5||geometry_filter.size() >= 50) break;
        }
        history = geometry_filter;
        swap(dp, geometry_filter);
#if DEBUGSHOW > 1
        show = org.clone();
        for (auto box : dp) {
            cv::rectangle(show, box, cv::Scalar(0, 0, 255));
        }
        imshow("start", show);
        waitKey();
        destroyWindow("start");
#endif
    }
    bool check(vector<Rect>& a, bool isFirst = true) {
        
        if (a.size() < 10) return false;
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
            //cout << "debug" << sum << " " << num << " "<<avg<< endl;
            if (num == last) break;
            last = num;
        }
        if (last < 8) return false;
        vector<Rect> tmp;
        for (auto it : a) if (it.height < 1.5*avg) tmp.push_back(it);
        swap(tmp, a);
        cout << "check " << (isFirst ? "first" : "second") << endl;
        cout << "height filter size:"<<a.size() << " "<<avg<<endl;
#if DEBUGSHOW > 2
        Mat s0 = org.clone();
        for (auto box : a) {
            cv::rectangle(s0, box, cv::Scalar(0, 0, 255));
        }
        imshow("s0", s0);
        waitKey();
        destroyWindow("s0");
#endif
        sort(a.begin(), a.end(), sortByY);
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
        vector<int> height;
        vector<pair<int, int>> lh;
        sum = 0;
        for (auto it:res) {
            int low = INT_MAX;
            int high = INT_MIN;
            for (auto v:it) {
                high = max(high, v.y + v.height);
                low = min(low, v.y);
            }
            height.push_back(high - low + 1);
            lh.push_back({ low, high });
            sum += height.back();
        }
        if (sum < 9 * avg) return false;
        int mh = max_element(height.begin(), height.end()) - height.begin();
        cout << "cluster size: "<<res.size() << " " << mh << endl;
        tmp.clear();
        tmp.insert(tmp.end(), res[mh].begin(), res[mh].end());
        if (mh + 1 < height.size()&&res[mh+1][0].y - lh[mh].second <2*avg) {
            tmp.insert(tmp.end(), res[mh + 1].begin(), res[mh + 1].end());
        }
        if (mh > 0 && height[mh - 1] > 3.6*avg) {
            tmp.insert(tmp.end(), res[mh - 1].begin(), res[mh - 1].end());
        }
        swap(a, tmp);
        if(isFirst) return check(a, false);
        return true;
    }
    bool isOverlap(Rect& a, Rect& b) {
        return a.x < b.x + b.width&&b.x < a.x + a.width&&a.y < b.y + b.height&&b.y < a.y + a.height;
    }
    void lastchance(vector<Rect>& a) {

        sort(a.begin(), a.end(), sortByY);
        vector<Rect> tmp;
        vector<int> bad;
        bool flag = false;
        for (int i = 0; i < a.size(); i++) {
            if (i&&i+1<a.size()&&isOverlap(a[i], a[i - 1]) && isOverlap(a[i], a[i + 1])&& a[i].width<a[i - 1].width&&a[i].width<a[i + 1].width) {
                flag = true;
                continue;
            }
            else {
                tmp.push_back(a[i]);
            }
        }
        if (flag) {
#if DEBUGSHOW > 2
            Mat s1 = org.clone();
            for (auto box : a) {

                cv::rectangle(s1, box, cv::Scalar(0, 0, 255));
            }
            imshow("s1", s1);
            waitKey();
            destroyWindow("s1");
#endif
        }
        swap(tmp, a);
        tmp.clear();
        flag = false;
        for (int i = 0; i < a.size(); i++) {
            int dh = i + 1<a.size()? a[i].y + a[i].height - a[i + 1].y:0;
            if (i+1<a.size()&&isOverlap(a[i], a[i + 1]) && (i == 0 || !isOverlap(a[i - 1], a[i]))&& (dh > 0.8*a[i].height || dh > 0.8*a[i + 1].height)) {
                tmp.push_back(a[i] | a[i + 1]);
                flag = true;
                i++;
            }
            else {
                tmp.push_back(a[i]);
            }
        }

        if (flag) {
#if DEBUGSHOW > 2
            Mat s2 = org.clone();
            for (auto box : a) {

                cv::rectangle(s2, box, cv::Scalar(0, 0, 255));
            }
            imshow("s2", s2);
            waitKey();
            destroyWindow("s2");
#endif
        }
        swap(tmp, a);
    }

    int lastcheck(vector<Rect>& a) {
        if (a.size() != 11) return false;
        int h = 0;
        for (int i = 0; i < 11; i++) {
            h += a[i].height;
        }
        h /= 11;
        h += 1;
        sort(a.begin(), a.end(), sortByY);
        for (int i = 0; i < 10; i++) {
            if (isOverlap(a[i], a[i + 1])) {
                int dh = a[i].y + a[i].height - a[i + 1].y;
                if (dh >= 0.2*h || dh >= 0.2*h) return false;
            }
        }
        vector<Point> vpt;
        for (auto it : a) {
            vpt.push_back(Point(it.x + it.width / 2, it.y + it.height / 2));
        }
        Vec4f paras;
        fitLine(vpt, paras, DIST_L1, 0, 1e-2, 1e-2);
        double dy = paras[1];
        double dx = paras[0];
        if (dy > -0.97 && dy < 0.99) return false;
        return true;

    }


    void test() {
        sort(dp.begin(), dp.end(), sortByX);
        vector<vector<Rect>> res;
        int end = -1;
        vector<Rect> tmp;
        for (auto it : dp) {
            if (it.x < 0.3*org.cols) continue;
            if (it.x > end) {
                if (check(tmp)) res.push_back(tmp);
                end = it.x + it.width;
                tmp.clear();
                tmp.push_back(it);
            }
            else {
                tmp.push_back(it);
                end = max(end, it.x + it.width);
            }
        }
        if (check(tmp)) res.push_back(tmp);
        Mat img = org.clone();
        int flag = 0;
        for (auto& rects : res) {
            cv::Scalar color(0, 0, 255);

            int last = 0;
            tmp.clear();
            while (true) {
                int num = 0;
                vector<Point> vpt;
                for (auto it : rects) {
                    vpt.push_back(Point(it.x + it.width / 2, it.y + it.height / 2));
                }
                Vec4f paras;
                fitLine(vpt, paras, DIST_L1, 0, 1e-2, 1e-2);
                cout << paras << endl;
                Point2f p0(paras[2], paras[3]);
                double y1 = p0.y + paras[1] * 500;
                double x1 = p0.x + paras[0] * 500;
                double y2 = p0.y - paras[1] * 500;
                double x2 = p0.x - paras[0] * 500;
               
                line(img, Point2f(x1, y1), Point2f(x2, y2), cv::Scalar(0, 255, 0));
                double dy = paras[1];
                double dx = paras[0];
                double x0 = paras[2];
                double y0 = paras[3];
                tmp.clear();
                for (auto it : rects) {
                    double x = it.x + it.width / 2;
                    double y = it.y + it.height / 2;
                    double dist = abs(dy * x - dx * y + dx * y0 - dy * x0);
                    //cout << dist << endl;
                    if (dist <= it.width/2) {
                        tmp.push_back(it);
                    }
                }
                rects.clear();
                for (auto it : tmp) {
                    rects.push_back(it);
                }
                //line(img, p0, p1, cv::Scalar(255, 0, 0), 2, 8, 0);
                if (paras[1] >= -1 && paras[1] <= -0.98) flag = 1;
                else flag = 2;
                break;
            }
            tmp.clear();
            cout << rects.size() << endl;
            sort(rects.begin(), rects.end(), sortByArea);
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
            lastchance(rects);
            cout << rects.size() << endl;
            result = rects.size();
            for (auto rect : rects)
                cv::rectangle(img, rect, color, 1);
        }
#if DEBUGSHOW
    imshow("result", img);
    waitKey();
    destroyWindow("result");
#elif
        if (!(res.size() == 1 && res[0].size() == 11)) {
            imshow("show", img);
            waitKey();
            destroyWindow("show");
        }
#endif
#if 0
        string name = "_";
        for (auto ch : testpath) {
            if (ch >= '0'&&ch <= '9') name += ch;
        }
        name.erase(1, 1);
        if (flag==2) {
            s2++;
            fstream out(savefilepath, ios::app);
            out << s2<<" "<<testpath << endl;
            imwrite("C:\\Users\\archlab\\Desktop\\showImg\\s2\\" + to_string(s2) + name + ".jpg", img);
        }
        else if (flag == 1) {
            s1++;
            imwrite("C:\\Users\\archlab\\Desktop\\showImg\\s1\\" + to_string(s1) + name + ".jpg", img);
        }
        else {
            s0++;
            fstream out(savefilepath2, ios::app);
            out << s0<<" "<<testpath << endl;
            imwrite("C:\\Users\\archlab\\Desktop\\showImg\\s0\\" + to_string(s0) + name + ".jpg", img);
        }
        cout << s0 << " " << s1 << " " << s2 << endl;
#endif
    }
   

}
string gateDetect() {
    org = imread(testpath);
    cvtColor(org, gray, CV_BGR2GRAY);
    show = org.clone();
    mser();
    return "";
}

string gateDetect(string imgpath) {
#if DEBUGSHOW > 1
    imgpath = testpath;
#endif // DEBUGSHOW > 1
    boost::filesystem::path path_file(imgpath);
    if (boost::filesystem::exists(path_file))      cout << "boost: path exist." << endl;
    else                                                   cout << "boost: path not exist." << endl;
    
    //testpath = imgpath;
    org = imread(imgpath);
    Rect deeplab_box;
    string deeplab_imgpath(imgpath);
    deeplab_imgpath.replace(deeplab_imgpath.find(".jpg"), 4, "_dl.jpg");
    initSocket();
    deeplab_request(imgpath, deeplab_box);
    org = org(deeplab_box);
    cvtColor(org, gray, CV_BGR2GRAY);
    show = org.clone();
    result = -1;
    lvl = 5;
    minarea = 50;
    maxd = 0.6;
    mser();
    test();
    //if (result == -1) return "";
    while (result > 11&&lvl<=20) {
        lvl += 5;
        mser();
        test();
    }

    while (result < 11&&lvl>1) {
        lvl--;
        mser();
        test();
    }

    while (result < 11 && minarea > 30) {
        minarea -= 10;
        mser();
        test();
    }

    if (result > 11) {

    }
    /*
    while (result != 11 && maxd < 5) {
        maxd *= 2;
        mser();
        test();
    }
    */
    DeleteFile(deeplab_imgpath.c_str());
    return "";
}
