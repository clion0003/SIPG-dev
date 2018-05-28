#ifndef __GATE_UTILS_H
#define __GATE_UTILS_H

#include <string>

#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/stitching.hpp>

namespace utils{
    extern bool ifFileExist(std::string filepath);
    extern bool sortByArea(const cv::Rect& a, const cv::Rect& b);
    extern bool sortByX(const cv::Rect& a, const cv::Rect& b);
    extern bool sortByY(const cv::Rect& a, const cv::Rect& b);
    extern bool isOverlap(const cv::Rect& a, const cv::Rect& b);
}

#endif // !__GATE_UTILS_H

