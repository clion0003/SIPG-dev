#include "gateUtils.h"

using namespace std;
using namespace cv;

namespace utils {
    bool ifFileExist(string filepath)
    {
        boost::filesystem::path path_file(filepath);
        if (boost::filesystem::exists(path_file))
            return true;
        else
            return false;
    }

    bool sortByArea(const Rect& lhs, const Rect& rhs) {
        return lhs.area() < rhs.area();
    }

    bool sortByX(const Rect& a, const Rect& b) {
        return a.x < b.x;
    }

    bool sortByY(const Rect& a, const Rect& b) {
        return a.y < b.y;
    }

    bool isOverlap(const Rect& a, const Rect& b) {
        return a.x < b.x + b.width&&b.x < a.x + a.width&&a.y < b.y + b.height&&b.y < a.y + a.height;
    }
}
