#include "side_utils.h"

vector<string> getDirFileNames(string dirPath)
{
	intptr_t handle;
	_finddata_t findData;
	vector<string> files;

	char Path[200] = { 0 };
	dirPath = dirPath + "/*.*";
	strcpy(Path, dirPath.c_str());

	handle = _findfirst(Path, &findData);    // 查找目录中的第一个文件
	for (int i = 0; ; i++)
	{
		if (_findnext(handle, &findData) != 0)
			break;
		
		if (strcmp(findData.name, "..")!=0 && strcmp(findData.name, ".")!=0)
			files.push_back(string(findData.name));
	}
	
	_findclose(handle);    // 关闭搜索句柄
	return files;
}

// 近计算一组Rect的centerX和centerY的方差
double heightSD(vector<cv::Rect> c)
{
	int charH = getClusAvgH(c, 0);
	double var = 0;
	for (int i = 0; i < c.size(); i++)
		var = var + (c[i].height - charH) * (c[i].height - charH);
	var = var / c.size();
	return sqrt(var) / charH;
}

double yCenterSD(vector<cv::Rect> c)
{
	return 0;
}

// 把矩形按照x的值进行升序排列
bool sideSortByX(cv::Rect r1, cv::Rect r2)
{
	return r1.x < r2.x;
}

// 把矩形按照y的值进行升序排列
bool sideSortByY(cv::Rect r1, cv::Rect r2)
{
	return r1.y < r2.y;
}

bool sortByH(cv::Rect r1, cv::Rect r2)
{
	return r1.height < r2.height;
}

bool sortByW(cv::Rect r1, cv::Rect r2)
{
	return r1.width < r2.width;
}

bool clusSortByY(vector<cv::Rect> c1, vector<cv::Rect> c2)
{
	//sort(c1.begin(), c1.end(), sortByY);
	//sort(c2.begin(), c2.end(), sortByY);
	return c1[0].y < c2[0].y;
}

// 把vector按照长度进行降序排列
bool sortByLen(vector<cv::Rect> v1, vector<cv::Rect> v2)
{
	return v1.size() > v2.size();
}

cv::Rect getRowClusBbox(vector<cv::Rect> cluster)
{
	cv::Rect bbox(0, 0, 0, 0);
	if (cluster.size() == 0)
		return bbox;
	int minX = 10000, minY = 10000, maxX = 0, maxY = 0;
	int charH = getClusTypicalH(cluster);
	for (int i = 0; i < cluster.size(); i++)
	{
		minX = min(minX, cluster[i].x);
		maxX = max(maxX, cluster[i].x + cluster[i].width);
		if (cluster[i].height < 1.3*charH)	//对于高度异常的Rect不计算在内
		{
			minY = min(minY, cluster[i].y);
			maxY = max(maxY, cluster[i].y + cluster[i].height);
		}
	}
	bbox.x = minX;
	bbox.y = minY;
	bbox.width = maxX - minX;
	bbox.height = maxY - minY;
	return bbox;
}

cv::Rect getColClusBbox(vector<cv::Rect> cluster)
{
	cv::Rect bbox(0, 0, 0, 0);
	if (cluster.size() == 0)
		return bbox;
	int minX = 10000, minY = 10000, maxX = 0, maxY = 0;
	int charH = getClusTypicalH(cluster);
	for (int i = 0; i < cluster.size(); i++)
	{
		minX = min(minX, cluster[i].x);
		maxX = max(maxX, cluster[i].x + cluster[i].width);
		minY = min(minY, cluster[i].y);
		maxY = max(maxY, cluster[i].y + cluster[i].height);
	}
	bbox.x = minX;
	bbox.y = minY;
	bbox.width = maxX - minX;
	bbox.height = maxY - minY;
	return bbox;
}

int getClusAvgH(vector<cv::Rect> cluster, int deOutlier)	//会去除一些异常值
{
	if (cluster.size() == 0)
		return 0;
	int avgHeight = 0;
	if (2 * deOutlier >= cluster.size())
		deOutlier = (cluster.size() - 1) / 2;
	sort(cluster.begin(), cluster.end(), sortByH);
	for (int j = deOutlier; j < cluster.size()-deOutlier; j++)
		avgHeight += cluster[j].height;
	avgHeight = avgHeight / (cluster.size()-2*deOutlier);
	return avgHeight;
}

int getClusAvgW(vector<cv::Rect> cluster, int deOutlier)
{
	if (cluster.size() == 0)
		return 0;
	int avgWidth = 0;
	if (2 * deOutlier >= cluster.size())
		deOutlier = (cluster.size() - 1) / 2;
	sort(cluster.begin(), cluster.end(), sortByW);
	for (int j = deOutlier; j < cluster.size() - deOutlier; j++)
		avgWidth += cluster[j].width;
	avgWidth = avgWidth / (cluster.size() - 2 * deOutlier);
	return avgWidth;
}

//经验表明，一个cluster中比正常值小的bbox往往有很多，但是大的却不多，所以取出几个大的值之后的那个作为typical value
int getClusTypicalH(vector<cv::Rect> cluster)
{
	if (cluster.size() == 0)
		return 0;
	sort(cluster.begin(), cluster.end(), sortByH);
	int deOutlier = 0;
	if (cluster.size() > 2 && cluster.size() < 6)
		deOutlier = 1;
	if (cluster.size() >= 6 && cluster.size() < 9)
		deOutlier = 2;
	if (cluster.size() >= 9 && cluster.size() < 12)
		deOutlier = 3;
	if (cluster.size() >= 12)
		deOutlier = 4;
	return cluster[cluster.size() - 1 - deOutlier].height;
}

int getClusTypicalW(vector<cv::Rect> cluster)
{
	if (cluster.size() == 0)
		return 0;
	sort(cluster.begin(), cluster.end(), sortByW);
	int deOutlier = 0;
	if (cluster.size() > 2 && cluster.size() < 6)
		deOutlier = 1;
	if (cluster.size() >= 6)
		deOutlier = 2;
	return cluster[cluster.size() - 1 - deOutlier].width;
}

void clearDir(string path)
{
	boost::filesystem::path rootPath(path, boost::filesystem::native);
	boost::filesystem::directory_iterator end_iter;
	for (boost::filesystem::directory_iterator iter(rootPath); iter != end_iter; iter++)
		boost::filesystem::remove(iter->path());
}

int getMaxClusIdx(vector<vector<cv::Rect>> clus)
{
	int maxLen = 0, idx = 0;
	for (int i = 0; i < clus.size(); i++)
		if (maxLen <= clus[i].size())
		{
			maxLen = clus[i].size();
			idx = i;
		}
	return idx;
}

int getMinYDis(cv::Rect r1, cv::Rect r2)
{
	if (r1.y < r2.y)
		return r2.y - r1.y - r1.height > 0 ? r2.y - r1.y - r1.height : 0;
	else
		return r1.y - r2.y - r2.height > 0 ? r1.y - r2.y - r2.height : 0;
}
