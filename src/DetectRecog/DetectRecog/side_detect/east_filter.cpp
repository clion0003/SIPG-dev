#include "east_filter.h"

using namespace cv;
using namespace std;

EastFilter::EastFilter(string imgPath, Rect deeplabBbox)
{
	initSocket();
	east_request(imgPath, eastBboxes);
	this->srcImg = imread(imgPath);
	this->deeplabBbox = deeplabBbox;
	for (int i = 0; i < eastBboxes.size(); i++)
		this->bboxes.push_back(eastbox2rect(eastBboxes[i]));
	
	//设定阈值
	double heightRatio = (double(deeplabBbox.height)) / srcImg.rows;
	//对于另一个码头摄像机位置有时候过高进行一次校正
	if (heightRatio < 0.5 && deeplabBbox.width>2 * deeplabBbox.height && deeplabBbox.y > srcImg.rows / 2)
		heightRatio += 0.2;
	switch ((int)(heightRatio * 10))
	{
	case 1:
	case 2:
	case 3:
	case 4: eastHeightSmall = 15; eastHeightHuge = 27; eastWidthHuge = 20; break;
	case 5: eastHeightSmall = 18; eastHeightHuge = 35; eastWidthHuge = 25; break;
	case 6: eastHeightSmall = 20; eastHeightHuge = 50; eastWidthHuge = 32; break;
	case 7: eastHeightSmall = 22; eastHeightHuge = 52; eastWidthHuge = 38; break;
	case 8:
	case 9: eastHeightSmall = 25; eastHeightHuge = 55; eastWidthHuge = 42; break;
	}
}

EastFilter::EastFilter(string imgPath, Rect deeplabBbox, vector<EastBbox> eastBboxes)
{
	this->srcImg = imread(imgPath);
	this->deeplabBbox = deeplabBbox;
	for (int i = 0; i < eastBboxes.size(); i++)
		this->bboxes.push_back(eastbox2rect(eastBboxes[i]));

	//设定阈值
	double heightRatio = (double(deeplabBbox.height)) / srcImg.rows;
	//对于另一个码头摄像机位置有时候过高进行一次校正
	if (heightRatio < 0.5 && deeplabBbox.width>2 * deeplabBbox.height && deeplabBbox.y > srcImg.rows / 2)
		heightRatio += 0.2;
	switch ((int)(heightRatio * 10))
	{
	case 1:
	case 2:
	case 3:
	case 4: eastHeightSmall = 15; eastHeightHuge = 27; eastWidthHuge = 20; break;
	case 5: eastHeightSmall = 18; eastHeightHuge = 35; eastWidthHuge = 25; break;
	case 6: eastHeightSmall = 20; eastHeightHuge = 50; eastWidthHuge = 32; break;
	case 7: eastHeightSmall = 22; eastHeightHuge = 52; eastWidthHuge = 38; break;
	case 8:
	case 9: eastHeightSmall = 25; eastHeightHuge = 55; eastWidthHuge = 42; break;
	}
}

void EastFilter::drawBboxes(string outputPath, vector<Rect> bboxes)
{
	Mat eastImg = srcImg.clone();
	for (int i = 0; i < bboxes.size(); i++)
	{
		Rect bbox = bboxes[i];
		rectangle(eastImg, bbox, Scalar(0, 255, 0), 1, 8, 0);
	}
	rectangle(eastImg, deeplabBbox, Scalar(0, 255, 0), 1, 8, 0);
	imwrite(outputPath, eastImg);
}

void EastFilter::drawBboxes(string outputPath, vector<vector<Rect>> clus)
{
	Mat eastImg = srcImg.clone();
	for (int i = 0; i < clus.size(); i++)
	{
		for (int j = 0; j < clus[i].size(); j++)
		{
			Rect bbox = clus[i][j];
			rectangle(eastImg, bbox, Scalar(0, 255, 0), 1, 8, 0);
		}
		
	}
	rectangle(eastImg, deeplabBbox, Scalar(0, 255, 0), 1, 8, 0);
	imwrite(outputPath, eastImg);
}

void EastFilter::filter(void)
{
	deeplabFilter();
	singleFilter();
	clusterFilter();
}

void EastFilter::deeplabFilter(void)
{
	for (int i = 0; bboxes.begin() + i != bboxes.end(); i++)
	{
		if (bboxes[i].x < deeplabBbox.x || bboxes[i].x + bboxes[i].width > deeplabBbox.x + deeplabBbox.width
			|| bboxes[i].y < deeplabBbox.y || bboxes[i].y + bboxes[i].height > deeplabBbox.y + deeplabBbox.height)
		{
			bboxes.erase(bboxes.begin() + i);
			i--;
		}
	}
}

void EastFilter::singleFilter(void)
{
	for (int i = 0; bboxes.begin() + i != bboxes.end(); i++)
	{
		if (bboxes[i].area() > eastHeightHuge*eastWidthHuge*15 || bboxes[i].height < eastHeightSmall || bboxes[i].height > 1.3*eastHeightHuge //特别大/小的删掉
			|| (!(deeplabBbox.y + deeplabBbox.height > 0.99*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5) && (bboxes[i].y-deeplabBbox.y)*1.0/deeplabBbox.height > 0.4)) //水平bbox中特别靠下的删掉					
		{
			bboxes.erase(bboxes.begin() + i);
			i--;
		}
	}
}

void EastFilter::clusterFilter(void)
{
	cluster(2.5*eastWidthHuge, bboxes, clus, ROW_COL_MODE);
	vector<Rect> mainRegion;
	findMainRegion(mainRegion);
	//drawBboxes(debugSavePrefix + "_2.jpg", mainRegion);

	buildResult(mainRegion);
	drawBboxes(debugSavePrefix + "_3.jpg", clus);
}

void EastFilter::findMainRegion(vector<Rect>& mainRegion)	//找到一个包含箱号信息和
{
	for (int i = 0; i < clus.size(); i++)
		if (mainRegion.size() < clus[i].size())
		{
			if(mainRegion.size() < 3)
				mainRegion.assign(clus[i].begin(), clus[i].end());
			if (mainRegion.size() >= 3)
			{
				sort(mainRegion.begin(), mainRegion.end(), sideSortByY);
				sort(clus[i].begin(), clus[i].end(), sideSortByY);
				if(clus[i][0].y < mainRegion[0].y)
					mainRegion.assign(clus[i].begin(), clus[i].end());
			}
		}
}

void EastFilter::buildResult(vector<Rect> mainRegion)
{
	clus.clear();
	clus.resize(3);
	if (mainRegion.size() == 0)
		return;

	vector<vector<Rect>> tmpClus;
	cluster(2.5*eastWidthHuge, mainRegion, tmpClus, ROW_MODE);
	sort(tmpClus.begin(), tmpClus.end(), clusSortByY);

	int mainRowIdx = 0;	//先找到mainRow
	for (int i = 1; i < tmpClus.size(); i++)
	{
		Rect mainRowBbox = getRowClusBbox(tmpClus[mainRowIdx]);
		Rect tmpBbox = getRowClusBbox(tmpClus[i]);
		if (tmpBbox.width > mainRowBbox.width)
			mainRowIdx = i;
	}

	sort(tmpClus[mainRowIdx].begin(), tmpClus[mainRowIdx].end(), sideSortByX);
	clus[1].assign(tmpClus[mainRowIdx].begin(), tmpClus[mainRowIdx].end());
	if (mainRowIdx - 1 >= 0)
	{
		sort(tmpClus[mainRowIdx-1].begin(), tmpClus[mainRowIdx-1].end(), sideSortByX);
		clus[0].assign(tmpClus[mainRowIdx - 1].begin(), tmpClus[mainRowIdx - 1].end());
	}
	if (mainRowIdx + 1 < tmpClus.size())
	{
		sort(tmpClus[mainRowIdx + 1].begin(), tmpClus[mainRowIdx + 1].end(), sideSortByX);
		clus[2].assign(tmpClus[mainRowIdx + 1].begin(), tmpClus[mainRowIdx + 1].end());
	}
}

Rect EastFilter::eastbox2rect(EastBbox bbox)
{
	int x = min(bbox.x0, bbox.x3);
	int y = min(bbox.y0, bbox.y1);
	int width = max(bbox.x1, bbox.x2) - x;
	int height = max(bbox.y2, bbox.y3) - y;
	return Rect(x, y, width, height);
}

void EastFilter::cluster(int disThres, vector<cv::Rect> bboxes, vector<vector<cv::Rect>>& clus, int mode)
{
	// idx中存的是每个bbox所属的类，初始时每个bbox都属于自己一类
	vector<int> idx(bboxes.size());
	for (int i = 0; i < idx.size(); i++)
		idx[i] = i;

	// 开始聚类，此for循环结果是一个和bboxes等长的数组，数组中元素表示对应的bbox所属的类
	for (int i = 0; i < bboxes.size(); i++)
		for (int j = i + 1; j < bboxes.size(); j++)
			if (isClusterable(bboxes[i], bboxes[j], disThres, mode) == true)
			{
				if (idx[j] == idx[i])	//已在同一类中
					continue;
				else if (idx[j] == j)	//无所属类
					idx[j] = idx[i];
				else                    //close但不属于一个类，合并这两个类
				{
					int srcClas = idx[j]>idx[i] ? idx[j] : idx[i];
					int desClas = idx[i] == srcClas ? idx[j] : idx[i];
					for (int k = 0; k < idx.size(); k++)
						if (idx[k] == srcClas)
							idx[k] = desClas;
				}
			}

	// 将聚类结果用顺序表来保存，一个类对应一个顺序表，更方便之后的操作
	// 此处避免使用多层for循环
	clus.clear();
	clus.resize(idx.size());
	for (int i = 0; i < idx.size(); i++)
		clus[idx[i]].push_back(bboxes[i]);
	for (int i = 0; clus.begin() + i != clus.end(); i++)
	{
		if (clus[i].size() <= 0)
		{
			clus.erase(clus.begin() + i);
			i--;
		}
	}
}

bool EastFilter::isClusterable(Rect r1, Rect r2, int disThres, int mode)
{
	int r1CenterX = r1.x + r1.width / 2, r1CenterY = r1.y + r1.height / 2;
	int r2CenterX = r2.x + r2.width / 2, r2CenterY = r2.y + r2.height / 2;

	if (mode == ROW_MODE || mode == ROW_COL_MODE)
		if (max(abs(r1.y - r2.y - r2.height), abs(r2.y - r1.y - r1.height)) < r1.height + r2.height)	//两者y方向有重合
		{
			double yCover = min(abs(r1.y - r2.y - r2.height), abs(r2.y - r1.y - r1.height));//两个rect之间y方向重合的长度
			if ((yCover / r1.height > 0.7 || yCover / r2.height > 0.7) && (r1.height*1.0 / r2.height < 1.7 && r2.height*1.0 / r1.height < 1.7))	//两者height差别不大
				if (abs(r1CenterX - r2CenterX) - 0.5*(r1.width + r2.width) < disThres) //两者distance
						return true;
		}
	if (mode == ROW_COL_MODE)
		if (max(abs(r1.x - r2.x - r2.width), abs(r2.x - r1.x - r1.width)) < r1.width + r2.width)	//两者x方向有重合
		{
			double xCover = min(abs(r1.x - r2.x - r2.width), abs(r2.x - r1.x - r1.width));//两个rect之间y方向重合的长度
			if (xCover / r1.width > 0.5 || xCover / r2.width > 0.5)	//两者height差别不大
				if (abs(r1CenterY - r2CenterY) - 0.5*(r1.height + r2.height) < 1.3*eastHeightHuge) //两者distance
					return true;
		}
	return false;
}