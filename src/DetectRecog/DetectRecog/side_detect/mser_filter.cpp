#include "mser_filter.h"
using namespace cv;

MserFilter::MserFilter(Mat srcImg, Rect deeplabBbox)
{
	Mat grayImg = Mat::zeros(srcImg.size(), CV_8UC1);
	cvtColor(srcImg, grayImg, COLOR_BGR2GRAY);
	this->srcImg = srcImg;
	this->deeplabBbox = deeplabBbox;
	Ptr<MSER> mser = MSER::create(5, 50, 500, 0.4);
	mser->detectRegions(grayImg, msers, bboxes);
	

	//设定阈值
	double heightRatio = (double(deeplabBbox.height)) / srcImg.rows;
	//对于另一个码头摄像机位置有时候过高进行一次校正
	if (deeplabBbox.y+deeplabBbox.height > 0.99*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5)
		heightRatio += 0.2;
	//摄像头位置过低的校正
	switch ((int)(heightRatio*10))
	{
	case 1:
	case 2:
	case 3: 
	case 4: mserHeightSmall = 15; mserHeightHuge = 27; mserWidthHuge = 20; break;
	case 5: mserHeightSmall = 18; mserHeightHuge = 35; mserWidthHuge = 25; break;
	case 6: mserHeightSmall = 20; mserHeightHuge = 50; mserWidthHuge = 32; break;
	case 7: mserHeightSmall = 22; mserHeightHuge = 52; mserWidthHuge = 38; break;
	case 8:
	case 9: mserHeightSmall = 25; mserHeightHuge = 55; mserWidthHuge = 42; break;
	}
}

void MserFilter::drawMser(string outputPath)
{
	Mat mserImg = Mat::zeros(srcImg.size(), CV_8UC1);
	for (int i = 0; i < msers.size(); i++)
	{
		vector<Point>& r = msers[i];
		for (int j = 0; j < (int)r.size(); j++)
		{
			Point pt = r[j];
			mserImg.at<unsigned char>(pt) = 255;
		}
	}
	rectangle(mserImg, deeplabBbox, 255, 1, 8, 0);
	imwrite(outputPath, mserImg);
}

// 在全黑图上画bbox
void MserFilter::drawBboxes(string outputPath, vector<Rect> bboxes)
{
	Mat mserImg = Mat::zeros(srcImg.size(), CV_8UC1);
	for (int i = 0; i < bboxes.size(); i++)
	{
		Rect bbox = bboxes[i];
		rectangle(mserImg, bbox, 255, 1, 8, 0);
	}
	rectangle(mserImg, Rect(0, 0, int(mserWidthHuge), int(mserHeightSmall)), 255, 1, 8, 0);
	rectangle(mserImg, Rect(0, 0, int(mserWidthHuge), int(mserHeightHuge)), 255, 1, 8, 0);
	rectangle(mserImg, deeplabBbox, 255, 1, 8, 0);
	imwrite(outputPath, mserImg);
}

// 在一张全黑图中画聚类情况
void MserFilter::drawClus(string outputPath, vector<vector<Rect>> clus)
{
	Mat mserImg = Mat::zeros(srcImg.size(), CV_8UC1);
	int clusNum = 0;
	for (int i = 0; i < clus.size(); i++)
		if (clus[i].size() != 0)
			clusNum++;

	int clusCnt = 0;
	for (int i = 0; i < clus.size(); i++)
	{
		if(clus[i].size() != 0)
		{
			clusCnt++;
			int color = (240 / clusNum) * clusCnt;
			for (int j = 0; j < clus[i].size(); j++)
			{
				//color = 250;
				rectangle(mserImg, clus[i][j], color, CV_FILLED, 8, 0);
			}
		}
	}
	rectangle(mserImg, Rect(0, 0, int(mserWidthHuge), int(mserHeightSmall)), 255, 1, 8, 0);
	rectangle(mserImg, Rect(0, 0, int(mserWidthHuge), int(mserHeightHuge)), 255, 1, 8, 0);
	rectangle(mserImg, deeplabBbox, 255, 1, 8, 0);
	imwrite(outputPath, mserImg);
}

void MserFilter::drawOnSrcImg(string outputPath, vector<vector<cv::Rect>> clus)
{
	Mat resImg = srcImg.clone();
	for (int i = 0; i < clus.size(); i++)
		for (int j = 0; j < clus[i].size(); j++)
			rectangle(resImg, clus[i][j], Scalar(0, 255, 0), 1, 8, 0);
	rectangle(resImg, deeplabBbox, Scalar(0, 255, 0), 1, 8, 0);
	imwrite(outputPath, resImg);
}

int MserFilter::filter(void)
{
	deeplabFilter();
	singleFilter();
	clusFilter();
	if (rowClus.size() == 0 && colClus.size() != 0)
		return COL_MODE;
	else if (colClus.size() == 0 && rowClus.size() != 0)
		return ROW_MODE;
	else
		return WRONG;
}

void MserFilter::deeplabFilter(void) 
{
	for (int i = 0; bboxes.begin() + i != bboxes.end(); i++)	//超出deeplab bbox的删除
	{
		if (bboxes[i].x < deeplabBbox.x || bboxes[i].x + bboxes[i].width > deeplabBbox.x + deeplabBbox.width
			|| bboxes[i].y < deeplabBbox.y || bboxes[i].y + bboxes[i].height > deeplabBbox.y + deeplabBbox.height)
		{
			bboxes.erase(bboxes.begin() + i);
			i--;
		}
	}
}

void MserFilter::singleFilter(void)
{
	for (int i = 0; bboxes.begin() + i != bboxes.end(); i++)
	{
		// size明显不是文字的不要
		if (bboxes[i].height < bboxes[i].width || bboxes[i].height <= mserHeightSmall
			|| bboxes[i].height >= mserHeightHuge || bboxes[i].width >= mserWidthHuge
			|| bboxes[i].height > 10 * bboxes[i].width)
		{
			bboxes.erase(bboxes.begin() + i);
			i--;
		}
	}
	// 删除重叠矩形
	for (int i = 0; bboxes.begin() + i != bboxes.end(); i++)
		for (int j = i + 1; bboxes.begin() + j != bboxes.end(); j++)
		{
			Rect intersec = bboxes[i] & bboxes[j];
			if ((intersec.width > 0.8*bboxes[i].width && intersec.height > bboxes[i].height*0.8)
				|| (intersec.width > 0.8*bboxes[j].width && intersec.height > bboxes[j].height*0.8))
			{
				if (bboxes[i].area() < bboxes[j].area())
				{
					bboxes[j] = bboxes[i] | bboxes[j];
					bboxes.erase(bboxes.begin() + i);
					i--;
					break;
				}
				else
				{
					bboxes[i] = bboxes[i] | bboxes[j];
					bboxes.erase(bboxes.begin() + j);
					j--;
				}
			}
		}
}

void MserFilter::clusFilter(void)
{
	// 将bbox分两个方向聚类，聚类的结果为MserFilter成员
	cluster(mserWidthHuge*2.5, bboxes, rowClus, ROW_MODE);
	cluster(mserHeightHuge*4, bboxes, colClus, COL_MODE);
	
	int mainRowSize = clusProcess(ROW_MODE);
	int mainColsize = clusProcess(COL_MODE);

	if (mainRowSize==0 && mainColsize==0)	//都为0
	{
		rowClus.clear();
		colClus.clear();
	}
	else if(mainRowSize==0 && mainColsize != 0)	//有一者为0
		rowClus.clear();
	else if (mainColsize == 0 && mainRowSize != 0)
		colClus.clear();
	else	//均不为0
	{
		if((mainColsize >= 6 && mainRowSize <= 4) || mainColsize >= 11 || mainColsize / mainRowSize >= 2)	// col聚类得到的结果更好，保留col类
			rowClus.clear();
		else if ((mainRowSize >= 6 && mainColsize <= 4) || mainRowSize >= 11 || mainRowSize / mainColsize >= 2)
			colClus.clear();
		else	// hard，判断谁更在右边一点
		{
			if (rowClus[0][0].x + 0.5*rowClus[0][0].width >= colClus[0][0].x)
				colClus.clear();
			else
				rowClus.clear();
		}
	}
}

// 行聚类或列聚类，取决于mode，mode会传给isClose()函数判定是否属于一类(行方向或者列方向)
void MserFilter::cluster(int disThres, vector<Rect> bboxes, vector<vector<Rect>>& clus,int mode)
{
	// idx中存的是每个bbox所属的类，初始时每个bbox都属于自己一类
	vector<int> idx(bboxes.size());
	for (int i = 0; i < idx.size(); i++)
		idx[i] = i;

	// 开始聚类，此for循环结果是一个和bboxes等长的数组，数组中元素表示对应的bbox所属的类
	for (int i = 0; i < bboxes.size(); i++)
		for (int j = i + 1; j < bboxes.size(); j++)
			if (isClose(bboxes[i], bboxes[j], disThres, mode) == true)
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
			clus.erase(clus.begin()+i);
			i--;
		}
	}
}

//返回其最长的rowClus的size
int MserFilter::clusProcess(int mode)
{
	vector<Rect> mainClus;
	
	if (mode == ROW_MODE)	//找到最长的类
	{
		findMainRow(mainClus);
		if (mainClus.size() <= 4)
			return 0;
		getByMainRow(mainClus);
		return buildRowResult();
	}
	else
	{
		int mainCharH, mainCharW;
		findMainCol(mainClus, mainCharH, mainCharW);
		if (mainClus.size() <= 4)
			return 0;
		getByMainCol(mainClus, mainCharH, mainCharW);
		return buildColResult();
	}
}

void MserFilter::findMainRow(vector<Rect>& mainRow)
{
	//先删除比较靠下的类,避免干扰
	vector<vector<Rect>> clusTmp(rowClus);
	for (int i = 0; clusTmp.begin() + i != clusTmp.end(); i++)
		//对于摄像机位置过高导致图片有一半在镜头外面时，不做此种处理
		if (!(deeplabBbox.y + deeplabBbox.height > 0.98*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5))
			if ((clusTmp[i][0].y - deeplabBbox.y)*1.0 / deeplabBbox.height > 0.35)
			{
				clusTmp.erase(clusTmp.begin() + i);
				i--;
			}
	
	if (clusTmp.size() == 0)
		return;

	// 对类中小bbox进行合并，异常的bbox进行删除
	for (int i = 0; i < clusTmp.size(); i++)
	{
		mergeRowSmallBbox(clusTmp[i]);
		rmRowOutlier(clusTmp[i]);
	}

	// 从长到短排序，默认最长的那个作为mainRow
	int mainRowIdx = 0;
	sort(clusTmp.begin(), clusTmp.end(), sortByLen);
	if (clusTmp.size() > 1)
		if (clusTmp[0].size() >= 6 && clusTmp[1].size() >= 6)	//若这两个都很长，均有可能是mainRow
		{
			Rect clusBbox0 = getRowClusBbox(clusTmp[0]), clusBbox1 = getRowClusBbox(clusTmp[1]);
			if (abs(clusBbox0.x+0.5*clusBbox0.width-clusBbox1.x-clusBbox1.width*0.5) > max(clusBbox0.width, clusBbox1.width))	//如果这两者X方向距离很远，选择方差小的那个
			{
				if (heightSD(clusTmp[1]) < heightSD(clusTmp[0]))
					mainRowIdx = 1;
			}
			else // 否则选择clusBbox更宽的那个
			{
				if (clusBbox1.width > clusBbox0.width)
					mainRowIdx = 1;
			}
		}
	mainRow.assign(clusTmp[mainRowIdx].begin(), clusTmp[mainRowIdx].end());
}

void MserFilter::getByMainRow(vector<Rect> mainRow)
{		
	//计算mainRowClus的相关参数
	Rect mainBbox = getRowClusBbox(mainRow);
	int mainCharH = getClusTypicalH(mainRow);	//去除一些异常值再求均值
	int mainCharW = getClusTypicalW(mainRow);
	int mainLeftBorder = mainBbox.x - 8 * mainCharW, mainRightBorder = mainBbox.x + mainBbox.width + 8 * mainCharW;
	int mainUpBorder = mainBbox.y - mainCharH * 2, mainDownBorder = mainBbox.y + mainBbox.height + mainCharH * 2.5;

	mainRegion.x = mainLeftBorder; mainRegion.width = mainRightBorder - mainLeftBorder;
	mainRegion.y = mainUpBorder; mainRegion.height = mainDownBorder - mainUpBorder;

	vector<Rect> up, down;	//分别表示上方的公司号和下方的箱型
	for (int i = 0; rowClus.begin() + i != rowClus.end(); i++)
	{
		mergeRowSmallBbox(rowClus[i]);	//先对这个clus进行简单的处理
		rmRowOutlier(rowClus[i]);
		Rect clusBbox = getRowClusBbox(rowClus[i]);	//获取一个clus的相关参数
		int yCenter = clusBbox.y + clusBbox.height * 0.5, xCenter = clusBbox.x + clusBbox.width*0.5;

		if (clusBbox.x > mainLeftBorder && clusBbox.x + clusBbox.width < mainRightBorder	//如果这个clus在mainRow附近
			&& clusBbox.y > mainUpBorder && clusBbox.y + clusBbox.height < mainDownBorder)
		{
			if (yCenter > mainBbox.y + 0.3*mainBbox.height && yCenter < mainBbox.y + 0.7*mainBbox.height	//y方向可认为和mainRow在同一水线上
				&& xCenter < mainBbox.x && xCenter > mainBbox.x + mainBbox.width)	//X方向不在一个水平线上，此时则保留
				continue;
			if (mainRow.size() <= 7 && yCenter <= mainBbox.y)	//mainRow中只有箱号时，在mainRow上方查找公司号，并保存到up中
				up = getBetterRow(mainRow, up, rowClus[i]);
			if (yCenter >= mainBbox.y + mainBbox.height)				//在mainRow下方查找公司号
				down = getBetterRow(mainRow, down, rowClus[i]);
		}
		rowClus.erase(rowClus.begin() + i);
		i--;
	}

	//将删除后且最终过滤完成的结果重新整理一下
	for (int i = 0; i < rowClus.size(); i++)
		for (int j = 0; j < rowClus[i].size(); j++)	//把保留下来的可用于补充信息的bbox加入mainRow中
			mainRow.push_back(rowClus[i][j]);
	rowClus.clear();
	rowClus.resize(3);
	rowClus[0].assign(mainRow.begin(), mainRow.end());
	rowClus[1].assign(up.begin(), up.end());
	rowClus[2].assign(down.begin(), down.end());
}

int MserFilter::buildRowResult(void)
{
	int mainSize = rowClus[0].size() + rowClus[1].size();
	vector<cv::Rect> rowResult;
	for (int i = 0; i < rowClus.size(); i++)	//分别为mid, up, down找bbox
		rowResult.push_back(getRowClusBbox(rowClus[i]));

	//最长的箱号左边起码应该是在最左边
	int farLeft = rowResult[0].x;
	for (int i = 1; i < rowResult.size(); i++)
		if (rowResult[i].width != 0)
			farLeft = min(rowResult[i].x, farLeft);
	rowResult[0].width = rowResult[0].x - farLeft + rowResult[0].width;
	rowResult[0].x = farLeft;

	//三行的情况上面的公司号应该和mainRow左对齐，下面的箱型若是其在mainRow的左半部分，且mainRow是只有箱号的，也应该左对齐，且此时箱号和箱型也应该右对齐
	for (int i = 1; i < rowResult.size(); i++)
		if (rowResult[i].width != 0)
		{
			if (rowResult[i].y < rowResult[0].y)	//即公司号的情况
			{
				rowResult[i].width = rowResult[i].x + rowResult[i].width - rowResult[0].x;
				rowResult[i].x = rowResult[0].x;
			}
			if (rowResult[i].y > rowResult[0].y && rowResult[i].x + rowResult[i].width*0.5 < rowResult[0].x + rowResult[0].width*0.5 && rowClus[0].size() <= 8)
			{
				rowResult[i].width = rowResult[i].x + rowResult[i].width - rowResult[0].x;
				rowResult[i].x = rowResult[0].x;
				int upIdx = (i == 1) ? 2 : 1;
				int maxRight = max(rowResult[i].x + rowResult[i].width, rowResult[upIdx].x + rowResult[upIdx].width);
				rowResult[i].width = maxRight - rowResult[i].x;
				rowResult[upIdx].width = maxRight - rowResult[upIdx].x;
			}
		}
	
	// 若是三行，则箱号至少要有公司号的二倍长
	for (int i = 1; i < rowResult.size() && rowResult.size() == 3; i++)
		if (rowResult[i].y < rowResult[0].y)	//即公司号的情况
			rowResult[0].width = max(rowResult[0].width, rowResult[i].width*2);

	//左右再延长一点
	for (int i = 0; i < rowResult.size(); i++)
	{
		rowResult[i].x = max(deeplabBbox.x, rowResult[i].x - 5);
		rowResult[i].width = min(rowResult[i].width + 10, deeplabBbox.x + deeplabBbox.width - rowResult[i].x);
	}
	
	rowClus.clear();
	rowClus.resize(3);
	if(rowResult[1].width != 0)
		rowClus[0].push_back(rowResult[1]);
	rowClus[1].push_back(rowResult[0]);
	if (rowResult[2].width != 0)
		rowClus[2].push_back(rowResult[2]);
	return mainSize;
}

vector<Rect> MserFilter::getBetterRow(vector<cv::Rect> mainClus, vector<cv::Rect> oldClus, vector<cv::Rect> newClus)
{
	int charW = getClusTypicalW(mainClus);	//获取一下正常的size
	int charH = getClusTypicalH(mainClus);
	if (newClus.size() == 1)
	{
		if (newClus[0].height < 0.8*charH || newClus[0].height > 1.3*charH || newClus[0].width < 0.5*charW)
			return oldClus;
	}
	return oldClus.size() > newClus.size() ? oldClus : newClus;
}

// 就是删除一个clus中异常的值
void MserFilter::rmRowOutlier(vector<cv::Rect>& clus)
{
	int charW = getClusTypicalW(clus);	//获取一下正常的size
	int charH = getClusTypicalH(clus);
	cv::Rect bbox = getRowClusBbox(clus);
	sort(clus.begin(), clus.end(), sideSortByX);

	for (int i = 0; clus.begin() + i != clus.end(); i++)	//从左到右删除直到删除到正常值
	{
		if ((clus[0].width < 0.5*charW && abs(clus[0].height - charH)>0.2*charH)	// size比较异常
			|| (abs(clus[0].height - charH)>0.3*charH && (clus.begin() + 1 != clus.end() && clus[1].x - clus[0].x > 2.5*charW)))
			clus.erase(clus.begin());
		else
			break;
	}
	for (int i = clus.size() - 1; i >= 0; i--)	//从右到左删除，直到出现正常值
	{
		if ((clus[i].width < 0.5*charW && abs(clus[i].height - charH)>0.2*charH) //size很不正常
			|| (abs(clus[i].height - charH)>0.3*charH && (i>0 && clus[i].x - clus[i - 1].x > 5 * charW))) //size有点不正常，且离主区域有点远
			clus.erase(clus.begin() + i);
		else
			break;
	}

	//for (int i = 0; clus.begin() + i != clus.end(); i++)	//从左到右扫描，看是否有异常值
}

void MserFilter::mergeRowSmallBbox(vector<cv::Rect>& cluster)
{
	int charW = getClusTypicalW(cluster);	//获取一下正常值
	int charH = getClusTypicalH(cluster);

	sort(cluster.begin(), cluster.end(), sideSortByX);
	for (int i = 0; cluster.begin() + i != cluster.end(); i++)
	{
		if (cluster[i].width < 0.6*charW)
		{
			cv::Rect mergeLeft(0, 0, 1000, 1000), mergeRight(0, 0, 1000, 1000);
			if (i - 1 >= 0 && cluster[i].x - cluster[i - 1].x - cluster[i - 1].width < 3
				&& cluster[i].x + cluster[i].width - cluster[i - 1].x < 1.4*charW)
			{
				mergeLeft.x = cluster[i - 1].x;
				mergeLeft.y = cluster[i - 1].y;
				mergeLeft.width = cluster[i].x + cluster[i].width - cluster[i - 1].x;
				mergeLeft.height = cluster[i - 1].height;
			}
			if (cluster.begin() + i + 1 != cluster.end() && cluster[i + 1].x - cluster[i].x - cluster[i].width<3
				&& cluster[i + 1].x + cluster[i + 1].width - cluster[i].x<1.4*charW)
			{
				mergeRight.x = cluster[i].x;
				mergeRight.y = cluster[i + 1].y;
				mergeRight.width = cluster[i + 1].x + cluster[i + 1].width - cluster[i].x;
				mergeRight.height = cluster[i + 1].height;
			}

			if (mergeLeft.width < mergeRight.width)
			{
				cluster[i - 1] = mergeLeft;
				cluster.erase(cluster.begin() + i);
				i--;
			}
			else if (mergeRight.width < mergeLeft.width)
			{
				cluster[i + 1] = mergeRight;
				cluster.erase(cluster.begin() + i);
				i--;
			}
		}
	}
}

void MserFilter::findMainCol(vector<Rect>& mainCol, int &mainCharH, int &mainCharW)
{
	if (colClus.size() == 0)
		return;
	for (int i = 0; i < colClus.size(); i++)		// 合并竖类聚类中本不应该出现的并排的bbox
	{
		sort(colClus[i].begin(), colClus[i].end(), sideSortByY);
		for (int j = 1; colClus[i].begin() + j != colClus[i].end(); j++)
		{
			double yCover = colClus[i][j - 1].y + colClus[i][j - 1].height - colClus[i][j].y;
			if (yCover / colClus[i][j - 1].height > 0.8 || yCover / colClus[i][j].height > 0.8)	//竖直方向重合度大于0.8
			{
				colClus[i][j - 1] = colClus[i][j - 1] | colClus[i][j];
				colClus[i].erase(colClus[i].begin() + j);
				j--;
			}
		}
	}

	int mainColIdx = 0;	//查找最长列，作为最初步的mainCol
	sort(colClus.begin(), colClus.end(), sortByLen);	//从长到短排列
	//如果集装箱整个位置都出现在画面中，则要求mainCol必需有元素处于集装箱上半部，否则则直接选取最长的那个
	if (!(deeplabBbox.y + deeplabBbox.height>0.99*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5))	
		for (int i = 0; i < colClus.size(); i++)	//mainCol聚类是有处于集装箱上半部的元素的最长的聚类
			if (colClus[i][0].y - deeplabBbox.y < 0.4*deeplabBbox.height)
			{
				mainColIdx = i;
				break;
			}
	mainCol.assign(colClus[mainColIdx].begin(), colClus[mainColIdx].end());

	Rect mainBbox = getColClusBbox(mainCol);		//获取由此mainCol得到的参数，进行更细致的补充
	int mainXCenter = mainBbox.x + mainBbox.width*0.5;
	colReDiv(mainCol);	//利用校正后的mainCol来查找箱型
	vector<Rect> candiBbox; //箱型和防止箱号未找完全时的用于补充的数据
	for (int i = 0; colClus.begin() + i != colClus.end(); i++)
	{
		Rect rtmp = getColClusBbox(colClus[i]);
		int xCenter = rtmp.x + rtmp.width*0.5, yCenter = rtmp.y + rtmp.height*0.5;
		if (mainCol.size() < 9)
			if (xCenter >= mainBbox.x && xCenter <= mainBbox.x + mainBbox.width	//x方向有重合，且y方向无重合，保留
				&& yCenter < mainBbox.y && yCenter > mainBbox.y + mainBbox.height)
				candiBbox.insert(candiBbox.end(), colClus[i].begin(), colClus[i].end());
	}
	mainCol.insert(mainCol.end(), candiBbox.begin(), candiBbox.end());
	colReDiv(mainCol);	//对于得到的箱号等数据进行异常点去除，补充未检测点等工作
	mainCharH = getClusTypicalH(mainCol), mainCharW = getClusTypicalW(mainCol);
	rmColOutlier(mainCol, mainCharH, mainCharW, CONTAINER_ID);
	colSelfPatch(mainCol, mainCharH, mainCharW, CONTAINER_ID);
}

void MserFilter::getByMainCol(vector<Rect> mainCol, int mainCharH, int mainCharW)
{
	Rect mainBbox = getColClusBbox(mainCol);
	int mainXCenter = mainBbox.x + mainBbox.width*0.5;
	vector<Rect> containerType;		//箱型
	for (int i = 0; colClus.begin() + i != colClus.end(); i++)	//删除一些类
	{
		vector<vector<Rect>> tc;
		simpleClus(tc, colClus[i], 3*mainCharH);
		sort(tc.begin(), tc.end(), sortByLen);
		colClus[i].clear();
		colClus[i].assign(tc[0].begin(), tc[0].end());
		Rect rtmp = getColClusBbox(colClus[i]);
		int xCenter = rtmp.x + rtmp.width*0.5, yCenter = rtmp.y + rtmp.height*0.5;
		if (abs(xCenter - mainXCenter) < 10 * mainCharW && abs(xCenter - mainXCenter) > 2 * mainCharW
			&& rtmp.y+rtmp.height > mainBbox.y && rtmp.y < mainBbox.y + mainBbox.height)	// 水平方向离mainCol不远也不太近，且竖直方向有重合
			containerType = getBetterCT(mainCol, containerType, colClus[i]);
		colClus.erase(colClus.begin() + i);
		i--;
	}

	if (containerType.size() != 0)
	{
		colReDiv(containerType);
		rmColOutlier(containerType, mainCharH, mainCharW, CONTAINER_MOD);
		colSelfPatch(containerType, mainCharH, mainCharW, CONTAINER_MOD);
	}

	colClus.clear();
	colClus.resize(3);	//分别代表公司号，箱号，箱型 
	colClus[0].assign(mainCol.begin(), mainCol.end());
	colClus[2].assign(containerType.begin(), containerType.end());
}

int MserFilter::buildColResult(void)	//利用箱号加箱型间的关系补充箱型
{
	// 根据箱号和箱型间的关系，补充一下箱型数据
	colClus[1].clear();
	if (colClus[0].size() > 4)
	{
		colClus[1].assign(colClus[0].begin() + 4, colClus[0].end());
		colClus[0].resize(4);
	}
	return colClus[0].size() + colClus[1].size();
}

// curClus是当前认为合理的箱型，newClus是候选的，要拿出来和curClus进行比较的
vector<cv::Rect> MserFilter::getBetterCT(vector<cv::Rect> mainClus, vector<cv::Rect> curClus, vector<cv::Rect> newClus)
{
	int charW = getClusTypicalW(mainClus);	//获取一下正常的size
	int charH = getClusTypicalH(mainClus);

	// 自判断，看newClus是否有可能是箱型
	if (newClus[0].x < mainClus[0].x && newClus.size() < 3)	//经验来说，箱型很少在左边
		return curClus;
	if (deeplabBbox.x + deeplabBbox.width - newClus[0].x - newClus[0].width < 2*charW) //箱型不应该太贴着deeplabbbox的右边界
		return curClus;
	bool allBadFlag = true;
	for (int i = 0; i < newClus.size(); i++)
		if (newClus[i].width > 0.4*newClus[i].height)
		{
			allBadFlag = false;
			break;
		}
	if (newClus.size() > 2 && allBadFlag == true)
		return curClus;
	
	
	// 和curClus进行比较，看看谁更可能是箱型
	if (newClus.size() == 1)
	{
		if (newClus[0].height < 0.8*charH || newClus[0].height > 1.3*charH)
			return curClus;
	}
	return curClus.size() > newClus.size() ? curClus : newClus;
}

void MserFilter::colReDiv(vector<cv::Rect>& clus)	// 先合并竖直方向有重合的rect，再进行分割
{
	if (clus.size() == 0)
		return;
	int charH = getClusTypicalH(clus);
	sort(clus.begin(), clus.end(), sideSortByY);
	for (int i = 1; clus.begin() + i != clus.end(); i++)
	{
		double yCover = (clus[i - 1] & clus[i]).height;
		if (yCover > 0.15*clus[i - 1].height || yCover > 0.15*clus[i].height)	//说明可以合并，则合并
		{
			clus[i - 1] = clus[i - 1] | clus[i];
			clus.erase(clus.begin() + i);
			i--;
		}
		else	//说明不可以合并，则检查上一个bbox，看能否分拆
		{
			double ratio = clus[i-1].height*1.0 / charH;
			int segCnt = 0;
			if (abs(ratio - round(ratio)) < 0.1+round(ratio)*0.1)
				segCnt = round(ratio);

			if (segCnt >= 2)
				if ((i>1 && clus[i-1].y - clus[i-2].y - clus[i-2].height <= 6)
					|| (clus[i].y - clus[i-1].y - clus[i-1].height <= 6))	//如果是刚刚合并的，或者本来就有的
				{
					Rect r = clus[i-1];
					clus[i-1].height = r.height / segCnt;
					for (int j = 1; j < segCnt; j++)
						clus.insert(clus.begin()+i+j-1, Rect(r.x, r.y + r.height*j / segCnt, r.width, r.height / segCnt));
				}
		}
	}
}

void MserFilter::colSelfPatch(vector<cv::Rect>& tmpBboxes, int mainCharH, int mainCharW, int mode)
{
	if (tmpBboxes.size() <= 1)		//如果这个clus比较大，则进行一次更细的聚类
		return;
	vector<vector<cv::Rect>> tmpClus;
	simpleClus(tmpClus, tmpBboxes, 0.22*mainCharH);

	bool allNormal = true;			//判断结果是否需要做补充处理
	vector<int> dis;
	sort(tmpBboxes.begin(), tmpBboxes.end(), sideSortByY);
	for (int i = 0; i < tmpBboxes.size(); i++)
	{
		if (tmpBboxes[i].height > 1.4*mainCharH || tmpBboxes[i].width > 1.7*mainCharW)
		{
			allNormal = false;
			break;
		}
		if (i > 0)
			dis.push_back(tmpBboxes[i].y - tmpBboxes[i - 1].y - tmpBboxes[i - 1].height);
	}
	sort(dis.begin(), dis.end(), greater<int>());	//降序排列

	if (tmpBboxes.size() != 11 || allNormal == false || dis[2] > 0.4*mainCharH)		//还是需要进行自patch的
	{
		midPatch(tmpClus, mainCharH, mainCharW);	//进行补充
		if (mode == CONTAINER_ID)
			sidePatch(tmpClus, mainCharH, mainCharW);
		
		vector<vector<Rect>> tc(tmpClus);
		tmpClus.clear();
		int maxIdx = getMaxClusIdx(tc);
		tmpClus.push_back(tc[maxIdx]);
		if (mode == CONTAINER_ID)
		{
			if (maxIdx - 1 >= 0 && tc[maxIdx].size() != 10)
				tmpClus.push_back(tc[maxIdx-1]);
			if (maxIdx + 1 < tc.size())
				tmpClus.push_back(tc[maxIdx+1]);
		}
	}
	adjustBbox(tmpClus, mainCharH, mainCharW);			//自己再对这些结果中的cluster进行一次分割，得到更好的分割结果

	tmpBboxes.clear();								//将结果汇总
	for (int i = 0; i < tmpClus.size(); i++)
		tmpBboxes.insert(tmpBboxes.end(), tmpClus[i].begin(), tmpClus[i].end());
	for (int i = 0; tmpBboxes.begin()+i!=tmpBboxes.end(); i++)	//自己补充的可能有越界了，进行一波删除
	{
		if (tmpBboxes[i].x < deeplabBbox.x || tmpBboxes[i].x + tmpBboxes[i].width > deeplabBbox.x + deeplabBbox.width
			|| tmpBboxes[i].y < deeplabBbox.y || tmpBboxes[i].y + tmpBboxes[i].height > deeplabBbox.y + deeplabBbox.height)
		{
			tmpBboxes.erase(tmpBboxes.begin() + i);
			i--;
		}
	}
	sort(tmpBboxes.begin(), tmpBboxes.end(), sideSortByY);
}

void MserFilter::midPatch(vector<vector<cv::Rect>> &tmpClus, int mainCharH, int mainCharW)
{
	int allSize = 0;
	for (int i = 0; i < tmpClus.size(); i++)
		allSize += tmpClus[i].size();
	for (int i = 1; tmpClus.begin() + i != tmpClus.end(); i++)	// 一些断了的地方的填充，把两个clus并为一个
	{
		int cnt = 0;
		Rect leastUp = tmpClus[i - 1].back();	//上面一个聚类中最下面的一个
		int leastDis = getMinYDis(tmpClus[i][0], leastUp);
		double gapRatio = leastDis * 1.0 / mainCharH;
		if (abs(gapRatio - round(gapRatio)) < 0.2 + round(gapRatio)*0.1)	//若这个距离是一个字符高度的整数倍
			cnt = round(gapRatio);
		if (cnt > 0)	// 若两个合并后确实能得到长度为4或者6的结果
			if ((allSize > 6 && i - 1 > 0 && cnt + tmpClus[i].size() + tmpClus[i - 1].size() == 6)	//确保其不是最上面的那个类参与
				|| cnt + tmpClus[i].size() + tmpClus[i - 1].size() == 4)	//处于中间位置
			{
				int avgH = leastDis / cnt;	// 生成结果
				for (int j = 0; j < cnt; j++)
					tmpClus[i - 1].push_back(Rect(tmpClus[i][0].x, leastUp.y + leastUp.height + leastDis * j / cnt, tmpClus[i][0].width, avgH));
				tmpClus[i - 1].insert(tmpClus[i - 1].end(), tmpClus[i].begin(), tmpClus[i].end());
				tmpClus.erase(tmpClus.begin() + i);
				i--;
			}
	}
}

void MserFilter::sidePatch(vector<vector<cv::Rect>> &tmpClus, int mainCharH, int mainCharW)
{
	int mainIdx = getContainerIdIdx(tmpClus, mainCharH);	//获取箱号的Idx
	if (tmpClus[mainIdx].size() < 6)	//箱号需要补充
	{
		if (mainIdx - 1 >= 0 && tmpClus[mainIdx - 1].size() >= 4)	//利用公司号来补充
		{
			Rect compNumBottom = tmpClus[mainIdx - 1].back();	//公司号中最下面的bbox
			patchByBorder(tmpClus[mainIdx], mainCharH, compNumBottom.y + compNumBottom.height, 6);
		}
		else if (mainIdx + 1 < tmpClus.size() && tmpClus[mainIdx + 1][0].width>0.8*mainCharW && tmpClus[mainIdx + 1][0].height>0.8*mainCharH)	// 利用验证码来补充
			patchByBorder(tmpClus[mainIdx], mainCharH, tmpClus[mainIdx + 1][0].y, 6);
	}
	if (tmpClus[mainIdx].size() == 6)	//经过补充过后，箱号长度为6，以此为基准补充公司号，验证码
	{
		if (mainIdx - 1 >= 0 && tmpClus[mainIdx - 1].size() < 4)	// 公司号不齐全
		{
			patchByBorder(tmpClus[mainIdx - 1], mainCharH, tmpClus[mainIdx][0].y, 4);
		}
		else if (mainIdx == 0 /* || maxIdx - 1肯定不是*/)   //没有公司号
		{
			int dis = 0.35*mainCharH;
			if (mainIdx + 1 < tmpClus.size() && tmpClus[mainIdx + 1][0].width>0.8*mainCharW && tmpClus[mainIdx + 1][0].height>0.8*mainCharH)	//若有验证码则利用验证码来补充
				dis = tmpClus[mainIdx + 1][0].y - tmpClus[mainIdx].back().y - tmpClus[mainIdx].back().height;
			vector<cv::Rect> compNum;
			for (int i = 1; i <= 4; i++)
				compNum.insert(compNum.begin(), Rect(tmpClus[mainIdx][0].x, tmpClus[mainIdx][0].y - dis - mainCharH*i, tmpClus[mainIdx][0].width, mainCharH));
			tmpClus.insert(tmpClus.begin() + mainIdx, compNum);
			mainIdx++;
		}
		if (mainIdx + 1 == tmpClus.size() || tmpClus[mainIdx + 1][0].y - tmpClus[mainIdx].back().y > 2 * mainCharH
			|| tmpClus[mainIdx + 1][0].width<0.6*mainCharW || tmpClus[mainIdx + 1][0].height<0.6*mainCharH)	// 没有合适的验证码
		{
			int dis = 0.35*mainCharH;
			if (mainIdx - 1 >= 0 && tmpClus[mainIdx - 1].size() == 4)//上方有正确的公司号，则依据此公司号计算距离
				dis = tmpClus[mainIdx][0].y - tmpClus[mainIdx - 1].back().y - tmpClus[mainIdx - 1].back().height;
			Rect containNumBottom = tmpClus[mainIdx].back();
			vector<cv::Rect> vertiNum{ Rect(containNumBottom.x, containNumBottom.y+ containNumBottom.height + dis, containNumBottom.width, 1.2*mainCharH) };
			tmpClus.insert(tmpClus.begin() + mainIdx + 1, vertiNum);
		}
	}
}

void MserFilter::adjustBbox(vector<vector<cv::Rect>> &clus, int mainCharH, int mainCharW)
{
	for (int i = 0; i < clus.size(); i++)
	{
		if (clus[i].size() < 2)
		{
			if (clus[i][0].width < 0.9*mainCharW)
			{
				clus[i][0].x = clus[i][0].x + clus[i][0].width*0.5 - 0.6*mainCharW;
				clus[i][0].width = 1.3*mainCharW;
			}
			if (clus[i][0].height < 0.85*mainCharH)
			{
				clus[i][0].y = clus[i][0].y + clus[i][0].height*0.5 - 0.55*mainCharH;
				clus[i][0].height = 1.1*mainCharH;
			}
			continue;
		}
		else
		{
			sort(clus[i].begin(), clus[i].end(), sideSortByY);	//计算y方向信息
			int upBorder = clus[i][0].y, botBorder = clus[i].back().y + clus[i].back().height;
			if (clus[i][0].height < 0.85*mainCharH && clus[i][1].y - 1.1*mainCharH < clus[i][0].y)
				upBorder = clus[i][1].y - 1.1*mainCharH;
			if (clus[i].back().height < 0.85*mainCharH && clus[i][clus.size() - 2].y + clus[i][clus.size() - 2].height + 1.1*mainCharH > clus[i].back().height + clus[i].back().y)
				botBorder = clus[i][clus.size() - 2].y + clus[i][clus.size() - 2].height + 1.1*mainCharH;
			double avgHeight = (botBorder - upBorder)*1.0 / clus[i].size();

			int leftBorder = 10000, rightBorder = 0;	//计算x方向信息，此处leftBorder指的是矩形中线坐标
			double topHalf = 0, bottomHalf = 0;
			for (int j = 0; j < clus[i].size(); j++)
			{
				if (j + 1 < (clus[i].size()+1) / 2.0)
					topHalf += clus[i][j].x + clus[i][j].width*0.5;
				else if(j + 1 > (clus[i].size()+1) / 2.0)
					bottomHalf += clus[i][j].x + clus[i][j].width*0.5;
				if (clus[i][j].x + clus[i][j].width*0.5 < leftBorder)
					leftBorder = clus[i][j].x + clus[i][j].width*0.5;
				if (clus[i][j].x + clus[i][j].width*0.5 > rightBorder)
					rightBorder = clus[i][j].x + clus[i][j].width*0.5;
			}
			double xOffset = (rightBorder - leftBorder) * 1.0 / (clus[i].size()-1);

			for (int j = 0; j < clus[i].size(); j++)
			{
				clus[i][j].y = upBorder + round(avgHeight * j);
				clus[i][j].height = round(avgHeight);
				clus[i][j].width = mainCharW * 1.4 > clus[i][j].width ? mainCharW * 1.4 : clus[i][j].width;
				if (topHalf > bottomHalf)
					clus[i][j].x = leftBorder + round(xOffset*(clus[i].size() - j - 1) - 0.7*mainCharW);
				else
					clus[i][j].x = leftBorder + round(xOffset*j - 0.7*mainCharW);
			}
		}
	}
}

void MserFilter::patchByBorder(vector<cv::Rect>& bboxes, int mainCharH, int border, int targetLen)
{
    if(bboxes[0].y > border)  //  利用上边界来patch
	{
        while (bboxes[0].y - border > mainCharH && bboxes.size() < targetLen)	//不停的填充，直到两者距离很近
            bboxes.insert(bboxes.begin(), Rect(bboxes[0].x, bboxes[0].y-mainCharH, bboxes[0].width, mainCharH));
        while (bboxes.size() < targetLen)	// 上方到顶后，再继续往下方加
            bboxes.push_back(Rect(bboxes.back().x, bboxes.back().y + mainCharH, bboxes.back().width, mainCharH));
    }
    else if(bboxes.back().y + bboxes.back().height < border)    //利用下边界来patch
    {
		while (border - bboxes.back().y - bboxes.back().height > mainCharH && bboxes.size() < targetLen)	//不停的填充，直到两者距离很近
			bboxes.push_back(Rect(bboxes.back().x, bboxes.back().y + mainCharH, bboxes.back().width, mainCharH));
		while (bboxes.size() < targetLen)   // 下方到底后，再继续往上方加
			bboxes.insert(bboxes.begin(), Rect(bboxes[0].x, bboxes[0].y - mainCharH, bboxes[0].width, mainCharH));
	}
}

void MserFilter::simpleClus(vector<vector<cv::Rect>> &tmpClus, vector<cv::Rect> bboxes, double disThre)
{
	//先进行聚类，把箱号，公司号，校验码分别拿出来
	tmpClus.resize(1);
	tmpClus[0].clear();
	tmpClus[0].push_back(bboxes[0]);
	sort(bboxes.begin(), bboxes.end(), sideSortByY);
	for (int i = 1, clusIdx = 0; i < bboxes.size(); i++)
	{
		if (getMinYDis(bboxes[i], bboxes[i - 1]) > disThre)
		{
			clusIdx++;
			vector<Rect> t;
			tmpClus.push_back(t);
		}
		tmpClus[clusIdx].push_back(bboxes[i]);
	}
}

void MserFilter::rmColOutlier(vector<cv::Rect>& clus, int mainCharH, int mainCharW, int mode)
{
	double areaThre = (mode == CONTAINER_MOD) ? 1.7 : 2.5;
	for (int i = 0; i < clus.size(); i++)
		if (clus[i].width > clus[i].height || clus[i].area() > areaThre*mainCharH*mainCharW)	//一些非常明显的outlier，直接删除
		{
			clus.erase(clus.begin() + i);
			i--;
		}
		
	if (mode == CONTAINER_ID && clus.size() > 11)
	{
		sort(clus.begin(), clus.end(), sideSortByY);
		while(clus.size() > 11)
		{
			if (clus[1].y - clus[0].y - clus[0].height > 0.5*mainCharH)
				clus.erase(clus.begin());
			else
				break;
		}
	}
}

int MserFilter::getContainerIdIdx(vector<vector<cv::Rect>>& tmpClus, int mainCharH)
{
	sort(tmpClus.begin(), tmpClus.end(), clusSortByY);
	int maxIdx = 0, maxLen = 0;
	for (int i = 0; i<tmpClus.size(); i++)	//最长中最下面的那个
		if (tmpClus[i].size() >= maxLen)
		{
			maxLen = tmpClus[i].size();
			maxIdx = i;
		}

	if (maxLen > 4)
		return maxIdx;

	// 根据deeplabBbox的位置决定containerIdIdx的值
	if (deeplabBbox.y + deeplabBbox.height > 0.99*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5)	//集装箱只有上半部分在图片中，这种情况通常会有箱号漏掉，此时直接取最下面一个类
		return tmpClus.size() - 1;
	else	//集装箱全部在图片中，
	{
		if(tmpClus.back().size() > 1)	//若是最后一个的尺寸>1，则直接以最后一个作为目标项
			return tmpClus.size() - 1;

		maxIdx = tmpClus.size() - 1, maxLen = 0;
		for (int i = tmpClus.size() - 1; i >= 0; i--)	//containerId应该有Rect离最下面一个类不太远(<6*mainCharH)
		{
			sort(tmpClus[i].begin(), tmpClus[i].end(), sideSortByY);
			if (tmpClus.back()[0].y - tmpClus[i].back().y - tmpClus[i].back().height > 6 * mainCharH)
				break;
			if (tmpClus[i].size() >= maxLen)
			{
				maxLen = tmpClus[i].size();
				maxIdx = i;
			}
		}
		return maxIdx;
	}
}

MserFilter::~MserFilter()
{

}

// 判断两者是否可以被聚为一类
bool MserFilter::isClose(Rect r1, Rect r2, int disThres, int mode)
{
	int r1CenterX = r1.x + r1.width / 2, r1CenterY = r1.y + r1.height / 2;
	int r2CenterX = r2.x + r2.width / 2, r2CenterY = r2.y + r2.height / 2;
	if (mode == COL_MODE || mode == COL_LOOSE_MODE)		//需要把一列聚为一类
	{
		double xCoverRatio = 0.65;
		if (mode == COL_LOOSE_MODE) //宽松列聚类，放宽了条件
			xCoverRatio = 0.2;
		double xCover = min(abs(r1.x - r2.x - r2.width), abs(r2.x - r1.x - r1.width));//两个rect之间x方向重合的长度
		if(max(abs(r1.x - r2.x - r2.width), abs(r2.x - r1.x - r1.width)) < r1.width + r2.width)
			if (xCover / r1.width > xCoverRatio || xCover / r2.width > xCoverRatio) // 两个Rect是否在一列上
				if (abs(r1CenterY - r2CenterY) < disThres)		//两者distance满足条件
					return true;
	}
	if (mode == ROW_MODE || mode == ROW_LOOSE_MODE)		// 把一行聚为一类
	{
		double yCoverRatio = 0.8;
		if (mode == ROW_LOOSE_MODE) //宽松行聚类，放宽了条件
			yCoverRatio = 0.6;
		double yCover = min(abs(r1.y - r2.y - r2.height), abs(r2.y - r1.y - r1.height));//两个rect之间y方向重合的长度
		if (max(abs(r1.y - r2.y - r2.height), abs(r2.y - r1.y - r1.height)) < r1.height + r2.height)
			if (yCover / r1.height > yCoverRatio || yCover / r2.height > yCoverRatio) // 两个Rect是否在一行上，此处应该更严格一点
				if (abs(r1CenterX - r2CenterX) < disThres) //两者distance
					return true;
	}	
	return false;
}

int MserFilter::judgeSide(void)
{
	return filter();
}
