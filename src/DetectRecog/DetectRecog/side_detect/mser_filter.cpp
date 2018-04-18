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
				color = 250;
				rectangle(mserImg, clus[i][j], color, 1, 8, 0);
			}
		}
	}
	rectangle(mserImg, Rect(0, 0, int(mserWidthHuge), int(mserHeightSmall)), 255, 1, 8, 0);
	rectangle(mserImg, Rect(0, 0, int(mserWidthHuge), int(mserHeightHuge)), 255, 1, 8, 0);
	rectangle(mserImg, deeplabBbox, 255, 1, 8, 0);
	imwrite(outputPath, mserImg);
}

void MserFilter::drawResult(string outputPath, vector<Rect> result)
{
	Mat resImg = srcImg.clone();
	for (int i = 0; i < result.size(); i++)
		rectangle(resImg, result[i], Scalar(0, 255, 0), 1, 8, 0);
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
	if(mode == ROW_MODE)	//找到最长的类
		findMainRow(mainClus);
	else
		findMainCol(mainClus);

	if (mainClus.size() <=4)
		return 0;

	if (mode == ROW_MODE)	//根据最长的类做进一步的filter和patch
	{
		getByMainRow(mainClus);
		return buildRowResult();
	}
	else
	{
		getByMainCol(mainClus);
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

void MserFilter::findMainCol(vector<Rect>& mainCol)
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

	int mainColIdx = 0;	//查找最长列
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
}

void MserFilter::getByMainCol(vector<Rect> mainCol)
{
	Rect mainBbox = getColClusBbox(mainCol);
	int mainCharH = getClusTypicalH(mainCol), mainCharW = getClusTypicalW(mainCol);
	int mainXCenter = mainBbox.x + mainBbox.width*0.5;

	colReDiv(mainCol);	//利用校正后的mainCol来查找箱型
	vector<Rect> containerType;
	for (int i = 0; colClus.begin()+i != colClus.end(); i++)	//删除一些类
	{
		Rect rtmp = getColClusBbox(colClus[i]);
		int xCenter = rtmp.x + rtmp.width*0.5, yCenter = rtmp.y + rtmp.height*0.5;
		if (mainCol.size() < 9)
			if (xCenter >= mainBbox.x && xCenter <= mainBbox.x + mainBbox.width	//x方向有重合，且y方向无重合，保留
				&& yCenter < mainBbox.y && yCenter > mainBbox.y + mainBbox.height) 
					continue;
		if (abs(xCenter - mainXCenter) < 8 * mainCharW && abs(xCenter - mainXCenter) > 2 * mainCharW
			&& yCenter >= mainBbox.y && yCenter <= mainBbox.y + mainBbox.height)	// 水平方向离mainCol不远也不太近，且竖直方向有重合
			containerType = getBetterCol(mainCol, containerType, colClus[i]);
		colClus.erase(colClus.begin() + i);
		i--;
	}

	vector<Rect> keptBbox;	//被保留的Bbox
	for (int i = 0; i < colClus.size(); i++)
		keptBbox.insert(keptBbox.end(), colClus[i].begin(), colClus[i].end());
	colClus.clear();
	colClus.resize(3);	//分别代表公司号，箱号，箱型 
	colClus[0].assign(mainCol.begin(), mainCol.end());
	colClus[1].insert(colClus[0].end(), keptBbox.begin(), keptBbox.end());
	colClus[2].assign(containerType.begin(), containerType.end());
}

int MserFilter::buildColResult(void)
{
	drawClus(savePath + "_00_t0.jpg", colClus);	
	colReDiv(colClus[0]);	//对于得到的箱号等数据进行异常点去除，补充未检测点等工作
	int mainCharH = getClusTypicalH(colClus[0]), mainCharW = getClusTypicalW(colClus[0]);
	rmColOutlier(colClus[0], mainCharH, mainCharW);
	colSelfPatch(colClus[0], mainCharH, mainCharW);
	if (colClus[2].size() != 0)
	{
		colReDiv(colClus[2]);
		rmColOutlier(colClus[2], mainCharH, mainCharW);
		colSelfPatch(colClus[2], mainCharH, mainCharW);
	}

	// 根据箱号和箱型间的关系，补充一下箱型数据

	colClus[1].clear();
	if (colClus[0].size() > 4)
	{
		colClus[1].assign(colClus[0].begin() + 4, colClus[0].end());
		colClus[0].resize(4);
	}
	return colClus[0].size() + colClus[1].size();
}

vector<cv::Rect> MserFilter::getBetterCol(vector<cv::Rect> mainClus, vector<cv::Rect> oldClus, vector<cv::Rect> newClus)
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

void MserFilter::colSelfPatch(vector<cv::Rect>& tmpBboxes, int mainCharH, int mainCharW)
{
	if (tmpBboxes.size() <= 1)
		return;
	vector<vector<cv::Rect>> tmpClus;
	simpleClus(tmpClus, tmpBboxes, 0.22*mainCharH);
	drawClus(savePath + "_00_t1.jpg", tmpClus);

	for (int i = 1; tmpClus.begin() + i != tmpClus.end(); i++)	// 一些断了的地方的填充
	{
		int cnt = 0;
		Rect leastUp = tmpClus[i - 1].back();	//上面一个聚类中最下面的一个
		int leastDis = getMinYDis(tmpClus[i][0], leastUp);
		double gapRatio = leastDis * 1.0 / mainCharH;
		//cout << gapRatio << endl;
		if (abs(gapRatio - round(gapRatio)) < 0.2+ round(gapRatio)*0.1)	//若这个距离是一个字符高度的整数倍
			cnt = round(gapRatio);
		if (cnt > 0)	// 若两个合并后确实能得到长度为4或者6的结果
			if((tmpBboxes.size() > 6 && i-1!=0 && cnt + tmpClus[i].size() + tmpClus[i - 1].size() == 6)	//确保其不是最上面的那个类参与
				|| cnt + tmpClus[i].size() + tmpClus[i - 1].size() == 4)
			{
				int avgH = leastDis / cnt;	// 生成结果
				for (int j = 0; j < cnt; j++)
					tmpClus[i - 1].push_back(Rect(tmpClus[i][0].x, leastUp.y + leastUp.height + leastDis * j / cnt, tmpClus[i][0].width, avgH));
				tmpClus[i - 1].insert(tmpClus[i - 1].end(), tmpClus[i].begin(), tmpClus[i].end());
				tmpClus.erase(tmpClus.begin() + i);
				i--;
			}
	}
	drawClus(savePath + "_00_t2.jpg", tmpClus);
	
	int maxIdx = getMaxClusIdx(tmpClus);
	if (tmpBboxes.size() > 7)	//箱号那一列
	{
		if (tmpClus[maxIdx].size() < 6)	//箱号需要补充
		{
			if (maxIdx - 1 >= 0 && tmpClus[maxIdx - 1].size() >= 4)	//利用公司号来补充
			{
				Rect compNumBottom = tmpClus[maxIdx - 1].back();	//公司号中最下面的bbox
				patchByBorder(tmpClus[maxIdx], mainCharH, compNumBottom.y+compNumBottom.height, 6);
			}
			else if (maxIdx + 1 < tmpClus.size() && tmpClus[maxIdx + 1][0].width>0.8*mainCharW && tmpClus[maxIdx + 1][0].height>0.8*mainCharH)	// 利用验证码来补充
                patchByBorder(tmpClus[maxIdx], mainCharH, tmpClus[maxIdx+1][0].y, 6);
		}
		if(tmpClus[maxIdx].size() == 6)	//经过补充过后，箱号长度为6，以此为基准补充公司号，验证码
		{
			if (maxIdx - 1 >= 0 && tmpClus[maxIdx - 1].size() < 4)	// 公司号不齐全
			{
				patchByBorder(tmpClus[maxIdx - 1], mainCharH, tmpClus[maxIdx][0].y, 4);
				drawClus(savePath + "_00_t3.jpg", tmpClus);
			}
            else if(maxIdx == 0 /* || maxIdx - 1肯定不是*/)   //没有公司号
            {
                int dis = 0.35*mainCharH;
                if(maxIdx + 1 < tmpClus.size() && tmpClus[maxIdx + 1][0].width>0.8*mainCharW && tmpClus[maxIdx + 1][0].height>0.8*mainCharH)	//若有验证码则利用验证码来补充
                    dis = tmpClus[maxIdx+1][0].y - tmpClus[maxIdx].back().y - tmpClus[maxIdx].back().height;
                vector<cv::Rect> compNum;
				for (int i = 1; i <= 4; i++)
					compNum.insert(compNum.begin(), Rect(tmpClus[maxIdx][0].x, tmpClus[maxIdx][0].y - dis - mainCharH*i, tmpClus[maxIdx][0].width, mainCharH));
                tmpClus.insert(tmpClus.begin()+maxIdx, compNum);
                maxIdx ++;
            }
			if (maxIdx + 1 == tmpClus.size() || tmpClus[maxIdx + 1][0].y - tmpClus[maxIdx].back().y > 2 * mainCharH
				|| tmpClus[maxIdx + 1][0].width<0.6*mainCharW || tmpClus[maxIdx + 1][0].height<0.6*mainCharH)	// 没有合适的验证码
			{
				drawClus(savePath + "_00_t4.jpg", tmpClus);
                int dis = 0.35*mainCharH;
                if(maxIdx - 1>=0 && tmpClus[maxIdx - 1].size() == 4)//上方有正确的公司号，则依据此公司号计算距离
                    dis = tmpClus[maxIdx][0].y - tmpClus[maxIdx-1].back().y - tmpClus[maxIdx-1].back().height;
                Rect containNumBottom = tmpClus[maxIdx].back();
                vector<cv::Rect> vertiNum{Rect(containNumBottom.x, containNumBottom.y + dis, containNumBottom.width, 1.2*mainCharH)};
                tmpClus.insert(tmpClus.begin()+maxIdx+1, vertiNum);
			}
		}
	}

	int allSize = tmpBboxes.size();	//汇合一下结果
	tmpBboxes.clear();
	tmpBboxes.insert(tmpBboxes.end(), tmpClus[maxIdx].begin(), tmpClus[maxIdx].end());
	if (allSize > 7)
	{
		if(maxIdx - 1 >= 0)
			tmpBboxes.insert(tmpBboxes.end(), tmpClus[maxIdx-1].begin(), tmpClus[maxIdx-1].end());
		if(maxIdx + 1 < tmpClus.size())
		tmpBboxes.insert(tmpBboxes.end(), tmpClus[maxIdx+1].begin(), tmpClus[maxIdx+1].end());
	}

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

void MserFilter::patchByBorder(vector<cv::Rect>& bboxes, int mainCharH, int border, int targetLen)
{
    if(bboxes[0].y > border)  //  利用上边界来patch
	{
        while (bboxes[0].y - border > mainCharH)	//不停的填充，直到两者距离很近
            bboxes.insert(bboxes.begin(), Rect(bboxes[0].x, bboxes[0].y-mainCharH, bboxes[0].width, mainCharH));
        while (bboxes.size() < targetLen)	// 上方到顶后，再继续往下方加
            bboxes.push_back(Rect(bboxes.back().x, bboxes.back().y + mainCharH, bboxes.back().width, mainCharH));
    }
    else if(bboxes.back().y + bboxes.back().height < border)    //利用下边界来patch
    {
		while (border - bboxes.back().y - bboxes.back().height > mainCharH)	//不停的填充，直到两者距离很近
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

void MserFilter::rmColOutlier(vector<cv::Rect>& clus, int mainCharH, int mainCharW)
{
	for (int i = 0; i < clus.size(); i++)
		if (clus[i].width > clus[i].height || clus[i].area() > 2.5*mainCharH*mainCharW)	//一些非常明显的outlier，直接删除
		{
			clus.erase(clus.begin() + i);
			i--;
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
