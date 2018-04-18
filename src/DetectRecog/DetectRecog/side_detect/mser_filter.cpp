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

	//�趨��ֵ
	double heightRatio = (double(deeplabBbox.height)) / srcImg.rows;
	//������һ����ͷ�����λ����ʱ����߽���һ��У��
	if (deeplabBbox.y+deeplabBbox.height > 0.99*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5)
		heightRatio += 0.2;
	//����ͷλ�ù��͵�У��
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

// ��ȫ��ͼ�ϻ�bbox
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

// ��һ��ȫ��ͼ�л��������
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
	for (int i = 0; bboxes.begin() + i != bboxes.end(); i++)	//����deeplab bbox��ɾ��
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
		// size���Բ������ֵĲ�Ҫ
		if (bboxes[i].height < bboxes[i].width || bboxes[i].height <= mserHeightSmall
			|| bboxes[i].height >= mserHeightHuge || bboxes[i].width >= mserWidthHuge
			|| bboxes[i].height > 10 * bboxes[i].width)
		{
			bboxes.erase(bboxes.begin() + i);
			i--;
		}
	}
	// ɾ���ص�����
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
	// ��bbox������������࣬����Ľ��ΪMserFilter��Ա
	cluster(mserWidthHuge*2.5, bboxes, rowClus, ROW_MODE);
	cluster(mserHeightHuge*4, bboxes, colClus, COL_MODE);
	
	int mainRowSize = clusProcess(ROW_MODE);
	int mainColsize = clusProcess(COL_MODE);

	if (mainRowSize==0 && mainColsize==0)	//��Ϊ0
	{
		rowClus.clear();
		colClus.clear();
	}
	else if(mainRowSize==0 && mainColsize != 0)	//��һ��Ϊ0
		rowClus.clear();
	else if (mainColsize == 0 && mainRowSize != 0)
		colClus.clear();
	else	//����Ϊ0
	{
		if((mainColsize >= 6 && mainRowSize <= 4) || mainColsize >= 11 || mainColsize / mainRowSize >= 2)	// col����õ��Ľ�����ã�����col��
			rowClus.clear();
		else if ((mainRowSize >= 6 && mainColsize <= 4) || mainRowSize >= 11 || mainRowSize / mainColsize >= 2)
			colClus.clear();
		else	// hard���ж�˭�����ұ�һ��
		{
			if (rowClus[0][0].x + 0.5*rowClus[0][0].width >= colClus[0][0].x)
				colClus.clear();
			else
				rowClus.clear();
		}
	}
}

// �о�����о��࣬ȡ����mode��mode�ᴫ��isClose()�����ж��Ƿ�����һ��(�з�������з���)
void MserFilter::cluster(int disThres, vector<Rect> bboxes, vector<vector<Rect>>& clus,int mode)
{
	// idx�д����ÿ��bbox�������࣬��ʼʱÿ��bbox�������Լ�һ��
	vector<int> idx(bboxes.size());
	for (int i = 0; i < idx.size(); i++)
		idx[i] = i;

	// ��ʼ���࣬��forѭ�������һ����bboxes�ȳ������飬������Ԫ�ر�ʾ��Ӧ��bbox��������
	for (int i = 0; i < bboxes.size(); i++)
		for (int j = i + 1; j < bboxes.size(); j++)
			if (isClose(bboxes[i], bboxes[j], disThres, mode) == true)
			{
				if (idx[j] == idx[i])	//����ͬһ����
					continue;
				else if (idx[j] == j)	//��������
					idx[j] = idx[i];
				else                    //close��������һ���࣬�ϲ���������
				{
					int srcClas = idx[j]>idx[i] ? idx[j] : idx[i];
					int desClas = idx[i] == srcClas ? idx[j] : idx[i];
					for (int k = 0; k < idx.size(); k++)
						if (idx[k] == srcClas)
							idx[k] = desClas;
				}
			}

	// ����������˳��������棬һ�����Ӧһ��˳���������֮��Ĳ���
	// �˴�����ʹ�ö��forѭ��
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

//���������rowClus��size
int MserFilter::clusProcess(int mode)
{
	vector<Rect> mainClus;
	if(mode == ROW_MODE)	//�ҵ������
		findMainRow(mainClus);
	else
		findMainCol(mainClus);

	if (mainClus.size() <=4)
		return 0;

	if (mode == ROW_MODE)	//�������������һ����filter��patch
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
	//��ɾ���ȽϿ��µ���,�������
	vector<vector<Rect>> clusTmp(rowClus);
	for (int i = 0; clusTmp.begin() + i != clusTmp.end(); i++)
		//���������λ�ù��ߵ���ͼƬ��һ���ھ�ͷ����ʱ���������ִ���
		if (!(deeplabBbox.y + deeplabBbox.height > 0.98*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5))
			if ((clusTmp[i][0].y - deeplabBbox.y)*1.0 / deeplabBbox.height > 0.35)
			{
				clusTmp.erase(clusTmp.begin() + i);
				i--;
			}
	
	if (clusTmp.size() == 0)
		return;

	// ������Сbbox���кϲ����쳣��bbox����ɾ��
	for (int i = 0; i < clusTmp.size(); i++)
	{
		mergeRowSmallBbox(clusTmp[i]);
		rmRowOutlier(clusTmp[i]);
	}

	// �ӳ���������Ĭ������Ǹ���ΪmainRow
	int mainRowIdx = 0;
	sort(clusTmp.begin(), clusTmp.end(), sortByLen);
	if (clusTmp.size() > 1)
		if (clusTmp[0].size() >= 6 && clusTmp[1].size() >= 6)	//�����������ܳ������п�����mainRow
		{
			Rect clusBbox0 = getRowClusBbox(clusTmp[0]), clusBbox1 = getRowClusBbox(clusTmp[1]);
			if (abs(clusBbox0.x+0.5*clusBbox0.width-clusBbox1.x-clusBbox1.width*0.5) > max(clusBbox0.width, clusBbox1.width))	//���������X��������Զ��ѡ�񷽲�С���Ǹ�
			{
				if (heightSD(clusTmp[1]) < heightSD(clusTmp[0]))
					mainRowIdx = 1;
			}
			else // ����ѡ��clusBbox������Ǹ�
			{
				if (clusBbox1.width > clusBbox0.width)
					mainRowIdx = 1;
			}
		}
	mainRow.assign(clusTmp[mainRowIdx].begin(), clusTmp[mainRowIdx].end());
}

void MserFilter::getByMainRow(vector<Rect> mainRow)
{		
	//����mainRowClus����ز���
	Rect mainBbox = getRowClusBbox(mainRow);
	int mainCharH = getClusTypicalH(mainRow);	//ȥ��һЩ�쳣ֵ�����ֵ
	int mainCharW = getClusTypicalW(mainRow);
	int mainLeftBorder = mainBbox.x - 8 * mainCharW, mainRightBorder = mainBbox.x + mainBbox.width + 8 * mainCharW;
	int mainUpBorder = mainBbox.y - mainCharH * 2, mainDownBorder = mainBbox.y + mainBbox.height + mainCharH * 2.5;

	mainRegion.x = mainLeftBorder; mainRegion.width = mainRightBorder - mainLeftBorder;
	mainRegion.y = mainUpBorder; mainRegion.height = mainDownBorder - mainUpBorder;

	vector<Rect> up, down;	//�ֱ��ʾ�Ϸ��Ĺ�˾�ź��·�������
	for (int i = 0; rowClus.begin() + i != rowClus.end(); i++)
	{
		mergeRowSmallBbox(rowClus[i]);	//�ȶ����clus���м򵥵Ĵ���
		rmRowOutlier(rowClus[i]);
		Rect clusBbox = getRowClusBbox(rowClus[i]);	//��ȡһ��clus����ز���
		int yCenter = clusBbox.y + clusBbox.height * 0.5, xCenter = clusBbox.x + clusBbox.width*0.5;

		if (clusBbox.x > mainLeftBorder && clusBbox.x + clusBbox.width < mainRightBorder	//������clus��mainRow����
			&& clusBbox.y > mainUpBorder && clusBbox.y + clusBbox.height < mainDownBorder)
		{
			if (yCenter > mainBbox.y + 0.3*mainBbox.height && yCenter < mainBbox.y + 0.7*mainBbox.height	//y�������Ϊ��mainRow��ͬһˮ����
				&& xCenter < mainBbox.x && xCenter > mainBbox.x + mainBbox.width)	//X������һ��ˮƽ���ϣ���ʱ����
				continue;
			if (mainRow.size() <= 7 && yCenter <= mainBbox.y)	//mainRow��ֻ�����ʱ����mainRow�Ϸ����ҹ�˾�ţ������浽up��
				up = getBetterRow(mainRow, up, rowClus[i]);
			if (yCenter >= mainBbox.y + mainBbox.height)				//��mainRow�·����ҹ�˾��
				down = getBetterRow(mainRow, down, rowClus[i]);
		}
		rowClus.erase(rowClus.begin() + i);
		i--;
	}

	//��ɾ���������չ�����ɵĽ����������һ��
	for (int i = 0; i < rowClus.size(); i++)
		for (int j = 0; j < rowClus[i].size(); j++)	//�ѱ��������Ŀ����ڲ�����Ϣ��bbox����mainRow��
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
	for (int i = 0; i < rowClus.size(); i++)	//�ֱ�Ϊmid, up, down��bbox
		rowResult.push_back(getRowClusBbox(rowClus[i]));

	//�������������Ӧ�����������
	int farLeft = rowResult[0].x;
	for (int i = 1; i < rowResult.size(); i++)
		if (rowResult[i].width != 0)
			farLeft = min(rowResult[i].x, farLeft);
	rowResult[0].width = rowResult[0].x - farLeft + rowResult[0].width;
	rowResult[0].x = farLeft;

	//���е��������Ĺ�˾��Ӧ�ú�mainRow����룬�����������������mainRow����벿�֣���mainRow��ֻ����ŵģ�ҲӦ������룬�Ҵ�ʱ��ź�����ҲӦ���Ҷ���
	for (int i = 1; i < rowResult.size(); i++)
		if (rowResult[i].width != 0)
		{
			if (rowResult[i].y < rowResult[0].y)	//����˾�ŵ����
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
	
	// �������У����������Ҫ�й�˾�ŵĶ�����
	for (int i = 1; i < rowResult.size() && rowResult.size() == 3; i++)
		if (rowResult[i].y < rowResult[0].y)	//����˾�ŵ����
			rowResult[0].width = max(rowResult[0].width, rowResult[i].width*2);

	//�������ӳ�һ��
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
	int charW = getClusTypicalW(mainClus);	//��ȡһ��������size
	int charH = getClusTypicalH(mainClus);
	if (newClus.size() == 1)
	{
		if (newClus[0].height < 0.8*charH || newClus[0].height > 1.3*charH || newClus[0].width < 0.5*charW)
			return oldClus;
	}
	return oldClus.size() > newClus.size() ? oldClus : newClus;
}

// ����ɾ��һ��clus���쳣��ֵ
void MserFilter::rmRowOutlier(vector<cv::Rect>& clus)
{
	int charW = getClusTypicalW(clus);	//��ȡһ��������size
	int charH = getClusTypicalH(clus);
	cv::Rect bbox = getRowClusBbox(clus);
	sort(clus.begin(), clus.end(), sideSortByX);

	for (int i = 0; clus.begin() + i != clus.end(); i++)	//������ɾ��ֱ��ɾ��������ֵ
	{
		if ((clus[0].width < 0.5*charW && abs(clus[0].height - charH)>0.2*charH)	// size�Ƚ��쳣
			|| (abs(clus[0].height - charH)>0.3*charH && (clus.begin() + 1 != clus.end() && clus[1].x - clus[0].x > 2.5*charW)))
			clus.erase(clus.begin());
		else
			break;
	}
	for (int i = clus.size() - 1; i >= 0; i--)	//���ҵ���ɾ����ֱ����������ֵ
	{
		if ((clus[i].width < 0.5*charW && abs(clus[i].height - charH)>0.2*charH) //size�ܲ�����
			|| (abs(clus[i].height - charH)>0.3*charH && (i>0 && clus[i].x - clus[i - 1].x > 5 * charW))) //size�е㲻�����������������е�Զ
			clus.erase(clus.begin() + i);
		else
			break;
	}

	//for (int i = 0; clus.begin() + i != clus.end(); i++)	//������ɨ�裬���Ƿ����쳣ֵ
}

void MserFilter::mergeRowSmallBbox(vector<cv::Rect>& cluster)
{
	int charW = getClusTypicalW(cluster);	//��ȡһ������ֵ
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
	for (int i = 0; i < colClus.size(); i++)		// �ϲ���������б���Ӧ�ó��ֵĲ��ŵ�bbox
	{
		sort(colClus[i].begin(), colClus[i].end(), sideSortByY);
		for (int j = 1; colClus[i].begin() + j != colClus[i].end(); j++)
		{
			double yCover = colClus[i][j - 1].y + colClus[i][j - 1].height - colClus[i][j].y;
			if (yCover / colClus[i][j - 1].height > 0.8 || yCover / colClus[i][j].height > 0.8)	//��ֱ�����غ϶ȴ���0.8
			{
				colClus[i][j - 1] = colClus[i][j - 1] | colClus[i][j];
				colClus[i].erase(colClus[i].begin() + j);
				j--;
			}
		}
	}

	int mainColIdx = 0;	//�������
	sort(colClus.begin(), colClus.end(), sortByLen);	//�ӳ���������
	//�����װ������λ�ö������ڻ����У���Ҫ��mainCol������Ԫ�ش��ڼ�װ���ϰ벿��������ֱ��ѡȡ����Ǹ�
	if (!(deeplabBbox.y + deeplabBbox.height>0.99*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5))	
		for (int i = 0; i < colClus.size(); i++)	//mainCol�������д��ڼ�װ���ϰ벿��Ԫ�ص���ľ���
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

	colReDiv(mainCol);	//����У�����mainCol����������
	vector<Rect> containerType;
	for (int i = 0; colClus.begin()+i != colClus.end(); i++)	//ɾ��һЩ��
	{
		Rect rtmp = getColClusBbox(colClus[i]);
		int xCenter = rtmp.x + rtmp.width*0.5, yCenter = rtmp.y + rtmp.height*0.5;
		if (mainCol.size() < 9)
			if (xCenter >= mainBbox.x && xCenter <= mainBbox.x + mainBbox.width	//x�������غϣ���y�������غϣ�����
				&& yCenter < mainBbox.y && yCenter > mainBbox.y + mainBbox.height) 
					continue;
		if (abs(xCenter - mainXCenter) < 8 * mainCharW && abs(xCenter - mainXCenter) > 2 * mainCharW
			&& yCenter >= mainBbox.y && yCenter <= mainBbox.y + mainBbox.height)	// ˮƽ������mainCol��ԶҲ��̫��������ֱ�������غ�
			containerType = getBetterCol(mainCol, containerType, colClus[i]);
		colClus.erase(colClus.begin() + i);
		i--;
	}

	vector<Rect> keptBbox;	//��������Bbox
	for (int i = 0; i < colClus.size(); i++)
		keptBbox.insert(keptBbox.end(), colClus[i].begin(), colClus[i].end());
	colClus.clear();
	colClus.resize(3);	//�ֱ����˾�ţ���ţ����� 
	colClus[0].assign(mainCol.begin(), mainCol.end());
	colClus[1].insert(colClus[0].end(), keptBbox.begin(), keptBbox.end());
	colClus[2].assign(containerType.begin(), containerType.end());
}

int MserFilter::buildColResult(void)
{
	drawClus(savePath + "_00_t0.jpg", colClus);	
	colReDiv(colClus[0]);	//���ڵõ�����ŵ����ݽ����쳣��ȥ��������δ����ȹ���
	int mainCharH = getClusTypicalH(colClus[0]), mainCharW = getClusTypicalW(colClus[0]);
	rmColOutlier(colClus[0], mainCharH, mainCharW);
	colSelfPatch(colClus[0], mainCharH, mainCharW);
	if (colClus[2].size() != 0)
	{
		colReDiv(colClus[2]);
		rmColOutlier(colClus[2], mainCharH, mainCharW);
		colSelfPatch(colClus[2], mainCharH, mainCharW);
	}

	// ������ź����ͼ�Ĺ�ϵ������һ����������

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
	int charW = getClusTypicalW(mainClus);	//��ȡһ��������size
	int charH = getClusTypicalH(mainClus);
	if (newClus.size() == 1)
	{
		if (newClus[0].height < 0.8*charH || newClus[0].height > 1.3*charH || newClus[0].width < 0.5*charW)
			return oldClus;
	}

	return oldClus.size() > newClus.size() ? oldClus : newClus;
}

void MserFilter::colReDiv(vector<cv::Rect>& clus)	// �Ⱥϲ���ֱ�������غϵ�rect���ٽ��зָ�
{
	if (clus.size() == 0)
		return;
	int charH = getClusTypicalH(clus);
	sort(clus.begin(), clus.end(), sideSortByY);
	for (int i = 1; clus.begin() + i != clus.end(); i++)
	{
		double yCover = (clus[i - 1] & clus[i]).height;
		if (yCover > 0.15*clus[i - 1].height || yCover > 0.15*clus[i].height)	//˵�����Ժϲ�����ϲ�
		{
			clus[i - 1] = clus[i - 1] | clus[i];
			clus.erase(clus.begin() + i);
			i--;
		}
		else	//˵�������Ժϲ���������һ��bbox�����ܷ�ֲ�
		{
			double ratio = clus[i-1].height*1.0 / charH;
			int segCnt = 0;
			if (abs(ratio - round(ratio)) < 0.1+round(ratio)*0.1)
				segCnt = round(ratio);

			if (segCnt >= 2)
				if ((i>1 && clus[i-1].y - clus[i-2].y - clus[i-2].height <= 6)
					|| (clus[i].y - clus[i-1].y - clus[i-1].height <= 6))	//����Ǹոպϲ��ģ����߱������е�
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

	for (int i = 1; tmpClus.begin() + i != tmpClus.end(); i++)	// һЩ���˵ĵط������
	{
		int cnt = 0;
		Rect leastUp = tmpClus[i - 1].back();	//����һ���������������һ��
		int leastDis = getMinYDis(tmpClus[i][0], leastUp);
		double gapRatio = leastDis * 1.0 / mainCharH;
		//cout << gapRatio << endl;
		if (abs(gapRatio - round(gapRatio)) < 0.2+ round(gapRatio)*0.1)	//�����������һ���ַ��߶ȵ�������
			cnt = round(gapRatio);
		if (cnt > 0)	// �������ϲ���ȷʵ�ܵõ�����Ϊ4����6�Ľ��
			if((tmpBboxes.size() > 6 && i-1!=0 && cnt + tmpClus[i].size() + tmpClus[i - 1].size() == 6)	//ȷ���䲻����������Ǹ������
				|| cnt + tmpClus[i].size() + tmpClus[i - 1].size() == 4)
			{
				int avgH = leastDis / cnt;	// ���ɽ��
				for (int j = 0; j < cnt; j++)
					tmpClus[i - 1].push_back(Rect(tmpClus[i][0].x, leastUp.y + leastUp.height + leastDis * j / cnt, tmpClus[i][0].width, avgH));
				tmpClus[i - 1].insert(tmpClus[i - 1].end(), tmpClus[i].begin(), tmpClus[i].end());
				tmpClus.erase(tmpClus.begin() + i);
				i--;
			}
	}
	drawClus(savePath + "_00_t2.jpg", tmpClus);
	
	int maxIdx = getMaxClusIdx(tmpClus);
	if (tmpBboxes.size() > 7)	//�����һ��
	{
		if (tmpClus[maxIdx].size() < 6)	//�����Ҫ����
		{
			if (maxIdx - 1 >= 0 && tmpClus[maxIdx - 1].size() >= 4)	//���ù�˾��������
			{
				Rect compNumBottom = tmpClus[maxIdx - 1].back();	//��˾�����������bbox
				patchByBorder(tmpClus[maxIdx], mainCharH, compNumBottom.y+compNumBottom.height, 6);
			}
			else if (maxIdx + 1 < tmpClus.size() && tmpClus[maxIdx + 1][0].width>0.8*mainCharW && tmpClus[maxIdx + 1][0].height>0.8*mainCharH)	// ������֤��������
                patchByBorder(tmpClus[maxIdx], mainCharH, tmpClus[maxIdx+1][0].y, 6);
		}
		if(tmpClus[maxIdx].size() == 6)	//�������������ų���Ϊ6���Դ�Ϊ��׼���乫˾�ţ���֤��
		{
			if (maxIdx - 1 >= 0 && tmpClus[maxIdx - 1].size() < 4)	// ��˾�Ų���ȫ
			{
				patchByBorder(tmpClus[maxIdx - 1], mainCharH, tmpClus[maxIdx][0].y, 4);
				drawClus(savePath + "_00_t3.jpg", tmpClus);
			}
            else if(maxIdx == 0 /* || maxIdx - 1�϶�����*/)   //û�й�˾��
            {
                int dis = 0.35*mainCharH;
                if(maxIdx + 1 < tmpClus.size() && tmpClus[maxIdx + 1][0].width>0.8*mainCharW && tmpClus[maxIdx + 1][0].height>0.8*mainCharH)	//������֤����������֤��������
                    dis = tmpClus[maxIdx+1][0].y - tmpClus[maxIdx].back().y - tmpClus[maxIdx].back().height;
                vector<cv::Rect> compNum;
				for (int i = 1; i <= 4; i++)
					compNum.insert(compNum.begin(), Rect(tmpClus[maxIdx][0].x, tmpClus[maxIdx][0].y - dis - mainCharH*i, tmpClus[maxIdx][0].width, mainCharH));
                tmpClus.insert(tmpClus.begin()+maxIdx, compNum);
                maxIdx ++;
            }
			if (maxIdx + 1 == tmpClus.size() || tmpClus[maxIdx + 1][0].y - tmpClus[maxIdx].back().y > 2 * mainCharH
				|| tmpClus[maxIdx + 1][0].width<0.6*mainCharW || tmpClus[maxIdx + 1][0].height<0.6*mainCharH)	// û�к��ʵ���֤��
			{
				drawClus(savePath + "_00_t4.jpg", tmpClus);
                int dis = 0.35*mainCharH;
                if(maxIdx - 1>=0 && tmpClus[maxIdx - 1].size() == 4)//�Ϸ�����ȷ�Ĺ�˾�ţ������ݴ˹�˾�ż������
                    dis = tmpClus[maxIdx][0].y - tmpClus[maxIdx-1].back().y - tmpClus[maxIdx-1].back().height;
                Rect containNumBottom = tmpClus[maxIdx].back();
                vector<cv::Rect> vertiNum{Rect(containNumBottom.x, containNumBottom.y + dis, containNumBottom.width, 1.2*mainCharH)};
                tmpClus.insert(tmpClus.begin()+maxIdx+1, vertiNum);
			}
		}
	}

	int allSize = tmpBboxes.size();	//���һ�½��
	tmpBboxes.clear();
	tmpBboxes.insert(tmpBboxes.end(), tmpClus[maxIdx].begin(), tmpClus[maxIdx].end());
	if (allSize > 7)
	{
		if(maxIdx - 1 >= 0)
			tmpBboxes.insert(tmpBboxes.end(), tmpClus[maxIdx-1].begin(), tmpClus[maxIdx-1].end());
		if(maxIdx + 1 < tmpClus.size())
		tmpBboxes.insert(tmpBboxes.end(), tmpClus[maxIdx+1].begin(), tmpClus[maxIdx+1].end());
	}

	for (int i = 0; tmpBboxes.begin()+i!=tmpBboxes.end(); i++)	//�Լ�����Ŀ�����Խ���ˣ�����һ��ɾ��
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
    if(bboxes[0].y > border)  //  �����ϱ߽���patch
	{
        while (bboxes[0].y - border > mainCharH)	//��ͣ����䣬ֱ�����߾���ܽ�
            bboxes.insert(bboxes.begin(), Rect(bboxes[0].x, bboxes[0].y-mainCharH, bboxes[0].width, mainCharH));
        while (bboxes.size() < targetLen)	// �Ϸ��������ټ������·���
            bboxes.push_back(Rect(bboxes.back().x, bboxes.back().y + mainCharH, bboxes.back().width, mainCharH));
    }
    else if(bboxes.back().y + bboxes.back().height < border)    //�����±߽���patch
    {
		while (border - bboxes.back().y - bboxes.back().height > mainCharH)	//��ͣ����䣬ֱ�����߾���ܽ�
			bboxes.push_back(Rect(bboxes.back().x, bboxes.back().y + mainCharH, bboxes.back().width, mainCharH));
		while (bboxes.size() < targetLen)   // �·����׺��ټ������Ϸ���
			bboxes.insert(bboxes.begin(), Rect(bboxes[0].x, bboxes[0].y - mainCharH, bboxes[0].width, mainCharH));
	}
}

void MserFilter::simpleClus(vector<vector<cv::Rect>> &tmpClus, vector<cv::Rect> bboxes, double disThre)
{
	//�Ƚ��о��࣬����ţ���˾�ţ�У����ֱ��ó���
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
		if (clus[i].width > clus[i].height || clus[i].area() > 2.5*mainCharH*mainCharW)	//һЩ�ǳ����Ե�outlier��ֱ��ɾ��
		{
			clus.erase(clus.begin() + i);
			i--;
		}
}

MserFilter::~MserFilter()
{

}

// �ж������Ƿ���Ա���Ϊһ��
bool MserFilter::isClose(Rect r1, Rect r2, int disThres, int mode)
{
	int r1CenterX = r1.x + r1.width / 2, r1CenterY = r1.y + r1.height / 2;
	int r2CenterX = r2.x + r2.width / 2, r2CenterY = r2.y + r2.height / 2;
	if (mode == COL_MODE || mode == COL_LOOSE_MODE)		//��Ҫ��һ�о�Ϊһ��
	{
		double xCoverRatio = 0.65;
		if (mode == COL_LOOSE_MODE) //�����о��࣬�ſ�������
			xCoverRatio = 0.2;
		double xCover = min(abs(r1.x - r2.x - r2.width), abs(r2.x - r1.x - r1.width));//����rect֮��x�����غϵĳ���
		if(max(abs(r1.x - r2.x - r2.width), abs(r2.x - r1.x - r1.width)) < r1.width + r2.width)
			if (xCover / r1.width > xCoverRatio || xCover / r2.width > xCoverRatio) // ����Rect�Ƿ���һ����
				if (abs(r1CenterY - r2CenterY) < disThres)		//����distance��������
					return true;
	}
	if (mode == ROW_MODE || mode == ROW_LOOSE_MODE)		// ��һ�о�Ϊһ��
	{
		double yCoverRatio = 0.8;
		if (mode == ROW_LOOSE_MODE) //�����о��࣬�ſ�������
			yCoverRatio = 0.6;
		double yCover = min(abs(r1.y - r2.y - r2.height), abs(r2.y - r1.y - r1.height));//����rect֮��y�����غϵĳ���
		if (max(abs(r1.y - r2.y - r2.height), abs(r2.y - r1.y - r1.height)) < r1.height + r2.height)
			if (yCover / r1.height > yCoverRatio || yCover / r2.height > yCoverRatio) // ����Rect�Ƿ���һ���ϣ��˴�Ӧ�ø��ϸ�һ��
				if (abs(r1CenterX - r2CenterX) < disThres) //����distance
					return true;
	}	
	return false;
}

int MserFilter::judgeSide(void)
{
	return filter();
}
