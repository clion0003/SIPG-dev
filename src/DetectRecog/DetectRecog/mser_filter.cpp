#include "mser_filter.h"
using namespace cv;

extern double charHeightSmall, charHeightHuge, charWidthHuge;

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

void MserFilter::drawMsers(string outputPath)
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
void MserFilter::drawClusters(string outputPath, vector<vector<Rect>> clusters)
{
	Mat mserImg = Mat::zeros(srcImg.size(), CV_8UC1);
	int clusterNum = 0;
	for (int i = 0; i < clusters.size(); i++)
		if (clusters[i].size() != 0)
			clusterNum++;

	int clusterCnt = 0;
	for (int i = 0; i < clusters.size(); i++)
	{
		if(clusters[i].size() != 0)
		{
			clusterCnt++;
			int color = (240 / clusterNum) * clusterCnt;
			for (int j = 0; j < clusters[i].size(); j++)
			{
				//color = 250;
				rectangle(mserImg, clusters[i][j], color, CV_FILLED, 8, 0);
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

void MserFilter::filter(void)
{
	deeplabFilter();
	singleBboxFilter();
	clusterFilter();
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

void MserFilter::singleBboxFilter(void)
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
			if (((double)(bboxes[i] & bboxes[j]).area()) / bboxes[i].area() > 0.25
				|| ((double)(bboxes[i] & bboxes[j]).area()) / bboxes[j].area() > 0.25)
			{
				if (bboxes[i].area() < bboxes[j].area())
				{
					bboxes.erase(bboxes.begin() + i);
					i--;
					break;
				}
				else
				{
					bboxes.erase(bboxes.begin() + j);
					j--;
				}
			}
}

void MserFilter::clusterFilter(void) //���˲�ɾbbox��Ҫ��ʼ������
{
	// ��bbox������������࣬����Ľ��ΪMserFilter��Ա
	cluster(mserWidthHuge*2.5, bboxes, rowClusters, ROW_MODE);
	cluster(mserHeightHuge*2, bboxes, colClusters, COL_MODE);
	
	int mainRowSize = clusterProcess(ROW_MODE);
	int mainColsize = clusterProcess(COL_MODE);

	if (mainRowSize==0 && mainColsize==0)	//��Ϊ0
	{
		rowClusters.clear();
		colClusters.clear();
	}
	else if(mainRowSize==0 && mainColsize != 0)	//��һ��Ϊ0
		rowClusters.clear();
	else if (mainColsize == 0 && mainRowSize != 0)
		colClusters.clear();
	else	//����Ϊ0
	{
		if((mainColsize >= 6 && mainRowSize <= 4) || mainColsize >= 11 || mainColsize / mainRowSize >= 2)	// col����õ��Ľ�����ã�����col��
			rowClusters.clear();
		else if ((mainRowSize >= 6 && mainColsize <= 4) || mainRowSize >= 11 || mainRowSize / mainColsize >= 2)
			colClusters.clear();
		else	// hard���ж�˭�����ұ�һ��
		{
			if (rowResult[0].x + 0.5*rowResult[0].width >= colResult[0].x)
				colClusters.clear();
			else
				rowClusters.clear();
		}
	}
}

// �о�����о��࣬ȡ����mode��mode�ᴫ��isClose()�����ж��Ƿ�����һ��(�з�������з���)
void MserFilter::cluster(int disThres, vector<Rect> bboxes, vector<vector<Rect>>& clusters,int mode)
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
					int srcClass = idx[j]>idx[i] ? idx[j] : idx[i];
					int desClass = idx[i] == srcClass ? idx[j] : idx[i];
					for (int k = 0; k < idx.size(); k++)
						if (idx[k] == srcClass)
							idx[k] = desClass;
				}
			}

	// ����������˳��������棬һ�����Ӧһ��˳���������֮��Ĳ���
	// �˴�����ʹ�ö��forѭ��
	clusters.clear();
	clusters.resize(idx.size());
	for (int i = 0; i < idx.size(); i++)
		clusters[idx[i]].push_back(bboxes[i]);
	for (int i = 0; clusters.begin() + i != clusters.end(); i++)
	{
		if (clusters[i].size() <= 0)
		{
			clusters.erase(clusters.begin()+i);
			i--;
		}
	}
}

//���������rowCluster��size
int MserFilter::clusterProcess(int mode)
{
	vector<Rect> mainCluster;
	if(mode == ROW_MODE)	//�ҵ������
		findMainRowCluster(mainCluster);
	else
		findMainColCluster(mainCluster);

	if (mainCluster.size() < 4)
		return 0;

	if (mode == ROW_MODE)	//�������������һ����filter��patch
	{
		delByMainRow(mainCluster);
		buildRowResult();
	}
	else
	{
		delByMainCol(mainCluster);
		buildColResult();
	}
	return mainCluster.size();
}

void MserFilter::findMainRowCluster(vector<Rect>& mainRowCluster)
{
	//��ɾ���ȽϿ��µ���,�������
	vector<vector<Rect>> clusterTmp(rowClusters);
	for (int i = 0; clusterTmp.begin() + i != clusterTmp.end(); i++)
		//���������λ�ù��ߵ���ͼƬ��һ���ھ�ͷ����ʱ���������ִ���
		if (!(deeplabBbox.y + deeplabBbox.height > 0.95*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5))
			if ((clusterTmp[i][0].y - deeplabBbox.y)*1.0 / deeplabBbox.height > 0.4)
			{
				clusterTmp.erase(clusterTmp.begin() + i);
				i--;
			}
	// �ӳ���������Ĭ������Ǹ���ΪmainRow
	int mainRow = 0;
	if (clusterTmp.size() == 0)
		return;
	sort(clusterTmp.begin(), clusterTmp.end(), sortByLen);
	if (clusterTmp.size() > 1)
		if (clusterTmp[0].size() >= 6 && clusterTmp[1].size() >= 6)	//ǰ�������ܳ�����ѡ���ұߵ��Ǹ�
		{
			sort(clusterTmp[0].begin(), clusterTmp[0].end(), sortByX);
			sort(clusterTmp[1].begin(), clusterTmp[1].end(), sortByX);
			if (clusterTmp[1][0].x > clusterTmp[0][0].x)
				mainRow = 1;
		}
	mainRowCluster.assign(clusterTmp[mainRow].begin(), clusterTmp[mainRow].end());
}

void MserFilter::delByMainRow(vector<Rect> mainRowCluster)
{
	vector<Rect> clusterBboxes(rowClusters.size());	//����ÿ�������bbox
	vector<int> avgWidth(rowClusters.size());		//����ƽ��size
	vector<int> avgHeight(rowClusters.size());
	for (int i = 0; i < rowClusters.size(); i++)
	{
		getClusterBbox(rowClusters[i], clusterBboxes[i]);
		getClusterAvgHW(rowClusters[i], avgHeight[i], avgWidth[i]);
	}
		
	//����mainRowCluster����ز���
	Rect mainBbox;
	int mainAvgHeight, mainAvgWidth;
	getClusterBbox(mainRowCluster, mainBbox);
	getClusterAvgHW(mainRowCluster, mainAvgHeight, mainAvgWidth);
	int mainXCenter = mainBbox.x + 0.5*mainBbox.width;
	int mainYCenter = mainBbox.y + 0.5*mainBbox.height;

	for (int i = 0; i < rowClusters.size(); i++)	// ����mainRowɾ��
	{
		int xCenter = clusterBboxes[i].x + 0.5*clusterBboxes[i].width;
		int yCenter = clusterBboxes[i].y + 0.5*clusterBboxes[i].height; 
		// �˴�����ֻ����mainRow���ϣ����£����������������(������б�ŵ�����)��xCenter�����������һ������Ϊ��ʱ��mainRow��߻�©�켸λ
		if((xCenter+0.25*clusterBboxes[i].width< mainBbox.x || xCenter > mainBbox.x + mainBbox.width)
			&&(yCenter < mainBbox.y || yCenter > mainBbox.y + mainBbox.height))
		{
			rowClusters[i].clear();
			continue;
		}
		// ɾ����mainRow��y�����Զ����
		if ((mainRowCluster.size() <= 8 && abs(yCenter - mainYCenter) >= 2 * mainBbox.height)	//�Ų������е����
			|| (mainRowCluster.size()>8 && (yCenter-mainYCenter>=2*mainBbox.height||yCenter-mainYCenter<=-0.75*mainBbox.height)))	//�Ų������е��������ʱ�Ϸ��ļ���ŵĸ���
		{
			rowClusters[i].clear();
			continue;
		}
		if (clusterBboxes[i].x - mainBbox.x - mainBbox.width >= 8*mainAvgWidth
			|| mainBbox.x-clusterBboxes[i].x- clusterBboxes[i].width>=8* mainAvgWidth)	// ɾ����mainRow��x�����Զ����
		{
			rowClusters[i].clear();
			continue;
		}
	}

	//��ɾ���������չ�����ɵĽ����������һ��
	vector<Rect> rowFilteredBboxes;
	for (int i = 0; i < rowClusters.size(); i++)
		for (int j = 0; j < rowClusters[i].size(); j++)
			rowFilteredBboxes.push_back(rowClusters[i][j]);
	cluster(1000, rowFilteredBboxes, rowClusters, ROW_LOOSE_MODE);	//���¾��࣬һ����Ϊһ�࣬��ʱ�����������ɣ�����������rowCluster��
}

void MserFilter::buildRowResult(void)
{
	sort(rowClusters.begin(), rowClusters.end(), sortByLen);

	for (int i = 0; i < rowClusters.size() && i<3; i++)	//�����������Ϊǰ�������Ҵ�bbox
	{
		sort(rowClusters[i].begin(), rowClusters[i].end(), sortByX);
		int minYTop = 10000, maxYBottom = 0;
		for (int j = 0; j < rowClusters[i].size(); j++)
		{
			minYTop = min(minYTop, rowClusters[i][j].y);
			maxYBottom = max(maxYBottom, rowClusters[i][j].y + rowClusters[i][j].height);
		}
		int endIdx = rowClusters[i].size() - 1;
		rowResult.push_back(Rect(rowClusters[i][0].x, minYTop, rowClusters[i][endIdx].x + rowClusters[i][endIdx].width - rowClusters[i][0].x, maxYBottom - minYTop));
	}

	//�������������Ӧ�����������
	int farLeft = rowResult[0].x;
	for (int i = 1; i < rowResult.size(); i++)
		farLeft = min(rowResult[i].x, farLeft);
	rowResult[0].width = rowResult[0].x - farLeft + rowResult[0].width;
	rowResult[0].x = farLeft;

	//���е��������Ĺ�˾��Ӧ�ú�mainRow����룬�����������������mainRow����벿�֣���mainRow��ֻ����ŵģ�ҲӦ������룬�Ҵ�ʱ��ź�����ҲӦ���Ҷ���
	for (int i = 1; i < rowResult.size() && rowResult.size()==3; i++)
	{
		if (rowResult[i].y < rowResult[0].y)	//����˾�ŵ����
		{
			rowResult[i].width = rowResult[i].x + rowResult[i].width - rowResult[0].x;
			rowResult[i].x = rowResult[0].x;
		}
		if(rowResult[i].y > rowResult[0].y && rowResult[i].x+rowResult[i].width*0.5 < rowResult[0].x+rowResult[0].width*0.5 && rowClusters[0].size() <= 8)
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
}

void MserFilter::findMainColCluster(vector<Rect>& mainColCluster)
{
	// �ϲ���������б���Ӧ�ó��ֵĲ��ŵ�bbox
	for (int i = 0; i < colClusters.size(); i++)	
	{
		sort(colClusters[i].begin(), colClusters[i].end(), sortByY);	//���ϵ�������һ��
		for (int j = 1; colClusters[i].begin() + j != colClusters[i].end(); j++)
			if ((colClusters[i][j - 1].y + colClusters[i][j - 1].height - colClusters[i][j].y)*1.0 / colClusters[i][j - 1].height > 0.7)	//��ֱ�����غ϶ȴ���0.7
			{
				colClusters[i][j - 1] = colClusters[i][j - 1] | colClusters[i][j];
				colClusters[i].erase(colClusters[i].begin() + j);
				j--;
			}
	}
	if (colClusters.size() == 0)
		return ;

	int mainCol = 0;	//�������
	sort(colClusters.begin(), colClusters.end(), sortByLen);	//�ӳ���������
	//�����װ������λ�ö������ڻ����У���Ҫ��mainCol������Ԫ�ش��ڼ�װ���ϰ벿��������ֱ��ѡȡ����Ǹ�
	if (!(deeplabBbox.y + deeplabBbox.height>0.99*srcImg.rows && deeplabBbox.y > srcImg.rows*0.5))	
		for (int i = 0; i < colClusters.size(); i++)	//mainCol�������д��ڼ�װ���ϰ벿��Ԫ�ص���ľ���
			if (colClusters[i][0].y - deeplabBbox.y < 0.4*deeplabBbox.height)
			{
				mainCol = i;
				break;
			}
	mainColCluster.assign(colClusters[mainCol].begin(), colClusters[mainCol].end());
}

void MserFilter::delByMainCol(vector<Rect> mainColCluster)
{
	int mainAvgWidth = 0, mainAvgHeight = 0;	// ��ȡmainCluster��bbox��ز�����������Щ������ɾ��һЩ��
	Rect mainBbox;
	getClusterAvgHW(mainColCluster, mainAvgHeight, mainAvgWidth);
	getClusterBbox(mainColCluster, mainBbox);
	int mainXCenter = mainBbox.x + mainBbox.width*0.5;

	for (int i = 0; colClusters.begin()+i != colClusters.end(); i++)	//ɾ��һЩ��
	{
		Rect rtmp;
		getClusterBbox(colClusters[i], rtmp);
		int xCenter = rtmp.x + rtmp.width*0.5, yCenter = rtmp.y + rtmp.height*0.5;
		bool deleteFlag = false;
		if ((xCenter<mainBbox.x || xCenter>mainBbox.x + mainBbox.width) //x, y�����û���غϵ��࣬ɾ��
			&& (yCenter<mainBbox.y || yCenter>mainBbox.y + mainBbox.height))
			deleteFlag = true;
		if (abs(xCenter - mainXCenter) >= 8 * mainAvgWidth)	// ɾ��ˮƽ������mainCol��Զ����
			deleteFlag = true;
		if(deleteFlag == true)
		{
			colClusters.erase(colClusters.begin() + i);
			i--;
		}
	}

	//��ɾ����Ľ����������һ�£��ٽ�����ֱ����ľ���
	vector<Rect> colFilteredBboxes;
	for (int i = 0; i < colClusters.size(); i++)
		for (int j = 0; j < colClusters[i].size(); j++)
			colFilteredBboxes.push_back(colClusters[i][j]);
	cluster(2000, colFilteredBboxes, colClusters, COL_MODE);	//���¾��࣬�˴���ֵ��ĺܴ�ֻҪ��һ�������ϼ�����Ϊ��һ��
}

void MserFilter::buildColResult(void)
{
	sort(colClusters.begin(), colClusters.end(), sortByLen);
	// �������еĽ�һ�����࣬�Խ���colCluster[0](Ҳ��mainCol)��������Ŀ���Ǿ�Ϊ����(4+6+1)
	vector<vector<Rect>> mainCol;
	int mainAvgHeight = 0, mainAvgWidth = 0;
	getClusterAvgHW(colClusters[0], mainAvgHeight, mainAvgWidth);
	cluster(1.25*mainAvgHeight, colClusters[0], mainCol, COL_MODE);
	/*for (int i = 0; mainCol.begin() + i + 1 != mainCol.end(); i++)	//��һ�׶Σ���mainCol[0]��sizeУ����4
	{
		if (mainCol[0].size() >= 4)
			break;
		if(mainCol[0] + mainCol[1])
	}*/

	//���з�mainCol�Ĵ���Ҫ�ѷ�mainCol�ϲ���һ��
	for (int i = 1; i < colClusters.size(); i++)
	{
		if (i == 1 || colClusters[i].size() >= 4)
		{
			vector<vector<Rect>> tmpCluster;
			int avgHeight = 0, avgWidth = 0;
			getClusterAvgHW(colClusters[0], avgHeight, avgWidth);
			cluster(1.25*mainAvgHeight, colClusters[i], tmpCluster, COL_MODE);
			sort(tmpCluster.begin(), tmpCluster.end(), clusterSortByY);

			int cnt = tmpCluster.size();
			while(tmpCluster.size() > 1)	//
			{
				cnt--;
				if (cnt < 0)
					break;
				if (tmpCluster[0].size() == 4)
				{
					colClusters[i].assign(tmpCluster[0].begin(), tmpCluster[0].end());
					break;
				}
				if (tmpCluster[1].size() > 4)
				{
					colClusters[i].clear();
					break;
				}
				if (tmpCluster[1].size() == 4)
				{
					colClusters[i].assign(tmpCluster[1].begin(), tmpCluster[1].end());
					break;
				}
				if (tmpCluster[0].size() + tmpCluster[1].size() >= 4)
				{
					if (tmpCluster[0].size() < tmpCluster[1].size())
						tmpCluster[0].assign(tmpCluster[1].begin(), tmpCluster[1].end());
					tmpCluster.erase(tmpCluster.begin() + 1);
				}
			}
		}
	}

	for (int i = 0; i < colClusters.size() && i<2; i++)
		for (int j = 0; j < colClusters[i].size(); j++)
		colResult.push_back(colClusters[i][j]);
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
		if (mode == ROW_LOOSE_MODE) //�����о��࣬�ſ�������
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
	filter();
	if (rowClusters.size() != 0)
		return ROW_MODE;
	else if (colClusters.size() != 0)
		return COL_MODE;
	else
		return WRONG;

}
