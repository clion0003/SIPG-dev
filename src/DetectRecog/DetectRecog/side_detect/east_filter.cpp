#include "east_filter.h"

using namespace cv;

EastFilter::EastFilter(string imgPath, Rect deeplabBbox)
{
	initSocket();
	east_request(imgPath, eastBboxes);
	this->srcImg = imread(imgPath);
	this->deeplabBbox = deeplabBbox;
	for (int i = 0; i < eastBboxes.size(); i++)
		this->bboxes.push_back(eastbox2rect(eastBboxes[i]));
	
	//�趨��ֵ
	double heightRatio = (double(deeplabBbox.height)) / srcImg.rows;
	//������һ����ͷ�����λ����ʱ����߽���һ��У��
	if (heightRatio < 0.5 && deeplabBbox.width>2 * deeplabBbox.height && deeplabBbox.y > srcImg.rows / 2)
		heightRatio += 0.2;
	switch ((int)(heightRatio * 10))
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

void EastFilter::drawBboxes(string outputPath)
{
	Mat mserImg = srcImg;
	for (int i = 0; i < bboxes.size(); i++)
	{
		Rect bbox = bboxes[i];
		rectangle(mserImg, bbox, Scalar(0, 255, 0), 1, 8, 0);
	}
	//rectangle(mserImg, deeplabBbox, Scalar(0, 255, 0), 1, 8, 0);
	imwrite(outputPath, mserImg);
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
		if (bboxes[i].area() > mserHeightHuge*mserWidthHuge*20	//�ر���bboxesɾ��
			|| bboxes[i].area() < mserHeightHuge*mserWidthHuge  //�ر�С��ɾ��
			|| (bboxes[i].width>bboxes[i].height && ((bboxes[i].y-deeplabBbox.y)*1.0/deeplabBbox.height>0.4))) //ˮƽbbox���ر��µ�ɾ��					
		{
			bboxes.erase(bboxes.begin() + i);
			i--;
		}
	}
}

void EastFilter::clusterFilter(void)
{
	// idx�д����ÿ��bbox�������࣬��ʼʱÿ��bbox�������Լ�һ��
	vector<int> idx(bboxes.size());
	for (int i = 0; i < idx.size(); i++)
		idx[i] = i;

	// ��ʼ���࣬��forѭ�������һ����bboxes�ȳ������飬������Ԫ�ر�ʾ��Ӧ��bbox��������
	for (int i = 0; i < bboxes.size(); i++)
		for (int j = i + 1; j < bboxes.size(); j++)
			if (isClose(bboxes[i], bboxes[j]) == true)
			{
				if (idx[j] == idx[i])	//����ͬһ����
					continue;
				else if (idx[j] == j)	//��������
					idx[j] = idx[i];
				else                    //close��������һ���࣬�ϲ���������
				{
					int srcClass = idx[j] > idx[i] ? idx[j] : idx[i];
					int desClass = idx[i] == srcClass ? idx[j] : idx[i];
					for (int k = 0; k < idx.size(); k++)
						if (idx[k] == srcClass)
							idx[k] = desClass;
				}
			}

	// ����������˳��������棬һ�����Ӧһ��˳���������֮��Ĳ���
	// �˴�����ʹ�ö��forѭ��
	vector<vector<Rect>> bboxClusters(idx.size());
	for (int i = 0; i < idx.size(); i++)
		bboxClusters[idx[i]].push_back(bboxes[i]);
	for (int i = 0; bboxClusters.begin() + i != bboxClusters.end(); i++)
	{
		if (bboxClusters[i].size() == 0)
		{
			bboxClusters.erase(bboxClusters.begin() + i);
			i--;
		}
	}
	clusters = bboxClusters;
}

Rect EastFilter::eastbox2rect(EastBbox bbox)
{
	int x = min(bbox.x0, bbox.x3);
	int y = min(bbox.y0, bbox.y1);
	int width = max(bbox.x1, bbox.x2) - x;
	int height = max(bbox.y2, bbox.y3) - y;
	return Rect(x, y, width, height);
}

bool EastFilter::isClose(Rect r1, Rect r2)
{
	int r1CenterX = r1.x + r1.width / 2, r1CenterY = r1.y + r1.height / 2;
	int r2CenterX = r2.x + r2.width / 2, r2CenterY = r2.y + r2.height / 2;

	// x�������غϣ�����y����ľ���
	if (max(abs(r1.x - r2.x - r2.width), abs(r2.x - r1.x - r1.width)) < r1.width + r2.width)
		if (abs(r1CenterY - r2CenterY) - (r1.height+r2.height)/2 < 150)		//����distance��������
			return true;

	// y�������غϣ�����x����ľ���
	if (max(abs(r1.y - r2.y - r2.height), abs(r2.y - r1.y - r1.height)) < r1.height + r2.height)
		if (abs(r1CenterX - r2CenterX) - (r1.width+r2.width)/2 < 150)		//����distance
				return true;
	return false;
}