#include "synth_detector.h"

SynthDetector::SynthDetector(string imgPath, string tmpSavePath)
{
	this->imgPath = imgPath;
	this->tmpSavePath = tmpSavePath;
	this->srcImg = cv::imread(imgPath);
}

string SynthDetector::getStr(void)
{
	cv::Rect deeplabBbox;
	initSocket();
	deeplab_request(imgPath, deeplabBbox);
	vector<vector<cv::Rect>> mserRes, eastRes;
	getMserRes(mserRes, deeplabBbox);
	getEastRes(eastRes, deeplabBbox);
	synthRes(mserRes, eastRes);
	string resStr;
	if (resMode == COL_MODE)
	{
		string colStr;
		initSocket();
		alex_request(tmpSavePath, colStr);
		colStr.insert(mserRes[0].size(), " ");
		colStr.insert(mserRes[1].size() + mserRes[0].size() + 1, " ");
		resStr = colStr;
	}
	else if (resMode == ROW_MODE)
	{
		vector<string> rowStr;
		initSocket();
		crnn_request(tmpSavePath, rowStr);
		resStr = rowStr[0];
		for (int i = 1; i < rowStr.size(); i++)
			resStr = resStr + ' ' + rowStr[i];
	}
	return resStr;
}

void SynthDetector::getMserRes(vector<vector<cv::Rect>> &mserRes, cv::Rect deeplabBbox)
{
	MserFilter mf(srcImg, deeplabBbox);
	resMode = mf.filter();
	if (resMode == COL_MODE)
		mserRes = mf.colClus;
	else if (resMode == ROW_MODE)
		mserRes = mf.rowClus;
}

void SynthDetector::getEastRes(vector<vector<cv::Rect>> &eastRes, cv::Rect deeplabBbox)
{
	//EastFilter ef(imgPath, deeplabBbox);
}

int SynthDetector::synthRes(vector<vector<cv::Rect>> &mserRes, vector<vector<cv::Rect>> &eastRes)
{
	/******** ½øÐÐ×ÛºÏ ********/

	/*************************/

	int cnt = 0;
	for (int i = 0; i < mserRes.size(); i++)
		for (int j = 0; j < mserRes[i].size(); j++)
		{
			cout << mserRes[i][j].width << endl;
			if (mserRes[i][j].height != 0)
			{
				cv::Mat tmpImg(srcImg, mserRes[i][j]);
				cv::imwrite(tmpSavePath + to_string(cnt) + ".jpg", tmpImg);
				cnt++;
			}
		}
	return 0;
}