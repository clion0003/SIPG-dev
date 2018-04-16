#include "side_detect.h"
#include <ctime>
#include <fstream>

using namespace std;
using namespace cv;

int main(void)
{
	string imgDir("C:/Users/yuecore-sg001/Desktop/tmpTest/");
	//string saveDir("C:/Users/yuecore-sg001/Desktop/colHardRes/");
	//string tmpSaveDir("D:/SideSaveImg/");
	//string deeplabRes("C:/Users/yuecore-sg001/Desktop/colDeeplab.txt");
	//string imgDir("C:/Users/yuecore-sg001/Desktop/speSrc/");
	//string saveDir("C:/Users/yuecore-sg001/Desktop/speGen/");

	vector<string> fileNames = getDirFileNames(imgDir);
	//ofstream ofs(deeplabRes, ios::out);
	//ifstream ifs(deeplabRes, ios::in);
	// 开始逐张处理图片
	for (int i = 0; i < fileNames.size(); i++)
	{
		clock_t start = clock();
		char buf[100];
		cout << fileNames[i] << endl;
		sideDetect(buf, (imgDir+fileNames[i]).c_str());
		cout << buf << endl;
		clock_t end = clock();
		cout << "Time: " << end - start << "ms" << endl;
	}
	system("pause");
}