#include"TextDetection.h"
#include <string>
#include <vector>
#include <fstream>
using namespace std;
using namespace cv;

int main()
{
	vector<string> gtset;string line;
	cv::Mat image = cv::imread("./resource/demo.jpg");
	ifstream fin("./resource/gt_demo");
	while(getline(fin,line))
	{
		gtset.push_back(line);
	}
	TextDetection td;
	cout<<td.detectText(image,gtset)<<endl;
	cv::waitKey();
}
