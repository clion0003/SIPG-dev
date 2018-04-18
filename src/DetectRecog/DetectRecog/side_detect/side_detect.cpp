#include "side_detect.h"

void sideDetect(char *outputStr, const char *inputPath)
{
	string imgPath(inputPath);
	string tmpImgPath("D:/SideTmpDir/");
	//if (_access(tmpImgPath.c_str(), 0) == -1)
	//	CreateDirectory(tmpImgPath.c_str(), NULL);

	clearDir(tmpImgPath);
	SynthDetector sd(imgPath, tmpImgPath);
	string output = sd.getStr();
	transform(output.begin(), output.end(), output.begin(), ::toupper);
	strcpy(outputStr, output.c_str());
}