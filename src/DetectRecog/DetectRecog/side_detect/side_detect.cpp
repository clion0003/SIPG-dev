#include "side_detect.h"

void sideDetect(char *outputStr, const char *inputPath)
{
	string imgPath(inputPath);
	string tmpImgPath("D:/SideTmpDir/");
	// 应该加一个文件夹的判断

	SynthDetector sd(imgPath, tmpImgPath);
	string output = sd.getStr();
	strcpy(outputStr, output.c_str());
}