#include "side_detect.h"

void sideDetect(char *outputStr, const char *inputPath)
{
	string imgPath(inputPath);
	string tmpImgPath("D:/SideTmpDir/");
	// Ӧ�ü�һ���ļ��е��ж�

	SynthDetector sd(imgPath, tmpImgPath);
	string output = sd.getStr();
	strcpy(outputStr, output.c_str());
}