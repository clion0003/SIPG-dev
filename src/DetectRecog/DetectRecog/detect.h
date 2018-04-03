#pragma once
#include<string>
#include<vector>
using std::vector;
using std::string;

#ifdef DLL_GEN
#define DLL_API _declspec(dllexport) 
#else
#define DLL_API _declspec(dllimport) 
#endif
/*input1: ��һ������ͼƬ·��
input2: �ڶ�������ͼƬ·��
output1: ��һ��ͼʶ����
output2: �ڶ���ͼʶ����
mode: =0 �Զ��ж�������
=1 �趨����һ��ͼΪ���棬�ڶ���ͼΪ����
*/
#define AUTO 0
#define FCOL_MODE 1
#define FROW_MODE 2

DLL_API bool detect(char* output, const char* input, int side);
DLL_API bool detectSide(char *output,const char *input, int side);