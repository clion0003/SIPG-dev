#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>    //max()����
#include <string>

using namespace std;

void lcss(const string str1, const string str2, string& str) 
{
	int size1 = str1.length();
	int size2 = str2.length();
	const char* s1 = str1.data() - 1;//��1��ʼ�����������Ĵ����д(���ַ���ָ��str1ָ���� ��Ԫ�ص�ǰ�档)��������0��/�У�LCS����c[0][j],c[i][0]) Ϊ0��
	const char* s2 = str2.data() - 1;
	vector<vector<int>> chess(size1 + 1, vector<int>(size2 + 1));
	//��Ϊ�Ǵ�1����������0��/������ӵģ���Ϊ�ַ�����մ���LCS��ԶΪ0��;�����ܳ���Ҫ+1
	//chess���̣���ά���飬����Ϊsize1������Ϊsize2
	int i, j;
	//for (i = 0; i < size1; i++) {//��0�г�ʼ��Ϊ0����ʵ�Զ���ʼ���ģ����Բ�д��
	//  chess[i][0] = 0;
	//}
	//for (j = 0; j < size2; j++) {//��0�г�ʼ��Ϊ0
	//  chess[0][j] = 0;
	//}

	for (i = 1; i < size1; i++) { // һ��Ҫ��1�𣡣���
		for (j = 1; j < size2; j++) {
			if (s1[i] == s2[j]) {
				//��������ǰԪ����ͬ�����䵱ǰLCS�������Ͻ�LCS��chess[i - 1][j - 1] ��+ 1���˴��ӵ�1����Ϊ��ǰԪ��(i,j).��
				chess[i][j] = chess[i - 1][j - 1] + 1;
			}
			else {  // if (s1[i] != s2[j])
				//��������ǰԪ�ز�ͬ�����䵱ǰLCSһ��������ߡ��ϱߵ�LCS�����ֵ���������ˣ�i,j���������ǹ���Ԫ�ء���������
				chess[i][j] = max(chess[i - 1][j], chess[i][j - 1]);
			}
		}
	}
	//���ˣ��ű����̣�������

	i = size1;//��i��j�±��䵽ĩβԪ���ϡ�
	j = size2;
	while (i != 0 && j != 0)
	{
		if (s1[i] == s2[j]) {   //����ͬ����Ԫ��ѹջ��Ȼ��ָ��ǰ�ƣ�ֱ��i��jָ��0��ֹ(��Ϊ�κ��ַ��� ��0 �󹫹������У�����0)
			str.push_back(s1[i]);
			i--;
			j--;
		}
		else { //�����߲���ȣ��������������һ������LCS��chess[i][j-1] or chess[i-1][j]���Ľϴ��ߵ������ʽ��ϴ��ߵ�ָ��ǰ�ƣ����ű�����
			if (chess[i][j - 1] > chess[i - 1][j]) {
				j--;        //����ǰ��ǰ�Ƶ�j-1��
			}
			else { // if(chess[i][j - 1] <= chess[i - 1][j])
				i--;
			}
		}
	}
	//��LCS��֮һ��
	reverse(str.begin(), str.end());

	//��LCS�ĳ���
	/*
	int maxLen = 0;
	for (i = 0; i < size1; i++){
		for (j = 0; j < size2; j++){
			maxLen = max(maxLen, chess[i][j]);
		}
	}
	return  maxLen;
	*/

}