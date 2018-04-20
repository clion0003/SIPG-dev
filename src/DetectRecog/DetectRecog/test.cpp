#include "detect\detect.h"
#include "side_detect\side_detect.h"
#include "socketcpp\tcpClient.h"

#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <ctime>

using namespace std;

int main() {
    cout << "helloworld" << endl;
    vector<bool> results;
    string path = "G:\\xianghao\\0001\\";
   /*
    front_classifier_request(path, results);
    for (auto it : results) {
        cout << it << endl;
    }
    */
    vector<string> fileNames = getDirFileNames(path);
    
    for (auto it : fileNames) {
        clock_t start = clock();
        char buf[100];
        cout << it << endl;
        if(it.find("Side",0)!=string::npos) sideDetect(buf, (path + it).c_str());
        else detect(buf, (path + it).c_str(), AUTO);
        cout << buf << endl;
        clock_t end = clock();
        cout << "Time: " << end - start << "ms" << endl;
    }
   
    system("pause");
    return 0;
}