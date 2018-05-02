#include "detect\detect.h"
#include "side_detect\side_detect.h"
#include "socketcpp\tcpClient.h"

#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <ctime>
#include <fstream>
using namespace std;

void test() {
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
        if (it.find("Side", 0) != string::npos) sideDetect(buf, (path + it).c_str());
        else detect(buf, (path + it).c_str(), AUTO);
        cout << buf << endl;
        clock_t end = clock();
        cout << "Time: " << end - start << "ms" << endl;
    }

}


void help() {
    string path = "G:\\xianghao\\";
    vector<string> fileNames = getDirFileNames(path);
    fstream out("C:\\Users\\archlab\\Desktop\\out4.txt");
    if (out) cout << "success" << endl;
    else return;
    for (auto it : fileNames) {
        
        if (it.length() == 4 && it[0] == '0') {
            cout << it << endl;
            out << it << endl;
            string p = path + it + "\\";
            vector<string> fnames = getDirFileNames(p);
            for (auto f : fnames) {
                clock_t start = clock();
                char buf[100];
                out << f << endl;
                if (f.find("Side", 0) != string::npos) sideDetect(buf, (p + f).c_str());
                else detect(buf, (p + f).c_str(), AUTO);
                cout << buf << endl;
                out << buf << endl;
                clock_t end = clock();
                cout << "Time: " << end - start << "ms" << endl;
                out << "Time: " << end - start << "ms" << endl;
            }
        }
    }
    out.close();
}

int main() {
    int round = 0;
    while (true) {
        cout << round++ << endl;
        help();
    }
    system("pause");
    return 0;
}