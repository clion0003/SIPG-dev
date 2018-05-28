#include "detect\detect.h"
#include "side_detect\side_detect.h"
#include "socketcpp\tcpClient.h"
#include "analysis\analysis.h"
#include "detect\utils.h"
#include "gate_detect\gateDetect.h"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>
#include <ctime>
#include <fstream>
using namespace std;
int myrandom(int i) { return std::rand() % i; }
void test_qianhou1() {
    string fpath = "G:\\qianhou1\\front\\";
    string rpath = "G:\\qianhou1\\rear\\";
    vector<string> fname = getDirFileNames(fpath);
    vector<string> rname = getDirFileNames(rpath);
    map<string, string> fmap;
    cout << "front size: " << fname.size() << endl;
    cout << "rear size: " << rname.size() << endl;
    for (auto it : fname) {
        auto ed = it.find_first_of("FR", 0);
        string tmp = it.substr(0, ed);
        fmap[tmp] = it;
    }
    vector<vector<string>> vpss;
    for (auto it : rname) {
        auto ed = it.find_first_of("FR", 0);
        string tmp = it.substr(0, ed);
        if (fmap.find(tmp) != fmap.end()) {
            vpss.push_back({ fpath+fmap[tmp], rpath+it });
        }
    }
    int index = 0;
    fstream out("C:\\Users\\archlab\\Desktop\\out4.txt");
    if (out) cout << "success" << endl;
    for (auto it : vpss) {

        index++;
        cout << index << endl;
        out << index << endl;
        bool flag = true;
        vector<string> dfiles;
        getAllFiles("C:\\Users\\archlab\\Desktop\\showImg\\", dfiles);

        for (auto f : dfiles) {
            DeleteFile(f.c_str());
        }
        vector<string> cadlist;
        for (auto t : it) {
            clock_t start = clock();
            char buf[100];
            detect(buf, t.c_str(), flag ? FCOL_MODE : FROW_MODE);
            
            flag = false;
            cout << buf << endl;
            out << t << endl;
            clock_t end = clock();
            cout << "Time: " << end - start << "ms" << endl;
            cadlist.push_back(string(buf));
            out << buf << endl;
        }
        string res = getResult(cadlist);
        cout << "final:" << res << endl;
        out << "final:" << endl;
        out << res << endl;
        //system("pause");
       
    }
    cout << "vpss size: " << vpss.size() << endl;
}


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
            vector<string> dfiles;
            getAllFiles("C:\\Users\\archlab\\Desktop\\showImg\\", dfiles);

            for (auto f : dfiles) {
                DeleteFile(f.c_str());
            }
            vector<string> cadlist;
            cout << "--------------"<<it<<"-------------" << endl;
            out << it << endl;
            int num = stoi(it);
            //if (num != 126 && num != 113) continue;
            string p = path + it + "\\";
            cout << "path: " << p << endl;
            vector<string> fnames = getDirFileNames(p);
            for (auto f : fnames) {
                clock_t start = clock();
                char buf[100];
                cout << "==>"<<f << endl;
                if (f.find("Side", 0) != string::npos) sideDetect(buf, (p + f).c_str());
                else detect(buf, (p + f).c_str(), AUTO);
                cout << "  Detection Result: "<<buf << endl;
                //out << "  "<<buf << endl;
                clock_t end = clock();
                cout << "  Time: " << end - start << "ms" << endl;
                //out << "  Time: " << end - start << "ms" << endl;
                cadlist.push_back(string(buf));
            }
            cout << "-----------final result of "<<it<<":----------" << endl;
            string res = getResult(cadlist);
            cout << "  Final Result: "<<res << endl;
            out << res << endl;
            system("pause");
        }
        
    }
    out.close();
}

void simple() {
    string fpath = "G:\\qianhou1\\front\\";
    string rpath = "G:\\qianhou1\\rear\\";
    vector<string> fname = getDirFileNames(fpath);
    vector<string> rname = getDirFileNames(rpath);
    map<string, string> fmap;
    cout << "front size: " << fname.size() << endl;
    cout << "rear size: " << rname.size() << endl;
    for (auto it : fname) {
        auto ed = it.find_first_of("FR", 0);
        string tmp = it.substr(0, ed);
        fmap[tmp] = it;
    }
    vector<vector<string>> vpss;
    for (auto it : rname) {
        auto ed = it.find_first_of("FR", 0);
        string tmp = it.substr(0, ed);
        if (fmap.find(tmp) != fmap.end()) {
            vpss.push_back({ fpath + fmap[tmp], rpath + it });
        }
    }
    int index = 0;
    for (auto it : vpss) {
        index++;
        bool flag = true;
        vector<string> dfiles;
        getAllFiles("C:\\Users\\archlab\\Desktop\\showImg\\", dfiles);

        for (auto f : dfiles) {
            DeleteFile(f.c_str());
        }
        vector<string> cadlist;
        for (auto t : it) {
            clock_t start = clock();
            char buf[100];
            detect(buf, t.c_str(), flag ? FCOL_MODE : FROW_MODE);
            flag = false;
            cout << buf << endl;
            clock_t end = clock();
            cout << "Time: " << end - start << "ms" << endl;
            cadlist.push_back(string(buf));
            
        }
        string res = getResult(cadlist);
        cout << "final:" << res << endl;

    }
    cout << "vpss size: " << vpss.size() << endl;
}


void changename() {
    string fpath = "G:\\qianhou1\\front\\";
    string rpath = "G:\\qianhou1\\rear\\";
    vector<string> fname = getDirFileNames(fpath);
    vector<string> rname = getDirFileNames(rpath);
    map<string, string> fmap;
    cout << "front size: " << fname.size() << endl;
    cout << "rear size: " << rname.size() << endl;
    for (auto it : fname) {
        auto ed = it.find_first_of("FR", 0);
        string tmp = it.substr(0, ed);
        fmap[tmp] = it;
    }
    vector<vector<string>> vpss;
    for (auto it : rname) {
        auto ed = it.find_first_of("FR", 0);
        string tmp = it.substr(0, ed);
        if (fmap.find(tmp) != fmap.end()) {
            vpss.push_back({ fpath + fmap[tmp], rpath + it });
        }
    }
    int index = 0;
    for (auto it : vpss) {
        index++;
        bool flag = true;
        for (auto t : it) {
            cv::Mat img = cv::imread(t);
            string name = to_string(index) + (flag ? "_f" : "_r");
            saveTmpImage(img, name, "G:\\qianhou1\\database\\");
            flag = false;
        }
    }
    cout << "vpss size: " << vpss.size() << endl;
}

void changename2() {
    string fpath = "G:\\qianhou2\\front\\";
    string rpath = "G:\\qianhou2\\rear\\";
    vector<string> fname = getDirFileNames(fpath);
    vector<string> rname = getDirFileNames(rpath);
    map<string, string> fmap;
    cout << "front size: " << fname.size() << endl;
    cout << "rear size: " << rname.size() << endl;
    for (auto it : fname) {
        string tmp = it.substr(10, 1);
        tmp += it.substr(22, 4);
        fmap[tmp] = it;
    }
    vector<vector<string>> vpss;
    for (auto it : rname) {
        string tmp = it.substr(10, 1);
        tmp += it.substr(22, 4);
        if (fmap.find(tmp) != fmap.end()) {
            vpss.push_back({ fpath + fmap[tmp], rpath + it });
        }
    }
    int index = 0;
    for (auto it : vpss) {
        index++;
        bool flag = true;
        for (auto t : it) {
            cv::Mat img = cv::imread(t);
            string name = to_string(index) + (flag ? "_f" : "_r");
            saveTmpImage(img, name, "G:\\qianhou2\\database\\");
            flag = false;
        }
    }
    cout << "vpss size: " << vpss.size() << endl;
}

void test_qianhou2() {
    string fpath = "G:\\qianhou2\\front\\";
    string rpath = "G:\\qianhou2\\rear\\";
    vector<string> fname = getDirFileNames(fpath);
    vector<string> rname = getDirFileNames(rpath);
    map<string, string> fmap;
    cout << "front size: " << fname.size() << endl;
    cout << "rear size: " << rname.size() << endl;
    for (auto it : fname) {
        string tmp = it.substr(10, 1);
        tmp += it.substr(22, 4);
        fmap[tmp] = it;
    }
    vector<vector<string>> vpss;
    for (auto it : rname) {
        string tmp = it.substr(10, 1);
        tmp += it.substr(22, 4);
        if (fmap.find(tmp) != fmap.end()) {
            vpss.push_back({ fpath + fmap[tmp], rpath + it });
        }
    }
    int index = 0;
    fstream out("C:\\Users\\archlab\\Desktop\\out4.txt");
    if (out) cout << "success" << endl;
    for (auto it : vpss) {

        index++;
        cout << index << endl;
        out << index << endl;
        bool flag = true;
        vector<string> dfiles;
        getAllFiles("C:\\Users\\archlab\\Desktop\\showImg\\", dfiles);

        for (auto f : dfiles) {
            DeleteFile(f.c_str());
        }
        vector<string> cadlist;
        for (auto t : it) {
            clock_t start = clock();
            char buf[100];
            detect(buf, t.c_str(), flag ? FCOL_MODE : FROW_MODE);
            flag = false;
            cout << buf << endl;
            out << t << endl;
            clock_t end = clock();
            cout << "Time: " << end - start << "ms" << endl;
            cadlist.push_back(string(buf));
            out << buf << endl;
            system("pause");
        }
        string res = getResult(cadlist);
        cout << "final:" << res << endl;
        out << "final:" << endl;
        out << res << endl;
        system("pause");

    }
    cout << "vpss size: " << vpss.size() << endl;
}
/*
string gateDetect(string path);
string gateDetect();

void ttt() {
    string path = "G:\\qianhou2\\database\\";
    //path = "C:\\Users\\archlab\\Desktop\\showImg\\s2\\";
    vector<string> flist = getDirFileNames(path);
    random_shuffle(flist.begin(), flist.end(), myrandom);
    int index = 0;
    for (auto f : flist) {
        cout << "index:"<<index++ << endl;
        string name = path + f;
        if (f.find("_r") != -1) continue;
        cout << name << endl;
        gateDetect(name);
    }
}
void ttt1() {
    string path = "G:\\qianhou2\\database\\";
    //path = "C:\\Users\\archlab\\Desktop\\showImg\\s2\\";
    for (int i = 300; i <= 935; i++) {
        string name = path + to_string(i) + "_f.jpg";
        cout << name << endl;
        gateDetect(name);
    }
}
void special() {
    fstream in("C:\\Users\\archlab\\Desktop\\in1.txt");
    string path = "G:\\qianhou2\\database\\";
    string tmp;
    while (in >> tmp) {
        string name = path + tmp + "_f.jpg";
        cout << name << endl;
        gateDetect(name);
    }

}
*/
void gateTest() {
    string path = "G:\\qianhou2\\database\\";
    //path = "C:\\Users\\archlab\\Desktop\\showImg\\s2\\";
    for (int i = 1; i <= 935; i++) {
        string name = path + to_string(i) + "_f.jpg";
        cout << i << " " << name << endl;
        gateDetect detector(name);
        detector.detect();
    }
}
int main() {
    //test();
    //help();
    //changename2();
    //simple();
    //changename();
     //test_qianhou2();
    //gateDetect();
    //ttt1();
    gateTest();
    //special();
    system("pause");
    return 0;
}
