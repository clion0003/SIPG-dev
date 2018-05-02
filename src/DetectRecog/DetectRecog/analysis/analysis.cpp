#include "analysis.h"

using namespace std;

static map<char, int> help;

static void build() {
    char ch = '0';
    for (int i = 0; i <= 38; i++) {
        if (i&&i % 11 == 0) continue;
        help[ch] = i;
        if (ch == '9') {
            ch = 'A';
        }
        else {
            ch++;
        }
    }
}

int isVaild(string a) {
    if (a.length() != 11) return ERROR_INPUT_LENGTH;
    for (int i = 0; i<4; i++) {
        if (a[i]<'A' || a[i]>'Z') return ERROR_COMPANY_ID;
    }
    for (int i = 4; i<11; i++) {
        if (a[i]<'0' || a[i]>'9') return ERROR_BOX_ID;
    }
    if (!help.size()) build();
    int res = 0;
    for (int i = 0; i<10; i++) {
        int cur = help[a[i]];
        res += (cur << i);
    }
    res = (res % 11) % 10;
    if (res == (a[10] - '0')) return VAILD_BOX_ID;
    else return ERROR_CHECK_BIT;
}

vector<string> split(string s, string delim) {
    vector<string> ret;
    size_t last = 0;
    size_t index = s.find_first_of(delim, last);
    while (index != string::npos)
    {
        ret.push_back(s.substr(last, index - last));
        last = index + 1;
        index = s.find_first_of(delim, last);
    }
    if (index - last > 0)
    {
        ret.push_back(s.substr(last, index - last));
    }
    return ret;
}

static vector<map<char, int>> getNum(vector<string>& a) {
    vector<map<char, int>> dp(7, map<char, int>());
    for (auto it : a) {
        int len = it.length();
        if (len<4 || len>8) continue;
        if (it == "???????") continue;
        int n1 = 0;
        int n2 = 0;
        for (auto& ch : it) {
            if (ch == 'G') ch = '6';
            else if (ch == 'Z') ch = '2';
            else if (ch == 'O') ch = '0';
            else if (ch == 'B') ch = '8';
            else if (ch == 'K') ch = '7';
            else if (ch == 'I') ch = '1';
            else if (ch == 'S') ch = '5';
            else if (ch == '?') n1++;
            else if (ch<'0' || ch>'9') n2++;
        }
        if (n1>2 || n2>2) continue;
        string first = it.substr(0, 7);
        first.resize(7, ' ');
        for (int i = 0; i<7; i++) {
            char ch = first[i];
            if (ch >= '0'&&ch <= '9')dp[i][ch]++;
            else if (ch == '?') {
                for (char j = '0'; j <= '9'; j++) {
                    dp[i][j]++;
                }
            }
        }
        reverse(it.begin(), it.end());
        first = it.substr(0, 7);
        first.resize(7, ' ');
        for (int i = 6; i >= 0; i--) {
            char ch = first[6 - i];
            if (ch >= '0'&&ch <= '9')dp[i][ch]++;
            else if (ch == '?') {
                for (char j = '0'; j <= '9'; j++) {
                    dp[i][j]++;
                }
            }
        }
    }
    return dp;
}

static void dfs(vector<map<char, int>>& dict, vector<string>& res, string& dp, string& company) {
    int len = dp.length();
    if (len == 7) {
        string tmpnum = company + dp;
        if (isVaild(tmpnum) == VAILD_BOX_ID) {
            res.push_back(dp);
        }
        return;
    }
    for (auto it : dict[len]) {
        dp.push_back(it.first);
        dfs(dict, res, dp, company);
        dp.pop_back();
    }
}

static string lastchance(vector<string>& a, string company) {
    auto dict = getNum(a);
    string dp;
    vector<string> res;
    dfs(dict, res, dp, company);
    if (res.size()) return findSimilar(a, res);
    else return "???????";
}


string findSimilar(vector<string>& a, vector<string>& cad) {
    auto dict = getNum(a);
    map<string, int> dp;
    for (auto it : cad) {
        int res = 0;
        for (int i = 0; i<7; i++) {
            res += dict[i][it[i]];
        }
        dp[it] = res;
    }
    return getMostNum(dp);
}


static set<string> companyDict;

static void loadCompanyData() {
    fstream in(CompanyDataPath);
    string tmp;
    while (in >> tmp) {
        companyDict.insert(tmp);
    }
}

string getCompany(vector<string>& a) {
    if (!companyDict.size()) loadCompanyData();
    map<string, int> dp;
    for (auto it : a) {
        if (it.length()<3 || it.length()>5) continue;
        if (it == "????") continue;
        for (auto& ch : it) {
            if (ch == '2') {
                ch = 'Z';
            }
            else if (ch == '1') {
                ch = 'I';
            }
            else if (ch == '7') {
                ch = 'K';
            }
            else if (ch == '6') {
                ch = 'G';
            }
            else if (ch == '0') {
                ch = 'O';
            }
            else if (ch == '8') {
                ch = 'B';
            }
            else if (ch == '5') {
                ch = 'S';
            }
        }
        if (companyDict.find(it) != companyDict.end()) {
            dp[it]++;
        }
    }
    return getMostNum(dp, "????");
}


const set<string> typeDict = { "22G1","22U1","42G1","42U1","45G1","45U1" };

string getType(vector<string>& a) {
    map<string, int> dp;
    for (auto it : a) {
        if (it.length()<3 || it.length()>5) continue;
        if (it == "????") continue;
        for (char& ch : it) {
            if (ch == 'z' || ch == 'Z') {
                ch = '2';
            }
            else if (ch == '6' || ch == 'C') {
                ch = 'G';
            }
            else if (ch == 'u' || ch == 'V' || ch == 'O' || ch == '0' || ch == 'o') {
                ch = 'U';
            }
        }
        if (typeDict.find(it) != typeDict.end()) {
            dp[it]++;
        }
        else if (it.find("5G") != string::npos) {
            dp["45G1"]++;
        }
        else if (it.find("5U") != string::npos) {
            dp["45U1"]++;
        }
        else if (it.find("42G") != string::npos) {
            dp["42G1"]++;
        }
        else if (it.find("42U") != string::npos) {
            dp["42U1"]++;
        }
        else if (it.find("22G") != string::npos) {
            dp["22G1"]++;
        }
        else if (it.find("22U") != string::npos) {
            dp["22U1"]++;
        }
    }
    return getMostNum(dp, "22G1");
}

string getMostNum(map<string, int>& a, string init) {
    string res = init;
    int num = 0;
    for (auto it : a) {
        if (it.second>num) {
            num = it.second;
            res = it.first;
        }
    }
    return res;
}

string getResult(vector<string>& a) {
    map<string, int> dp;
    vector<string> typelist;
    vector<string> companylist;
    vector<string> numlist;
    for (auto it : a) {
        auto tmp = split(it, " ");
        tmp.resize(3, "");
        string tmpnum = tmp[0] + tmp[1];
        if (isVaild(tmpnum) == VAILD_BOX_ID) {
            tmpnum = tmp[0] + " " + tmp[1];
            dp[tmpnum]++;
        }
        companylist.push_back(tmp[0]);
        numlist.push_back(tmp[1]);
        typelist.push_back(tmp[2]);
    }
    string type = getType(typelist);
    string company = getCompany(companylist);
    if (dp.size()) {
        string res = getMostNum(dp);
        vector<string> cad;
        for (auto it : dp) {
            if (it.second == dp[res]) {
                cad.push_back(it.first);
            }
        }
        if (cad.size() == 1) {
            res += " " + type;
            return res;
        }
        else {
            vector<string> tmp;
            for (auto it : cad) {
                if (it.substr(0, 4) == company) {
                    tmp.push_back(it.substr(5, 12));
                }
            }
            res = company + " ";
            res += findSimilar(numlist, tmp);
            res += " " + type;
            return res;
        }
    }
    else {
        for (auto it : a) {
            auto tmp = split(it, " ");
            tmp.resize(3, "");
            string tmpnum = company + tmp[1];
            if (isVaild(tmpnum) == VAILD_BOX_ID) {
                tmpnum = company + " " + tmp[1];
                dp[tmpnum]++;
            }
        }
        if (dp.size()) {
            string res = getMostNum(dp);
            return res + " " + type;
        }
    }
    string res = lastchance(numlist, company);
    return company + " " + res + " " + type;
}
