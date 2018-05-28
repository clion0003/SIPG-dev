#ifndef _ANALYSIS_H
#define _ANALYSIS_H

#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <cctype>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <algorithm>

// company dataset save path
const std::string CompanyDataPath = "../../../company_id.txt";

// input: a list of "companyid + boxnum + boxtype" candidates from detection functions;
// output: return most likely result, which pass the check bit test, like "HJCU 1234567 22G1".
extern std::string getResult(std::vector<std::string>& a);

// input: first 4-char company id list;
// output: the most likely 4-char compant id, like: "HJCU"; 
extern std::string getCompany(std::vector<std::string>& a);

// input: 7-char boxnum list a, a list of candiates of 7-char boxnum;
// output: return the most likely 7-char boxnum, like "1234567".
extern std::string findSimilar(std::vector<std::string>& a, std::vector<std::string>& cad);

// input: last 4-char box type list;
// output: the most likely 4-char box type, like: "22G1";
extern std::string getType(std::vector<std::string>& a);

// input: 11-char std::string boxid contains 4-char company + 7-char boxnum;
// output: check if first 10-char successfully match last check bit.
extern int isVaild(std::string boxid);

// isVaild(std::string boxid) return value:
// result1: input length is not 11.
const int ERROR_INPUT_LENGTH = 1;
// result2: first 4 char of company id is not A~Z.
const int ERROR_COMPANY_ID = 2;
// result3: last 7 char of box id is not 0~9.
const int ERROR_BOX_ID = 3;
// result4: first 10 char does not match last check bit.
const int ERROR_CHECK_BIT = 4;
// result5: first 10 char successfully match last check bit.
const int VAILD_BOX_ID = 5;

// split std::string a with std::string delim.
extern std::vector<std::string> split(std::string a, std::string delim);

// input: a map of candidates of std::string and its counts;
// output: return the key std::string with largest value int.
extern std::string getMostNum(std::map<std::string, int>& a, std::string init = "");

#endif // _ANALYSIS_H
