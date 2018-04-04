# ACCRS
An Automatic Container-code Recognition System based on MSER, CTPN, EAST and CRNN.

### 说明
* AlexNet, CRNN, CTPN, deeplab, EAST均为神经网络服务，使用时用python运行对应文件夹底下的demo_server.py即可
* DetectRecog, 使用神经网络提供的服务，进行箱面分割，字符检测，过滤，识别等任务，是一个vs工程。编译生成一个动态链接库。
* conNumData.txt, 目前已知的公司号的数据库，DetectRecog中会将神经网络字符识别的结果中的公司号与conNumData进行相似字符串匹配。
* matchData.txt，目前已知的箱型的数据库，DetectRecog中会将神经网络字符识别的结果中的箱型与matchData.txt进行相似字符串匹配。
* specialCase.ini，经验总结出来的神经网络对于箱型的错检结果及其对应的正确结果。如456:45g1表示如果神经网络检测的结果为456，则真实结果应为45G1。