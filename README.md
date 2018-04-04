## 文件夹说明
**./DetectLib**  
VS工程生成的给上港去使用的库，包括提供库函数声明的detect.h文件，及其对应的.dll和.lib文件  
**./doc**  
项目相关文档  
**./model_para**  
项目中用到的神经网络的模型参数，底下有每个文件夹对应一个网络，目前有AlexNet，CRNN，CTPN，deeplab，EAST五个网络。由于这些模型参数比较大，所以此处只提供[百度云的链接](https://pan.baidu.com/s/1BGXgUZXmQ7MxESlyr4I2Dg)。使用时解压压缩包到此处即可。  
**./pycaffe**  
目前项目中deeplab, alexnet, CTPN是基于caffe的，其编译好的pycaffe放在此文件夹下，目前有CaffeDeepLab(deeplab和alexnet使用)，CaffeCTPN两个文件夹。由于这两个pycaffe包比较大，所以此处只提供[百度云的链接](https://pan.baidu.com/s/1a6KW0XBa8TXdD-BfqLpPkw)。使用时解压压缩包到此处即可。  
**./src**  
项目源码，其中DetectRecog是一个vs工程，利用神经网络提供的服务，进行箱号识别的任务，最终生成一个库，供上港调用。剩下5个文件夹均为封装了神经网络的server  
**./thirdparty**  
项目中用到的第三方库。主要是opencv, boost, rapidjson等库。这些库文件的[百度云的链接](https://pan.baidu.com/s/1_pg2-mjBb0L0TVjyXmOh-w)。使用时解压压缩包到此处即可。   
  
## 使用说明
### 环境安装
* 需要的环境包的[百度云链接](https://pan.baidu.com/s/1Rs6gaKwhYIDHZ_LNf2dadw)  
* 主要的开发环境是vs2015(DetectRecog) + python3.6(5个神经网络server) + CUDA9.0 + cuDNNv7，CUDA和cudnn安装包在上述百度云链接中。  
* 其中AlexNet, CTPN, deeplab使用的均为caffe，其需import的caffe均在pycaffe底下。  
* CRNN基于pytorch，其安装包在上述百度云链接中，使用conda安装。  
* EAST是基于tensorflow的，使用pip安装tf-nightly-gpu即可，版本应>1.5。  
  
  
### 使用流程
* 启用神经网络server：使用python分别运行./src底下每个神经网络server文件夹下面的demo_server.py即可
* 启用DetectRecog：用vs打开./DetectRecog/DetectRecog.sln