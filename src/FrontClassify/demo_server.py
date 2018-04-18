
# coding: utf-8

# In[1]:


import classifier
import os
import json
from socket import *
import sys
def front_classifier_server(gpu_memory_size):
    HOST = ''
    PORT = 6006
    BUFSIZ = 1024
    ADDR = (HOST, PORT)
    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind(ADDR)
    sock.listen(5)
    while True:
        print('waiting for connection')
        tcpClientSock, addr = sock.accept()
        print('connect from ', addr)
        try:
            data=tcpClientSock.recv(BUFSIZ)
        except:
            print("error")
            tcpClientSock.close()
            break
        if not data:
            print('not data')
            err_response={"error_code":1}
            response_str=json.dumps(err_response)+'\0'
            tcpClientSock.send(response_str.encode('utf8'))
            break
        imgsavepath=data.decode('utf8')

        imgpaths = [os.path.join(imgsavepath, f) for f in os.listdir(imgsavepath)]
        print (imgpaths)
        results = classifier.predict(imgpaths, gpu_memory_size)
        response_json = {}

        response_json={ str(index) : int(item) for index, item in enumerate(results)}

        response_json['strnum'] = len(results)
        response_json["error_code"] = 0
        print (response_json)
        response_str=json.dumps(response_json)+'\0'
        tcpClientSock.send(response_str.encode('utf8'))
        #print([ctime()], ':', imgpath)
        tcpClientSock.close()
        print ('finished ,succeed send back')
    sock.close()

if __name__ == '__main__':
    if len(sys.argv) == 2:
        print('server started!')
        #init()
        front_classifier_server(int(sys.argv[1]))
    else:
        print('Usage: python demo_server.py [gpu_memory_size]')
        print('#gpu_memory_size: 2/4/6/8/12, presents memory size(GB) of your GPU card.')
        print('#example: python demo_server.py 6')

def demo():
    '''测试demo，演示了一次预测test_img文件路径下的所有图片的正反面'''
    # 测试数据来源于test_img文件夹的图片。
    cwd = os.getcwd()
    test_path = os.path.join(cwd, 'test_img')
    # 讲测试图片放到list里面生成输入。
    test_img = [os.path.join(test_path, f) for f in os.listdir(test_path)]
    
    print('---------------------Input image file path----------------')
    print(test_img)
    # classifier 有两种输入方式，一种是一次预测一张图片的正反面，此时直接输入测试图片的路径字符串，直接返回预测结果；
    # 另一种是一次预测一组图片的正反面，此时输入一个list的图片路径，返回对应的一个list的预测结果。
    # 当需要同时预测多张图片时，推荐使用输入list的方式提高执行效率。
    result = classifier.predict(test_img)
    
    # 结果中的1表示有栏杆，即反面；0表示没有栏杆，即正面。
    print('-----------------------Final Result----------------------')
    print('# 1 presents the back and 0 presents the front.')
    print(result)


