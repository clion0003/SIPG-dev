
# coding: utf-8

# In[1]:


import classifier
import os
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
    
if __name__ == '__main__':
    demo()

