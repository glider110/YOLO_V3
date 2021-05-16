#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import numpy as np
import torch
from utils.config import Config
from nets.yolo3 import YoloBody
import random

#写入模型layer_name
def test1():
    model = YoloBody(Config)
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load("/mnt/HardDisk2T/2021_01_26-12_17/Epoch_5.pth", map_location=device)
    pretrained_dict_ = torch.load("model_data/yolo_weights.pth", map_location=device)

    # with open('voc.txt', 'w') as guo:
    #     for k, v in pretrained_dict.items():
    #         guo.write(k+'\n')
    #
    # with open('coco.txt', 'w') as coco:
    #     for k, v in pretrained_dict_.items():
    #         coco.write(k+'\n')
    #
    # with open('model.txt', 'w') as Mo:
    #     for k, v in model_dict.items():
    #         Mo.write(k+'\n')
    # print(type(pretrained_dict_.items()))
    # print(len(model_dict))
    # print(len(pretrained_dict))
    # print(len(pretrained_dict_))
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}

    # print(len(pretrained_dict))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

#数据变化的常用函数
def test2():
    a=[i for i in range(19)]
    print(a)

    list1 = [1, 3, 56, 7]
    list1 = np.array(list1)
    mask = [True, False, False, False]
    mask1=list1>4   #简便的掩码语法
    print(mask1)
    print(list1[mask])
    print(list1[list1>2])

    a = torch.tensor(np.array([[1, 2, 3]]))
    # b = a.squeeze()   #squeeze 函数：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    c=a.unsqueeze(2)
    print(c)

#reshape view resize
    d=random.randint(1,5)
    d=np.array([23])
    d=[1,3,4,5,5,6]
    d=np.array(d)
    e=d.reshape(2,3)   #reshape必须返回值才变化
    print(e)

#数据爬虫
def test3():
    # !/usr/bin/env python3
    # -*- coding: utf-8 -*-

    from sqlalchemy import Column, String, create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.ext.declarative import declarative_base

    # 创建对象的基类:
    Base = declarative_base()

    # 定义User对象:
    class User(Base):
        # 表的名字:
        __tablename__ = 'user'

        # 表的结构:
        id = Column(String(20), primary_key=True)
        name = Column(String(20))

    # 初始化数据库连接:
    engine = create_engine('mysql+mysqlconnector://sta:sta@localhost:3306/test')
    # 创建DBSession类型:
    DBSession = sessionmaker(bind=engine)

    # 创建session对象:
    session = DBSession()
    # 创建新User对象:
    new_user = User(id='5', name='Bob')
    # 添加到session:
    session.add(new_user)
    # 提交即保存到数据库:
    session.commit()
    # 关闭session:
    session.close()

    # 创建Session:
    session = DBSession()
    # 创建Query查询，filter是where条件，最后调用one()返回唯一行，如果调用all()则返回所有行:
    user = session.query(User).filter(User.id == '5').one()
    # 打印类型和对象的name属性:
    print('type:', type(user))
    print('name:', user.name)
    # 关闭Session:
    session.close()

#小语法
def test4():
    import random
    list1 = [ 2, 3, 4]
    # 先去重，再取出
    list2 = list(set(list1))
    print(list2)
    print(random.sample(list2, 1))

    aa=np.linspace(0,100,3)
    print(aa)

    for i in range(10):
         if i%2==0:print(i)



def test5():
    import cv2
    import sys
    import time

    dt = "2019-01-23 15:29:00"
    # 转换成时间数组
    timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
    # 转换成时间戳
    timestamp = time.mktime(timeArray)
    print(timeArray)
    print(timestamp)

    cap_1 = cv2.VideoCapture(1)
    cap_1.set(3, 1920)
    cap_1.set(4, 1080)
    # cap_2 = cv2.VideoCapture(2)
    # cap_3 = cv2.VideoCapture(3)
    # cap_4 = cv2.VideoCapture(4)

    write_ok = False

    sz = (int(cap_1.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(cap_1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = 30
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # fourcc = cv2.VideoWriter_fourcc(*'mpeg')

    vout_1 = cv2.VideoWriter()
    vout_1.open('./1/output.mp4', fourcc, fps, sz, True)
    # vout_2 = cv2.VideoWriter()
    # vout_2.open('./2/output.mp4',fourcc,fps,sz,True)
    # vout_3 = cv2.VideoWriter()
    # vout_3.open('./3/output.mp4',fourcc,fps,sz,True)

    cnt = 0
    while (True):
        if (write_ok):
            # print("video")
            # 获取当前时间
            time_now = int(time.time())
            # 转换成localtime
            # time_local = time.localtime(time_now)
            print(time_now)
            if time_now >= timestamp:
                while (cnt < 900):
                    cnt += 1
                    print(cnt)

                    ret_1, frame_1 = cap_1.read()
                    vout_1.write(frame_1)

                    # ret_2, frame_2 = cap_2.read()
                    # vout_2.write(frame_2)

                    # ret_3, frame_3 = cap_3.read()
                    # vout_3.write(frame_3)

                vout_1.release()
                # vout_2.release()
                # vout_3.release()
                sys.exit()
        else:
            print("stop")
            ret_1, frame_1 = cap_1.read()
            cv2.imshow("cam_1", frame_1)
            # ret_2, frame_2 = cap_2.read()
            # cv2.imshow("cam_2", frame_2)
            # ret_3, frame_3 = cap_3.read()
            # cv2.imshow("cam_3", frame_3)

        if cv2.waitKey(1) & 0xFF == ord("w"):
            write_ok = write_ok is not True


if __name__=="__main__":

   test5()

