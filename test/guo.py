import numpy as np
import torch
import os
import re
import sys
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#or and
def test2():
    url = "http://c.biancheng.net/cplus/"
    print(print(url))
    print("----False and xxx-----")
    print( False and print(url) )
    print("----True and xxx-----")
    print( True and print(url) )
    print("----False or xxx-----")
    print( False or print(url) )
    print("----True or xxx-----")
    print( True or print(url) )
    a = 10
    b = 0
    c= a or b
    print(c)

def test3():
    a=52222
    str='guoxiaofan'
    # str[0]='k'
    # a[1]='3'
    print(str)
    list=[2,3,66,'guo','xiao','fan']
    print(list)
    list.append('glier')
    list[5]='bo'
    print(list[-2:-1])
    # list1=()
    # list1=(2,3,66,'guo','xiao','fan1')
    # # list1[5]='bo'
    # print(list1[-2])
    list2=[2,3,66]
    list3=range(20)
    list4=list3[-1:0:-2]
    print(list4)
    # list2[:2]=[]
    print(list2[::-1])          # []号里面看冒号的技巧
    for i in range(len(list4)):
       print(list4[i],end=" ")
     # print(range(100)[5:18:2])

def test1():
    num = 3
    if num == 3:
        print('boss')
    elif num == 4:
        print('ssf')
    else:
        print("fsfs")

def test4():
    class people:
        name = ''
        age = 0
        __weight = 0

        def __init__(self, n, a, w):
            self.name = n
            self.age = a
            self.__weight = w

        def __speak1(self):
            print("%s 说: 我 %d 岁。" % (self.name, self.age))

        def speak2(self):
            print("%s 说: 我 %d 岁。" % (self.name, self.age))
    p=people("guoxiaofan",10,30)
    p.speak2()

def test5():
    num = 1
    def fun1():
        # global num  # 需要使用 global 关键字声明
        # nonlocal num  # nonlocal关键字声明
        print(num)
        # num = 123
        # print(num)
    fun1()
    print(num)
    list = ['Google', 'Runoob', "Zhihu", "Taobao", "Wiki"]
    nums = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    print(list[1][1])
    nums.append(111)
    nums+=[2222]
    print(random.random())


def test6():
    def reverseWords(input):
        inputwords=input.split(" ")
        inputwords=inputwords[-1::-1]
        # 重新组合字符串
        output =" ".join(inputwords)
        return output
    input = 'guo xiao fan'
    rw = reverseWords(input)
    print(rw)

# 批量修改文件名
def test7():
    # 批量修改图片文件名
    def renameall():
        fileList = os.listdir("/home/guoxiaofan/Downloads/home_picture/data_web/shoes/shoes_photo")  # 待修改文件夹
        print("修改前：" + str(fileList))  # 输出文件夹中包含的文件
        currentpath = os.getcwd()  # 得到进程当前工作目录
        os.chdir("/home/guoxiaofan/Downloads/home_picture/data_web/shoes/shoes_photo")  # 将当前工作目录修改为待修改文件夹的位置
        num = 70 # 名称变量
        for fileName in fileList:  # 遍历文件夹中所有文件
            pat = "(jpg|png|gif|jpeg)"  # 匹配文件名正则表达式
            pattern = re.findall(pat, fileName)  # 进行匹配
            os.rename(fileName, (str(num) + '.' + pattern[0]))  # 文件重新命名
            num = num + 1  # 改变编号，继续下一项
        print("---------------------------------------------------")
        os.chdir(currentpath)  # 改回程序运行前的工作目录
        sys.stdin.flush()  # 刷新
        print("修改后：" + str(
        os.listdir("/home/guoxiaofan/Downloads/home_picture/data_web/shoes/shoes_photo")))  # 输出修改后文件夹中包含的文件

    renameall()

def test8():
    #基本数据类型 range arange array avaiber tensor
    # torch.Tensor(1, 2, 3)  # 生成一个 shape 为 [1,2,3] 的 tensor
    # torch.Tensor([1, 2, 3])  # 生成一个值为 [1,2,3] 的 tensor
    a=range(1,10,2)
    print(a[1])
    for i in a:
        print(i)
    print("###################")
    b=np.arange(1,10,0.3)
    print(b[1])
    for i in b:
        print(i)
    print("###################")
    c=np.array(a)
    for i in c:
        print(i)
    print(a)
    print(b)    #pay attention to  the output  format
    d=np.random.randn(10,2)   #torch.randint(0, 10, (2, 3)) torch.rand((2, 3))  torch.randn   torch.linspace(-1, 1, steps=21)
    print(d)
    print("###################")
    e=torch.Tensor(3,2,2)
    print(e)
    f=e.resize(2,6)
    #e=torch.tensor([[1,3],[2.6]])

    print(f)
    print(e.dim())
    print(e.size())
    print(e.numel())
    print("###################")
    x=torch.rand(5)
    x = Variable(x,requires_grad = True)
    print(x)
    y=2**x
    grads = torch.FloatTensor([1, 2, 3, 4, 5])
    y.backward(grads)  #
    print(y)
    print(x.grad)
#cnn
def test9():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1=nn.Conv2d(1,6,5)
            self.conv2=nn.Conv2d(6,16,5)
            self.fc1=nn.Linear(16*5*5,120)
            self.fc2=nn.Linear(120,84)
            self.fc3 = nn.Linear(84, 10)
        def forward(self,x):
            x=F.max_pool2d(F.relu(self.conv1(x)),2)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x=x.view(-1,self.num_flat_feature(x))  #X.view(-1)中的-1本意是根据另外一个数来自动调整维度，但是这里只有一个维度，因此就会将X里面的所有维度数据转化成一维的，并且按先后顺序排列。
            x=F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def num_flat_feature(self,x):
            size=x.size()[1:]
            num_feature=1
            for i in size:
                num_feature*=i
            return  num_feature
    net=Net()
    print(net)
    input=Variable(torch.rand(1,1,32,32))
    print(input)
    output1=net.forward(input) #  为啥不是调用内部函数forward
    # output=net(input) #这个地方就神奇了，明明没有定义__call__()函数啊，所以只能猜测是父类实现了，并且里面还调用了forward函数
    print(output1)
    optimizer=optim.SGD(net.parameters(),lr=0.01)
    optimizer.zero_grad()
    loss= nn.CrossEntropyLoss(output1,target)
    loss.backward()
    optimizer.step()
#研究生期间搞了两年的东西就是这几句代码 搞毛线啊
def test10():
   #所有数据多次迭代 每次计算loss 更新一次 而yolo是按照一个batch训练一次 循环次数为epoch 所有数据只用一次  err!!!
    w = 2
    b = 1
    noise = torch.rand(100, 1)
    x1 = torch.linspace(-1, 1, 100)
    x = torch.unsqueeze(x1, dim=1)    # a.squeeze(N) 就是在a中指定位置N加上一个维数为1的维度。
    print(x1)
    print(x.shape)
    # 因为输入层格式要为(-1, 1)，所以这里将(100)的格式转成(100, 1)
    y = w * x + b + noise
    # 拟合分布在y=2x+1上并且带有噪声的散点
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 1),
    )
    # 自定义的网络，带有2个全连接层和一个tanh层
    loss_fun = torch.nn.MSELoss()
    # 定义损失函数为均方差
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 使用adam作为优化器更新网络模型的权重，学习率为0.01

    plt.ion()
    # 图形交互
    for _ in range(1000):
        ax = plt.axes()
        output = model(x)
        # 数据向后传播（经过网络层的一次计算）
        loss = loss_fun(output, y)
        # 计算损失值
        # print("before zero_grad:{}".format(list(model.children())[0].weight.grad))
        # print("-"*100)
        model.zero_grad()
        # 优化器清空梯度
        # print("before zero_grad:{}".format(list(model.children())[0].weight.grad))
        # print("-"*100)
        # 通过注释地方可以对比发现执行zero_grad方法以后倒数梯度将会被清0
        # 如果不清空梯度的话，则会不断累加梯度，从而影响到当前梯度的计算
        loss.backward()
        # 向后传播，计算当前梯度，如果这步不执行，那么优化器更新时则会找不到梯度
        optimizer.step()
        # 优化器更新梯度参数，如果这步不执行，那么因为梯度没有发生改变，loss会一直计算最开始的那个梯度
        if _ % 20 == 0:
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), output.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)
            # print("w:", list(model.children())[0].weight.t() @ list(model.children())[-1].weight.t())
            # 通过这句可以查看权值变化，可以发现最后收敛到2附近

    plt.ioff()
    plt.show()
#线性拟合
def test11():
    x1=torch.linspace(-100, 100, 1000)
    x2=x1.view(1,1000)
    x = torch.unsqueeze(x1, dim=1)
    print(x.shape)
    # 生成-100到100的1000个数的等差数列
    y = 2 * x2 + 1
    # 定义y=2x+1函数
    matrix = torch.nn.Linear(1000, 1000)
    # 定义一个1x1的矩阵
    optimizer = torch.optim.Adam(matrix.parameters(), lr=0.1)
    # 使用优化器求导更新矩阵权重

    for _ in range(2):
        # 训练100次
        value = matrix(x2)
        # value是x与矩阵相乘后的值
        score = torch.mean((value - y) ** 2)
        # 目标偏差，值为(value-y)的平方取均值，越接近0说明结果越准确
        matrix.zero_grad()
        score.backward()
        optimizer.step()
        # 根据求导结果更新权值
        print("第{}次训练权值结果:{}，结果偏差：{}".format(_, matrix.weight.data.numpy(), score))

    # 输出结果：
    # 第0次训练权值结果:[[0.9555]]，结果偏差：4377.27294921875
    # ...
    # 第99次训练权值结果:[[2.0048]]，结果偏差：0.10316929966211319
#tensor
def test12():
    #对于多维的tensor 看数据是有技巧的 最多是四维tensor 先看最里面括号内的个数，再看最外面的个数
    a=torch.Tensor(1,2,3,3)
    a1 = torch.Tensor(2, 1, 3, 3)
    b=torch.tensor([[2,3],[1,4]])
    c=torch.tensor(2)
    d=torch.Tensor([2,3])
    # e=b.resize(2,-1)   #resize() can't use arg(-1)
    f=b.view(2,-1)
    g=b.reshape(2,-1)
    print(g)

#numpy
def test13():
    import random
    import numpy as np
    zero = np.zeros((100, 100, 3))
    grid_x=[0,2,4]
    grid_y = [0,2,4]
    print(grid_x)
    rand_index=random.randint(1,9)
    rand_index_list=np.random.permutation(range(9))
    print(rand_index_list)
    # X, Y = np.meshgrid(grid_x, grid_y)
    x = np.array([0, 1, 2])
    y = np.array([4, 5,7])
    X, Y = np.meshgrid(x,y)
    xx=X.flatten()
    yy=Y.flatten()
    print(X.flatten())
    print(Y.flatten())

    loction1=[xx[rand_index_list[0]],yy[rand_index_list[0]]]
    [print(loction1)]

def test14():
    # coding:utf-8
    '''
      python图片处理
      @author:fc_lamp
      @blog:http://fc-lamp.blog.163.com/
    '''
    # 等比例压缩图片
    def resizeImg(**args):
        args_key = {'ori_img': '', 'dst_img': '', 'dst_w': '', 'dst_h': '', 'save_q': 75}
        arg = {}
        for key in args_key:
            if key in args:
                arg[key] = args[key]
        im = image.open(arg['ori_img'])
        ori_w, ori_h = im.size
        widthRatio = heightRatio = None
        ratio = 1
        if (ori_w and ori_w > arg['dst_w']) or (ori_h and ori_h > arg['dst_h']):
            if arg['dst_w'] and ori_w > arg['dst_w']:
                widthRatio = float(arg['dst_w']) / ori_w  # 正确获取小数的方式
            if arg['dst_h'] and ori_h > arg['dst_h']:
                heightRatio = float(arg['dst_h']) / ori_h
            if widthRatio and heightRatio:
                if widthRatio < heightRatio:
                    ratio = widthRatio
                else:
                    ratio = heightRatio
            if widthRatio and not heightRatio:
                ratio = widthRatio
            if heightRatio and not widthRatio:
                ratio = heightRatio
            newWidth = int(ori_w * ratio)
            newHeight = int(ori_h * ratio)
        else:
            newWidth = ori_w
            newHeight = ori_h
        im.resize((newWidth, newHeight), image.ANTIALIAS).save(arg['dst_img'], quality=arg['save_q'])
        '''
        image.ANTIALIAS还有如下值：
        NEAREST: use nearest neighbour
        BILINEAR: linear interpolation in a 2x2 environment
        BICUBIC:cubic spline interpolation in a 4x4 environment
        ANTIALIAS:best down-sizing filter
        '''

    # 裁剪压缩图片
    def clipResizeImg(**args):
        args_key = {'ori_img': '', 'dst_img': '', 'dst_w': '', 'dst_h': '', 'save_q': 75}
        arg = {}
        for key in args_key:
            if key in args:
                arg[key] = args[key]
        im = image.open(arg['ori_img'])
        ori_w, ori_h = im.size
        dst_scale = float(arg['dst_h']) / arg['dst_w']  # 目标高宽比
        ori_scale = float(ori_h) / ori_w  # 原高宽比
        if ori_scale >= dst_scale:
            # 过高
            width = ori_w
            height = int(width * dst_scale)
            x = 0
            y = (ori_h - height) / 3
        else:
            # 过宽
            height = ori_h
            width = int(height * dst_scale)
            x = (ori_w - width) / 2
            y = 0
        # 裁剪
        box = (x, y, width + x, height + y)
        # 这里的参数可以这么认为：从某图的(x,y)坐标开始截，截到(width+x,height+y)坐标
        # 所包围的图像，crop方法与php中的imagecopy方法大为不一样
        newIm = im.crop(box)
        im = None
        # 压缩
        ratio = float(arg['dst_w']) / width
        newWidth = int(width * ratio)
        newHeight = int(height * ratio)
        newIm.resize((newWidth, newHeight), image.ANTIALIAS).save(arg['dst_img'], quality=arg['save_q'])

    # 水印(这里仅为图片水印)
    def waterMark(**args):
        args_key = {'ori_img': '', 'dst_img': '', 'mark_img': '', 'water_opt': ''}
        arg = {}
        for key in args_key:
            if key in args:
                arg[key] = args[key]
        im = image.open(arg['ori_img'])
        ori_w, ori_h = im.size
        mark_im = image.open(arg['mark_img'])
        mark_w, mark_h = mark_im.size
        option = {'leftup': (0, 0), 'rightup': (ori_w - mark_w, 0), 'leftlow': (0, ori_h - mark_h),
                  'rightlow': (ori_w - mark_w, ori_h - mark_h)
                  }
        im.paste(mark_im, option[arg['water_opt']], mark_im.convert('RGBA'))
        im.save(arg['dst_img'])

    # Demon
    # 源图片
    ori_img = './utils/background.jpg'
    # 水印标
    mark_img = './utils/example.png'
    # 水印位置(右下)
    water_opt = 'rightlow'
    # 目标图片
    dst_img = './utils/python_2.jpg'
    # 目标图片大小
    dst_w = 1000
    dst_h = 1500
    # 保存的图片质量
    save_q = 35
    # 裁剪压缩
    clipResizeImg(ori_img=ori_img, dst_img=dst_img, dst_w=dst_w, dst_h=dst_h, save_q=save_q)
    # 等比例压缩
    # resizeImg(ori_img=ori_img,dst_img=dst_img,dst_w=dst_w,dst_h=dst_h,save_q=save_q)
    # 水印
    # waterMark(ori_img=ori_img,dst_img=dst_img,mark_img=mark_img,water_opt=water_opt)

def test15():
    dic_psd["label"] = 9
    print("dfdf",dic_psd["label"])
    for i in tqdm.tqdm(range(100)):
        pass
    dict2={"guo":100,"xiao":33,"fan":"rw"}

    dict22={k:v for k,v in dict2.items()}
    dict1=[i for i in range(10) if i>3]
    print(dict1)
    dict22["glider"]=34
    print(dict22)
    print(dict22.values())

#file operate
def test16():
    with open("hello.txt") as a:
        # print(a.readlines()[0])
        b = (a.readlines()).__str__().strip()
        # b = a.readlines()[0:2][1]
        print(b)
        # print(type(b))

        num = 1
        list=[22,34]
        print(id(num))
        def fun1():
            # nonlocal num # 需要使用 global 关键字声明
            # print(list)
            # print(num)
            num = 123
            # print(id(num))
            # list = 4
            list=4
            # print(num)
            print(list)

        fun1()
        # print(num)
        print(list)
        # return a.readlines()

    str = 'a_d_v'

    b = str.split('_')
    print(os.path.abspath(__file__))
    print(__file__)
    print(dir("guo.py"))

    list_1=['fef',"sfa","eee",23]
    with open("guo.txt","w") as f:
        # f.writelines(list_1+"\n")
        number=random.randint(1,50)
        f.writelines(i.__str__()+"\n" for i in list_1)
        # f.write( list_1.__str__())

#基金持仓抓取 database
def test17():
    import pandas as pd
    from sqlalchemy import create_engine

    # 初始化数据库连接
    # 按实际情况依次填写MySQL的用户名、密码、IP地址、端口、数据库名
    engine = create_engine('mysql+pymysql://root:123456@localhost:3306/testdb')

if __name__=="__main__":
     test17()






