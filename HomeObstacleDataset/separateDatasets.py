import os
import random
import xml.etree.ElementTree as ET
from os import getcwd

random.seed(0)

xmlfilepath = './chw.txt'
saveBasePath = r"./"

trainval_percent = 0.8
train_percent = 1

temp_ = open(xmlfilepath).readlines()

num = len(temp_)

list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)  # random.sample的用法，多用于截取列表的指定长度的随机数，但是不会改变列表本身的排序
train = random.sample(trainval, tr)

print("train and val size", tv)
print("train  size", tr)
ftrainval = open(os.path.join(saveBasePath, '2021_trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, '2021_test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, '2021_train.txt'), 'w')
fval = open(os.path.join(saveBasePath, '2021_val.txt'), 'w')

for i in list:  # 这个写法很技巧
    name = temp_[i]
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

