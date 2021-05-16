#----------------------------------------------------#
#   获取测试集的ground-truth
#   具体视频教程可查看
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
import sys
import os
import glob
import xml.etree.ElementTree as ET

class_label = {
    "shoes": 0,
    "bowl": 1,
    "sock": 2,
    "fan": 3,
    "weighting":4,
    "rag":5,
    "feces":6,
    "wire":7,
}

# pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
class_label_new=dict([val,key]for key,val in class_label.items())

if not os.path.exists("./input_obs"):
    os.makedirs("./input_obs")
if not os.path.exists("./input_obs/ground-truth"):
    os.makedirs("./input_obs/ground-truth")

image_ids = open('../HomeObstacleDataset/2021_test.txt').read().strip().split()
image_ids_annotation = open('../HomeObstacleDataset/2021_test.txt').readlines()
image_ids_strip=[i.strip() for i in image_ids_annotation]



#
for image_id in (image_ids_strip):
    image_path = image_id.split()  #去掉空格
    image_id=image_path[0].split("/")[-1].split(".")[0]
    box_all=image_path[1:]
    with open("./input_obs/ground-truth/" + image_id + ".txt", "w") as new_f:
       for i in range(len(box_all)):
            image_box=box_all[i][:-1]
            image_box=image_box.split(",")
            left = image_box[0]
            top = image_box[1]
            right = image_box[2]
            bottom = image_box[3]
            print(type(left))
            class_box=box_all[i][-1]
            print(type(class_box))
            new_f.write("%s %s %s %s %s \n" %(class_label_new[int(class_box)],left, top, right, bottom))
#
print("Conveddrsion completed!")
