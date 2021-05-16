#-------------------------------------#
#       mAP所需文件计算代码
#       具体教程请查看Bilibili
#       Bubbliiiing
#-------------------------------------#
import numpy as np
import os
import torch

from PIL import Image
from utils.utils import non_max_suppression, letterbox_image,yolo_correct_boxes
from tqdm import tqdm

# import sys
# sys.path.append("..")
from predict.yolo import YOLO

class mAP_Yolo(YOLO):       #继承yolo 重写detect_image
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self,image_id,image):
        self.confidence = 0.8
        self.iou = 0.5
        f = open("./input_obs/detection-results/"+image_id+".txt","w")
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
        photo = np.array(crop_img,dtype = np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)

        images = np.asarray(images)
        images = torch.from_numpy(images)
        if self.cuda:
            images = images.cuda()
        
        with torch.no_grad():
            outputs = self.net(images)
            output_list = []
            for i in range(3):
                output_list.append(self.yolo_decodes[i](outputs[i]))
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(output, self.config["yolo"]["classes"],
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)
        try :
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image
        top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence
        top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
        top_label = np.array(batch_detections[top_index,-1],np.int32)
        top_bboxes = np.array(batch_detections[top_index,:4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

        # 去掉灰条
        boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = str(top_conf[i])

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

yolo = mAP_Yolo()


if not os.path.exists("./input_obs"):
    os.makedirs("./input_obs")
if not os.path.exists("./input_obs/detection-results"):
    os.makedirs("./input_obs/detection-results")
if not os.path.exists("./input_obs/images-optional"):
    os.makedirs("./input_obs/images-optional")



# image_ids = open('../HomeObstacleDataset/2021_test.txt').read().strip().split()
image_ids_annotation = open('../HomeObstacleDataset/2021_test.txt').readlines()
image_ids_strip=[i.strip() for i in image_ids_annotation]


for image_id in tqdm(image_ids_strip):

    image_path = image_id.split()
    image = Image.open(image_path[0])
    # 开启后在之后计算mAP可以可视化
    # image.save("./input/images-optional/"+image_id+".jpg")
    image_id=image_path[0].split("/")[-1].split(".")[0]
    yolo.detect_image(image_id,image)
    
print("Conversion completed!")
