#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
import sys
sys.path.append("../")
from yolo import YOLO
from PIL import Image

yolo = YOLO()

# while True:
# img = input('Input image filename:')
img = '/home/sta/Pictures/shoes5.png'
# img = 'HomeObstacleDataset/20210105/shite/shite/Tue Jan 05 16-58-20.bmp'
# img = '/home/sta/Desktop/guo/yolo3-pytorch-master/HomeObstacleDataset/composite_picture/205.jpg'
try:
    image = Image.open(img)
except:
    print('Open Error! Try again!')
    # continue
else:
    r_image = yolo.detect_image(image)
    r_image.show()
