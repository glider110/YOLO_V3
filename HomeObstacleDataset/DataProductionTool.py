from psd_tools import PSDImage
import numpy as np
import random
import os
import shutil
import glob
import tqdm
from PIL import Image, ImageDraw
import cv2


class Annotation:
    def __init__(self):
        self.num_scence = None    #背景图
        self.same_scence = 10   # 重复次数
        self.class_num=8       #生成几类
        self.path_annotation = "./chw.txt"  # 标注文件
        self.path_psd = "./python_paste/template"  # 模板
        self.path_background = './python_paste/background/'  # 背景
        self.path_savepicture = os.path.dirname(os.path.abspath(__file__)) + "/" + "composite_picture/"  # 合成图片数据
        self.annotation_list = []
        self.class_label = {
            "shoes": 0,
            "bowl": 1,
            "sock": 2,
            "fan": 3,
            "weighting": 4,
            "rag": 5,
            "feces": 6,
            "wire": 7,
        }

    # 自动化实现人造数据集及标注功能
    def auto_annotation(self):
        if os.path.exists(self.path_annotation):
            os.remove(self.path_annotation)
        with open(self.path_annotation, "a+") as new_f:

            if not os.path.exists(self.path_savepicture):
                os.mkdir(self.path_savepicture)
            else:
                shutil.rmtree(self.path_savepicture)
                os.mkdir(self.path_savepicture)
            for m in tqdm.tqdm(range(8)):  # 第一个循环  8类障碍物
                obstacle = m  # 主体障碍物
                list_background = (os.listdir(self.path_background)) * (self.same_scence)

                list_background = list_background[:self.num_scence]
                batch_picture_num = len(list_background)
                rang_list = len(list_background)
                print(rang_list)
                for i in range(rang_list):  # 第二个循环 背景照片数
                    list_background_psd = list_background[i]
                    list_grid = [2, 3, 4]
                    # list_grid = [3]
                    num_grid = random.sample(list_grid, 1)[0]
                    print("num_grid:", num_grid)
                    out_narry, psd_list = self.run_poster(list_background_psd, obstacle, num_grid)
                    out_narry = np.transpose(out_narry, (1, 0, 2))
                    im = Image.fromarray((out_narry))
                    im.save(self.path_savepicture + str(m * batch_picture_num + i) + ".jpg")
                    # im.show()
                    list_box_labe = ""
                    for ii in range(num_grid + 1):
                        list_box_labe += str(psd_list[ii]["xmin"]) + "," \
                                         + str(psd_list[ii]["ymin"]) + "," \
                                         + str(psd_list[ii]["xmax"]) + "," \
                                         + str(psd_list[ii]["ymax"]) + "," \
                                         + str(psd_list[ii]["label"]) + " "

                    annotation_one = self.path_savepicture + str(m * batch_picture_num + i) + ".jpg" + " " + list_box_labe
                    new_f.write("%s\n" % (annotation_one))

    # 粘贴
    def run_poster(self, list_background_psd, obstacle, num_grid):
        base_img = Image.open(self.path_background + list_background_psd)
        bb = np.array(base_img)  # pil转np数组是转置过的！！！！
        bb = np.transpose(bb, (1, 0, 2))
        size_b = bb.shape
        x_step = np.int(np.floor(size_b[0] / num_grid))
        y_step = np.int(np.floor(size_b[1] / num_grid))  # 狗东西 向下取整不就是整数吗 尼妈还要int 搞鸡毛
        # print("step:",x_step,y_step)
        # grid_x1 = [0, x_step, 2 * x_step]
        # grid_y1 = [0, y_step, 2 * y_step]
        grid_x = np.linspace(0, size_b[0], num_grid + 1)
        grid_y = np.linspace(0, size_b[1], num_grid + 1)
        # print("step1:",grid_x1,grid_y1)
        # print("step:",grid_x,grid_y)
        grid_x=[np.int(i) for i in grid_x]
        grid_y=[np.int(j) for j in grid_y]
        X, Y = np.meshgrid(grid_x[:-1], grid_y[:-1])
        # X, Y = np.meshgrid(grid_x1, grid_y1)
        xx = X.flatten()
        yy = Y.flatten()

        psd_list = []
        tmplate = self.get_template(obstacle, num_grid)
        rand_index_list = np.random.permutation(range(num_grid * num_grid))
        for k in range(num_grid+1):           #第三个循环按单张图片的框数
            dic_psd = {
                        "label": 1,
                        "xmin": 1,
                        "xmax": 1,
                        "ymin": 1,
                        "xmax": 1,
                      }

            psd = PSDImage.open(tmplate[k])
            name1 = ((tmplate[k]).split("/")[-1])
            name_class = name1[:name1.index("_")]
            # print(name_class)
            # return
            # get class label
            label = self.class_label[name_class]
            dic_psd["label"] = label
            psd_png = psd.composite()
            box_get = psd_png.getbbox()
            psd1 = psd_png.crop(box_get)
            # psd_rotate = psd1.rotate(360 , expand=True)
            if obstacle==3:
               psd_rotate = psd1.rotate(90 , expand=True)
            else:
               psd_rotate = psd1.rotate(360 * np.random.rand(), expand=True)
            psd_rotate_crop2 = psd_rotate.crop(psd_rotate.getbbox())

            ratio_guo = 1 - 0.7 * np.random.rand()
            # ratio_guo = 1
            # print("ratio:", ratio_guo)
            guo, x_min, x_max, y_min, y_max = self.letterbox_image(psd_rotate_crop2, [y_step, x_step])
            # print('-----------------')
            # print(x_min)
            # print(y_min)
            # print(x_max)
            # print(y_max)
            # print('-----------------')
            x_min *= ratio_guo
            x_max *= ratio_guo
            y_min *= ratio_guo
            y_max *= ratio_guo
            guo_scale = guo.resize((int(ratio_guo * y_step), int(ratio_guo * x_step)))
            aa = np.array(guo_scale)  # 横纵坐标变化
            size = aa.shape

            loction1 = [xx[rand_index_list[k]], yy[rand_index_list[k]]]
            # print("bb:",bb.shape)
            # print("template:",size[0],size[1])
            # print("location:",loction1[0],loction1[1])
            for i in range(size[0]):
                for j in range(size[1]):
                    if aa[i, j, -1] >= 10:
                        bb[i + loction1[0], j + loction1[1]] = aa[i, j, 0:-1]

            x_min += loction1[1]
            x_max += loction1[1]
            y_min += loction1[0]
            y_max += loction1[0]
            # print('------rrrr-----------')
            # print(x_min)
            # print(y_min)
            # print(x_max)
            # print(y_max)
            # print('-----------------')
            dic_psd["xmin"] = int(y_min)
            dic_psd["ymin"] = int(x_min)
            dic_psd["xmax"] = int(y_max)
            dic_psd["ymax"] = int(x_max)
            psd_list.append(dic_psd)

        return bb, psd_list

    # 读取模板psd
    def get_template(self, obstacle,num_grid):
        def show_files(path, all_files):
            for root, dirs, files in os.walk(path):  # key point
                for dirs_sub in dirs:
                    cur_path = os.path.join(path, dirs_sub)
                    # 判断是否是文件夹
                    fileList_main11 = glob.glob(cur_path + '/*.psd')
                    all_files += (fileList_main11)
                return all_files

        list_1 = []
        for root, dirs, files in os.walk(self.path_psd):
            list_1.append(root)
        fileList_sub = show_files(self.path_psd, [])
        fileList_main = glob.glob(list_1[obstacle + 1] + '/*.psd')
        np.random.shuffle(fileList_main)
        random.shuffle(fileList_sub)
        # template_path = [fileList_main[0], fileList_main[1], fileList_sub[0]]
        template_path = [i for i in fileList_main[:num_grid]]+[fileList_sub[0]]
        return template_path

    # 按原比例pading填充
    def letterbox_image(self,image, size):
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)  # BILINEAR
        new_image = Image.new('RGBA', size)
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        # new_image.paste(image, (0, 0))
        # image.show()
        # new_image.show()

        # print('gggg',iw,ih,w,h,nw,nh)
        x_min = (w - nw) / 2
        x_max = (w - nw) / 2 + nw
        y_min = (h - nh) / 2
        y_max = (h - nh) / 2 + nh
        # x_min = 0
        # x_max = nw
        # y_min = 0
        # y_max = nh
        # print('gggg',iw,ih)

        # a = ImageDraw.ImageDraw(new_image)  # 用a来表示
        #
        # # 在边界框的两点（左上角、右下角）画矩形，无填充，边框红色，边框像素为5
        # a.rectangle(((x_min, y_min), (x_max, y_max)), fill=None, outline='red', width=5)
        # new_image.save("22.png")

        return new_image, x_min, x_max, y_min, y_max

    # 测试标注效果
    def save_drawimage(self):
        num_img = 0
        image_path = "./composite_picture/" + str(num_img) + ".jpg"
        boxes = []
        im = cv2.imread(image_path)
        # print('HHH',np.shape(im))
        with open("./chw.txt") as label_file:
            list = label_file.readlines()[num_img].split()[1:]
            # print(list)
            for i in range(len(list)):
                boxes=list[i].split(',')
                cv2.rectangle(im, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (0, 0, 255), 2)
            cv2.imwrite('2.jpg', im)


if __name__ == "__main__":
    my_annotation = Annotation()
    my_annotation.auto_annotation()
    # my_annotation.save_drawimage()
