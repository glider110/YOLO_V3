import matplotlib.pyplot as ply
import numpy as np
with open('/mnt/HardDisk2T/2021_01_28-08_59/2021_01_28-08_59_loss.txt', 'r') as loss_file:
    file_list = loss_file.readlines()
all=[]
for i in range(len(file_list)):
    all.append(file_list[i].split())
all = np.array(all)
epoch = all[:,0].astype(int)
loss_per_batch = all[:,1].astype(float)
val_loss_per_batch = all[:,2].astype(float)

ply.plot(epoch,loss_per_batch,color='r',label = 'train_loss')
ply.plot(epoch,val_loss_per_batch,color='b',label = 'val_loss')
ply.legend(loc = 0)
ply.xlabel('Epoch')
ply.ylabel('Loss')
ply.title('Loss per batch')
ply.show()