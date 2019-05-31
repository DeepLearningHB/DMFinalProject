import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
path = 'HK_dataset_align_labels_8.txt'
ver_path = './Verify'
f = open(path, 'r')
lines = f.readlines()

for line in lines:
    line = line.replace('\n', '')
    line = line.split(' ')
    image_name = line[0]
    # print(line[1])
    coord = list(map(float, line[1:]))
    # print(coord)
    image = Image.open(os.path.join('./AlignImages_added_8', image_name))
    image = image.resize((128, 128))
    coord = np.reshape(np.array(coord), (2, 4))
    #print(coord)
    #exit()
    image = np.array(image)
    ver_lab = np.array(coord*128).astype(np.int)
    print(ver_lab)
    image[ver_lab[0][1]:ver_lab[0][1]+2, ver_lab[0][0]:ver_lab[0][0]+2] = (0, 255, 0)
    image[ver_lab[0][3]:ver_lab[0][3]+2, ver_lab[0][2]:ver_lab[0][2]+2] = (0, 255, 0)
    image[ver_lab[1][1]:ver_lab[1][1]+2, ver_lab[1][0]:ver_lab[1][0]+2] = (0, 255, 0)
    image[ver_lab[1][3]:ver_lab[1][3]+2, ver_lab[1][2]:ver_lab[1][2]+2] = (0, 255, 0)
    
    Image.fromarray(image).save(os.path.join('./Verify', image_name))



