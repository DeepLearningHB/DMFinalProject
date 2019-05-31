import os
import numpy as np
import cv2
from PIL import Image, ImageEnhance
label_paths=['HK_dataset_align_labels.txt']
image_paths = ['./AlignImages']
image_size = (128, 128)

def rounding(x):
    return round(x, 6)
def load_dataset(augmented=False):

    image_list = []
    labels = []
    file_names = []
    w_h_list = []


    for idx, label_path in enumerate(label_paths):
        f = open(label_path, 'r')
        label_list = f.readlines()
        for _label in label_list:
            # print(_label)
            # print(type(_label))
            # exit()
            try:
                label = _label.replace("\n", '')
            except:
                print("Endgame.")
                continue
            label = label.split(" ")
            file_name = label[0]

            coord = list(map(float, label[1:]))

            #label coordinate preprocessing
            # [x1, y1, x2, y2, x3, y3, x4, y4]




            try:
                image = Image.open(os.path.join(image_paths[idx], file_name))
            except:
                continue
            image = image.resize(image_size, resample=Image.BILINEAR)
            image_list.append(np.array(image))
            file_names.append(file_name)
            # [x1, y1, x2, y2, x3, y3, x4, y4]
            labe = np.array([[coord[0], coord[1], coord[6], coord[7]], [coord[2], coord[3], coord[4], coord[5]]])
            # labe = np.array([[coord[0], coord[1]],
            #                  [coord[2], coord[3]],
            #                  [coord[4], coord[5]],
            #                  [coord[6], coord[7]]])
            labels.append(labe)


            #labe = np.array([coord[0], coord[1], coord[2], coord[3], coord[6], coord[7], coord[4], coord[5]])

            width = max(coord[::2]) - min(coord[::2])
            height = max(coord[1::2]) - min(coord[1::2])

            w_h_list.append(np.array([width, height]))

            # enhancer = ImageEnhance.Brightness(image)
            # enhanced_im = np.array(enhancer.enhance(0.7))
            # image_list.append(np.array(enhanced_im))
            # labels.append(labe)
            # file_names.append(file_name)
            # w_h_list.append(np.array([width, height]))

            if augmented == True: # Image augmentation options: flip

                image_ud = cv2.flip(np.array(image), 0)
                label_ud = np.round_(np.array([[coord[6], 1-coord[7], coord[0], 1-coord[1]],[coord[4], 1-coord[5], coord[2], 1-coord[3]]]), 5)

                image_list.append(image_ud)
                labels.append(label_ud)
                file_names.append(file_name)
                w_h_list.append(np.array([width, height]))

                image_lr = cv2.flip(np.array(image), 1)
                label_lr = np.round_(np.array([[1-coord[2], coord[3], 1-coord[4], coord[5]], [1-coord[0], coord[1], 1-coord[6], coord[7]]]), 5)

                image_list.append(image_lr)
                labels.append(label_lr)
                file_names.append(file_name)
                w_h_list.append(np.array([width, height]))

                image_udlr = cv2.flip(image_ud, 1)
                label_udlr = np.round_(np.array([[1-coord[4], 1-coord[5], 1-coord[2], 1-coord[3]],[1-coord[6], 1-coord[7], 1-coord[0], 1-coord[1]]]),5)
                image_list.append(image_udlr)
                labels.append(label_udlr)
                file_names.append(file_name)
                w_h_list.append(np.array([width, height]))
        f.close()


    image_list = np.array(image_list)
    labels = np.array(labels)
    labels = labels.astype(np.float32)
    w_h_list = np.array(w_h_list)
    w_h_list = np.expand_dims(w_h_list, axis=1)
    return image_list, labels, file_names, w_h_list
