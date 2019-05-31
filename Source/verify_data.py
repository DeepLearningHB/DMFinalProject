import os
from PIL import Image
new_file_name = "HK_dataset_align_labels_9.txt"


f = open(new_file_name, 'r')
label_list = f.readlines()
count = 0
for _label in label_list:
    label = _label.replace("\n", '')
    label = label.split(" ")
    c = label[1:]
    c = list(map(float, c))

    if c[0] > 0.5 or c[6] > 0.5 or c[2]  < 0.5 or c[4] < 0.5:
        print(label[0],end=" ")
        print(c)
        count+=1

print("Done")
print(count)
