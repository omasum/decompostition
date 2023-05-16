import cv2
import os
import numpy as np

path = r"decompose\results"
name = os.listdir(path=path)
original_path = r"decompose\dataset\radn_test"
ori_name = os.listdir(original_path)

for i in range(1,6):
    sum = []
    original = cv2.imread(os.path.join(original_path, ori_name[i-1]))
    for item in name:
        if item[0] == str(i):
            sum.append(cv2.imread(os.path.join(path, item))) # a list of [H, W, C]
    sum.append(original)
    image = np.concatenate(sum, axis = 1)
    cv2.imwrite( filename= os.path.join(path, str(i)+'.png'), img=image)

