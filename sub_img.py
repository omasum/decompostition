import cv2
import numpy as np

path = r"decompose\results\original_image.png"
pathm1 = r"decompose\results\rebuild_map.png"
pathm2 = r"decompose\results2\results\rebuild_map.png"
pathr = r"decompose\results\rebuild_image.png"

original = cv2.imread(path)
r = cv2.imread(pathr)

print("origianl - r", np.where((original - r) != 0))