import numpy as np
import torch
import torchvision.utils as tv_utils
import os
import matplotlib.pyplot as plt

img = plt.imread(r"decompose\results\original_image.png")
img2 = plt.imread(r"decompose\results\rebuild_image.png")
print(max(img - img2))