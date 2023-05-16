from guided_filter_pytorch.guided_filter import GuidedFilter
import matplotlib.pyplot as plt
import torch
import torchvision.utils as tv_utils

img = plt.imread(r"decompose\dataset\colortrans\new.png")
# guide = plt.imread("sample_images/cave-flash.bmp")
guide = img
tch_img = torch.from_numpy(img).permute(2, 0, 1)[None].float()
tch_guide = torch.from_numpy(guide).permute(2, 0, 1)[None].float()

r1 = 2
r2 = 4
eps = 1e-2
log_epsilon=0.0001
hr_x = torch.log(tch_guide + log_epsilon)
init_hr_y = torch.log(tch_img + log_epsilon)
hr_y = GuidedFilter(r1, eps)(hr_x, init_hr_y)
hr_y2 = GuidedFilter(r2, eps)(hr_x, init_hr_y)
out_filename = r"decompose\guided_filter_pytorch\result_lib.png"
tv_utils.save_image(hr_y, out_filename, normalize=True)
outname = r"decompose\guided_filter_pytorch\result_lib2.png"
tv_utils.save_image(hr_y2, outname, normalize=True)
difference = hr_y-hr_y2
name = r"decompose\guided_filter_pytorch\diff.png"
tv_utils.save_image(difference, name, normalize=True)