import os
os.chdir("..")
from copy import deepcopy

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure

from src.loftr import LoFTR, default_cfg

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# Initialize LoFTR
_default_cfg = deepcopy(default_cfg)
#_default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
_default_cfg['coarse']['temp_bug_fix'] = True 
matcher = LoFTR(config=_default_cfg)
matcher.load_state_dict(torch.load("/home/raul/GroundingDINO/LoFTR/weights/indoor_ds_new.ckpt")['state_dict'])
matcher = matcher.eval().cuda()

# Load example images
img0_pth = "/home/raul/GroundingDINO/LoFTR/assets/MicrosoftTeams-image.png"
img1_pth = "/home/raul/GroundingDINO/LoFTR/assets/ReviewTable.png"
img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
img0_raw = cv2.resize(img0_raw, (640, 480))
img1_raw = cv2.resize(img1_raw, (640, 480))

print(img0_raw)
print(img1_raw)

img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
batch = {'image0': img0, 'image1': img1}
input("ok")

# Inference
with torch.no_grad():
    matcher(batch)    # batch = {'image0': img0, 'image1': img1}
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf  = batch['mconf'].cpu().numpy()

print(batch)
print(mkpts0.dtype)
print(mkpts1.dtype)
input("oi")
    
# Draw
color = cm.jet(mconf)
text = [
    'LoFTR',
    'Matches: {}'.format(len(mkpts0)),
]
fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)

plt.show()
