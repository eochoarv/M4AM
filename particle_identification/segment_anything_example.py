# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os
import sys

dir_paht = '../../learning/segment_anything/Wide Field Images Cropped/'
sys.path.append(dir_paht)

# +
import torchvision.transforms as T
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import segment_anything
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import trange, tqdm
# -

if torch.cuda.is_available():
    print("Good to go!")
    DEVICE = torch.device("cuda")
else:
    print("Please set GPU via Edit -> Notebook Settings.")
    DEVICE = torch.device("cpu")

sam = sam_model_registry["default"](checkpoint=os.path.join("../../learning/segment_anything/", "sam_vit_h_4b8939.pth"))
sam.to(device=DEVICE)

images = [cv2.imread(os.path.join(dir_paht, x)) for x in os.listdir(dir_paht)]
images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

plt.imshow(images[1])

mask_generator = SamAutomaticMaskGenerator(sam)
mask = mask_generator.generate(images[1])

np.random.random((1, 3)).tolist()[0]


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = [.5,.5,.5]#np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


plt.imshow(images[1])
show_anns(mask[1:])
plt.axis('off')
plt.show() 

mask_true = cv2.imread('annotation_mask/SegmentationObject/MEC_02_038_OPTIR_quartz_3.png')
mask_1 = np.array([m['segmentation'] for m in mask[1:]]).sum(0)>0
plt.imshow(mask_1)

# +
mask_true_f = mask_true.reshape(608 * 812, 3)
particles = np.unique(mask_true_f, axis=0)#[mask_true_f.sum(1) > 0]

mask_true_0 = [[np.any([(mask_true[i,j] == p).all() for p in particles[1:]]) for i in range(608)] for j in range(812)]
mask_true_1 = np.array(mask_true_0).T
plt.imshow(mask_true_1)
# -

iou_1 = np.round(100*np.logical_and(mask_true_1, mask_1).sum() / np.logical_or(mask_true_1, mask_1).sum())
iou_1

# +
image = cv2.imread('MEC_02_063_FHS_back_Al_slide_03292023_04262023_run2.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# define the alpha and beta
alpha = 0 # Contrast control
beta = 2 # Brightness control

image_1 = cv2.convertScaleAbs(image, alpha, beta)
plt.imshow(image_1)

# +
# read the input image

# define the contrast and brightness value
contrast = 2. # Contrast control ( 0 to 127)
brightness = 0. # Brightness control (0-100)

# call addWeighted function. use beta = 0 to effectively only

out = cv2.addWeighted( image, contrast, image, 0, brightness)

# display the image with changed contrast and brightness
cv2.imshow('adjusted', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
# -

plt.imshow(image)

mask_generator = SamAutomaticMaskGenerator(sam)
mask = mask_generator.generate(image_1)

plt.imshow(image)
show_anns(mask[3:])
plt.axis('off')
plt.show() 

len(mask)

plt.imshow(image)
show_anns(mask[3:])

mask_1 = np.array([m['segmentation'] for m in mask[3:]]).sum(0)>0
plt.imshow(image)
plt.imshow(mask_1, alpha=.3)

mask_1 = np.array([m['segmentation'] for m in mask[0:]]).sum(0)>0
plt.imshow(mask_1, alpha=.3)

plt.imshow((mask_true[:,:,0]<255)*1)

mask_1

# +
fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(10)

_ = ax[0].imshow(image)
_ = ax[0].imshow(np.concatenate([(mask_1*1)[:,:,None],
 (mask_1*0)[:,:,None],
 (mask_1*0)[:,:,None],
 (mask_1*.5)[:,:,None]],2))
_ = ax[0].title.set_text('masks Segment Amything')

_ = ax[1].imshow(image)
_ = ax[1].imshow(np.concatenate([((mask_true[:,:,0]<255)*1)[:,:,None],
 ((mask_true[:,:,0]<255)*0)[:,:,None],
 ((mask_true[:,:,0]<255)*0)[:,:,None],
 ((mask_true[:,:,0]<255)*.5)[:,:,None]],2))
_ = ax[1].title.set_text('masks OPTIR')
# +
mask_1 = np.array([m['segmentation'] for m in mask]).sum(0)>0

fig, ax = plt.subplots(nrows=1, ncols=1)
fig.set_figheight(10)
fig.set_figwidth(10)

_ = ax.imshow(image)
_ = ax.imshow(np.concatenate([(mask_1*1)[:,:,None],
 (mask_1*0)[:,:,None],
 (mask_1*0)[:,:,None],
 (mask_1*.5)[:,:,None]],2))
_ = ax.title.set_text('masks Segment Amything')
# -



mask_true = cv2.imread('MEC_02_063_FHS_back_Al_slide_03292023_04262023_run2_masks.png')

np.unique(mask_true[:,:,0])
