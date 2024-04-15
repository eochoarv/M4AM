#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:48:17 2023

@author: eochoa
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import cv2
import cv2 as cv
from sklearn.metrics import pairwise_distances

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import trange, tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.cluster import KMeans


def particle_dection(image, thr_l, thr_h):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    low = np.array([thr_l])
    high = np.array([thr_h]) #np.array([np.int64(np.quantile(gray, .025))])
    
    #ret, thresh = cv.threshold(gray,0,255, 9)#,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    thresh = cv.inRange(gray, low, high)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)

    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.01*dist_transform.max(),255,0)

    mask = (sure_fg==255)
    return mask



def particle_filter(image, coords, alpha, beta, thr_l, thr_h, t=60):
    
    mask = particle_dection(image, thr_l, thr_h)
    ret, labels = cv.connectedComponents(((mask.astype('uint8')) * 255))
    coords_labels_list = [np.array(np.where(labels == i)).T for i in range(1,ret)]
    distances_part = np.array([pairwise_distances(c, coords).min() for c in coords_labels_list])
    keep_idx = np.where(distances_part < t)[0]+1
    
    # area and perimeter (size)
    #true_areas = np.array([(labels==i).sum() for i in keep_idx])
    #bbox_areas = np.array([np.prod(np.array(np.where(labels==i)).max(1) - np.array(np.where(labels==i)).min(1) + 1) for i in keep_idx])
    perimeters = []
    true_areas = []
    chull_areas = []
    for i in keep_idx:
        mask_1 = np.uint8(labels==i)
        contours,hierarchy = cv.findContours(mask_1, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        hull = cv.convexHull(cnt)
        perimeter = cv.arcLength(cnt, True)
        area = cv.contourArea(cnt)
        ch_area = cv.contourArea(hull)
        #mask_1_ = mask_1.copy()
        #cv.drawContours(mask_1_, [cnt], -1, (0,255,255), 1)
        #perimeters.append((mask_1.sum() - mask_1_.sum()))
        perimeters.append(perimeter)
        true_areas.append(area)
        chull_areas.append(ch_area)
        
    perimeters = np.int64(perimeters)
    true_areas = np.int64(true_areas)
    chull_areas = np.int64(chull_areas)
    
    sizes_distr = true_areas / (alpha*beta)
    
    # shape
    solidities = true_areas / chull_areas
    FF = 4*np.pi*true_areas / (perimeters ** 2)
    
    mask_keep_list = [labels == i for i in keep_idx]
    mask_keep = mask_keep_list[0] 
    for m in mask_keep_list:
        mask_keep = mask_keep + m
    
    return sizes_distr, FF, solidities, mask, mask_keep
    
    