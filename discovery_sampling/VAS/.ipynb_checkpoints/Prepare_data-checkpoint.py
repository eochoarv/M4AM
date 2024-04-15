# create two csv files with image name and label at each row
import os
import cv2
import aug_util as aug
import wv_util as wv
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image
import numpy as np
import json
from tqdm import tqdm
import csv
import pandas as pd
# %matplotlib inline
import torchvision.transforms as transforms

# +
#os.chdir('/Users/eochoa/Documents/Research_Ambuj/ASD/Code/')

from sklearn.linear_model import LogisticRegression
#import pandas as pd
from utils import *
#from linear_utils import *
from logistic_utils import *
#from GP_utils import *
#from linear_utils import *
#import expt_utils as expt_utils
# -

#Load the class number -> class string label map
labels = {}
with open('/Users/eochoa/Documents/Research_Ambuj/learning/VAS/xview_class_labels.txt') as f:
    for row in csv.reader(f):
        labels[int(row[0].split(":")[0])] = row[0].split(":")[1]


data_path = "/Users/eochoa/Documents/Research_Ambuj/learning/VAS/xView/train_images/"
input_path_list = os.listdir(data_path)
chip_name_list = list(); label_list = list()

coords0, chips0, classes0 = wv.get_labels('/Users/eochoa/Documents/Research_Ambuj/learning/VAS/xView/xView_train.geojson')
coord_check1 = np.logical_and(coords0[:,0] < coords0[:,2], coords0[:,1] < coords0[:,3])
coord_check2 = np.logical_and(coords0[:,2] > 0, coords0[:,3] > 0)
coord_check = np.logical_and(coord_check1, coord_check2)
coords0 = coords0[coord_check]
chips0 = chips0[coord_check]
classes0 = classes0[coord_check]

images = [wv.get_image(data_path + img) for img in tqdm(input_path_list)]
[image.shape for image in images]

# +
reg_list = {i:[] for i in range(len(images))}
act_list = {i:[] for i in range(len(images))}
tau_list = {i:[] for i in range(len(images))}

for i, img in tqdm(enumerate(input_path_list)): 
    chip_name_ = img
        #We only want to coordinates and classes that are within our chip
    coords, chips, classes = coords0.copy(), chips0, classes0.copy()
    coords = coords[chips==chip_name_]
    classes = classes[chips==chip_name_].astype(np.int64)
    if classes.shape[0]<=1:
        continue
    arr = images[i]
    
    coords_car = np.int32(coords)
    coords_car_ = coords
    rgb_ = [arr[max(0,y1):y2, max(0,x1):x2].mean((0,1)) for x1,y1,x2,y2 in zip(coords_car[:,0],coords_car[:,1],coords_car[:,2],coords_car[:,3])]
    cord_x = (coords_car[:,0] + coords_car[:,2]) / 2
    cord_y = (coords_car[:,1] + coords_car[:,3]) / 2
    x = np.concatenate([cord_x[:,None], cord_y[:,None], np.array(rgb_)], 1)
    x = (x - x.mean(0)) / x.std(0)
    y = (classes == 18)*1
    
    if y.sum()==0:
        continue
    clf = LogisticRegression(random_state=0).fit(x, y)
    #print(clf.score(x, y))
    
    n_row, n_col = y.shape[0], 1
    _, n_desc =  x.shape
    cutoff = n_desc
    T = n_row
    prob_dict = {
            "m": n_row,
            "noise": 0.01,
            "d": cutoff
          }
    prob = logistic_discover(**prob_dict)
    prob.x = x.copy()
    prob.y = y.copy()
    t_prob = deepcopy(prob)
    solver = G_logistic(t_prob, prob_threshold=0.5) 
    
    reg, rwd, tau = solver.run(n_row)

    reg_list[i].append(reg)
    act_list[i].append(solver.prob.actions)
    tau_list[i].append(tau)
    
open_file = open("reg_list.pkl", "wb")
pickle.dump(reg_list, open_file)
open_file.close()

open_file = open("act_list.pkl", "wb")
pickle.dump(act_list, open_file)
open_file.close()

open_file = open("tau_list.pkl", "wb")
pickle.dump(tau_list, open_file)
open_file.close()


# +
import pickle

with open('reg_list.pkl', 'rb') as f:
    reg_list = pickle.load(f)

with open('act_list.pkl', 'rb') as f:
    act_list = pickle.load(f)

with open('tau_list.pkl', 'rb') as f:
    tau_list = pickle.load(f)

#open_file = open("reg_list.pkl", "wb")
#pickle.dump(reg_list, open_file)
#open_file.close()

#open_file = open("act_list.pkl", "wb")
#pickle.dump(act_list, open_file)
#open_file.close()

#open_file = open("tau_list.pkl", "wb")
#pickle.dump(tau_list, open_file)
#open_file.close()
# -

reg_list

recall_list = [ ]
tau_prop_list = [ ]
n_obj_list = []
for i, img in enumerate(input_path_list[:63]): 
    chip_name_ = img
        #We only want to coordinates and classes that are within our chip
    coords, chips, classes = coords0.copy(), chips0, classes0.copy()
    coords = coords[chips==chip_name_]
    classes = classes[chips==chip_name_].astype(np.int64)
    if classes.shape[0]<=1:
        continue
    arr = images[i]
    
    coords_car = np.int32(coords)
    coords_car_ = coords
    rgb_ = [arr[max(0,y1):y2, max(0,x1):x2].mean((0,1)) for x1,y1,x2,y2 in zip(coords_car[:,0],coords_car[:,1],coords_car[:,2],coords_car[:,3])]
    cord_x = (coords_car[:,0] + coords_car[:,2]) / 2
    cord_y = (coords_car[:,1] + coords_car[:,3]) / 2
    x = np.concatenate([cord_x[:,None], cord_y[:,None], np.array(rgb_)], 1)
    x = (x - x.mean(0)) / x.std(0)
    y = (classes == 18)*1
    
    if len(act_list[i])>0:
        if not(np.isnan(tau_list[i][0])):
            recall_list.append(y[act_list[i][0][:tau_list[i][0]]].sum()/y.sum())
            tau_prop_list.append(tau_list[i][0]/y.shape[0])
            n_obj_list.append(y.shape[0])
            
            print(str([i, y.shape[0], tau_list[i][0]/y.shape[0], int(100*y[act_list[i][0][:tau_list[i][0]]].sum()/y.sum()), y.sum(), len(act_list[i][0]), np.max(reg_list[i])]))
        else:
            recall_list.append(y[act_list[i][0]].sum()/y.sum())
            tau_prop_list.append(1)
            n_obj_list.append(y.shape[0])
            print(str([i, y.shape[0], tau_list[i][0]/y.shape[0], int(100*y[act_list[i][0]].sum()/y.sum()), y.sum(), len(act_list[i][0]), np.max(reg_list[i])]))


np.sum(n_obj_list)

np.sum(np.array(n_obj_list) * np.array(tau_prop_list)) / np.sum(n_obj_list)

np.mean(recall_list)

np.mean(tau_prop_list)

act_list[i][0]

# +
open_file = open("reg_list.pkl", "wb")
pickle.dump(reg_list, open_file)
open_file.close()

open_file = open("act_list.pkl", "wb")
pickle.dump(act_list, open_file)
open_file.close()

open_file = open("tau_list.pkl", "wb")
pickle.dump(tau_list, open_file)
open_file.close()
# -

input_path_list[:63]

# +
#fig, ax = plt.subplots(nrows=10, ncols=1, sharey=True)
#fig.set_figheight(150)
#fig.set_figwidth(170)
#fig.tight_layout(pad=2.0)

for i, img in tqdm(enumerate(input_path_list[:63])): 
    chip_name_ = img
        #We only want to coordinates and classes that are within our chip
    coords, chips, classes = coords0.copy(), chips0, classes0.copy()
    coords = coords[chips==chip_name_]
    classes = classes[chips==chip_name_].astype(np.int64)
    image = images[i]
    
    cord_x = (coords[:,0] + coords[:,2]) / 2
    cord_y = (coords[:,1] + coords[:,3]) / 2
    
    #y = (masks[i].flatten() == 6)*1#.reshape(4000 * 3000, 1)
    #ss_y = y[idx]
    
    #cmap = plt.cm.Reds
    #line_colors = cmap(np.linspace(0,1,len(act_list[i][0])))

    #ax[j].imshow(images[i][:224,:298,:])

    #ax[j].scatter(cord_x[idx][act_list[i]].flatten( ), 
    #            cord_y[idx][act_list[i]].flatten(), c=line_colors)#np.array(range(n))/n)
    #ax[j].title.set_text(str([ss_y[act_list[i]].sum()/ss_y.sum(), ss_y.sum(), len(act_list[i][0]), np.max(reg_list[i])]))
    
    
    #
    for j in range(classes.shape[0]):
        if classes[j] == 18:
            image = cv2.rectangle(image, 
                              (int(coords[j,0]), int(coords[j,1])), 
                              (int(coords[j,2]), int(coords[j,3])), 
                              (255, 0, 0), 4)
        else:
            image = cv2.rectangle(image, 
                              (int(coords[j,0]), int(coords[j,1])), 
                              (int(coords[j,2]), int(coords[j,3])), 
                              (0, 0, 255), 2)
    #cv2.imshow(window_name, image) 
    
    _ = plt.figure(figsize=(10,10))
    _ = plt.imshow(image)
    
    if len(act_list[i]) ==0:
        continue
    if len(tau_list[i])>0 and not(np.isnan(tau_list[i][0])):
        act = act_list[i][0][:tau_list[i][0]]
    else:
        act = act_list[i][0]

    #_ = plt.scatter(cord_x[act], 
    #                cord_y[act], c='y', s=.5)
    
    #plt.savefig(os.path.join('output_images', 'image_'+str(i)+'.png'))


# +
alpha = 0
cmap = plt.cm.Blues
line_colors = cmap(np.linspace(0.5,1,62))
for i in range(62):
    if len(reg_list[i])==0:
        continue
    #algo_label = algos_dict[i][3]
    #algo_style = algos_dict[i][4]
    y_ = np.mean(reg_list[i], axis = 0)
    y_ = y_ / np.power((np.ones(y_.shape[0])).cumsum(), alpha)
    ci = np.std(reg_list[i], axis = 0)/ np.power((np.ones(y_.shape[0])).cumsum(), alpha)
    plt.plot(range(y_.shape[0]), y_, c=line_colors[i])
    #plt.fill_between(range(y_.shape[0]), (y_-ci), (y_+ci), alpha=.1)

#plt.legend()
# -

algos_dict = {
   'Random': [logistic_solver, {}, 1, 'Random', 'green'],    
  'TS_logistic': [TS_logistic, {'gamma':1e15, 'prob_threshold':0}, 10, 'TS_logistic', 'darkblue'],     
   #'IDS_logistic': [IDS_logistic, {}, 10, 'IDS_logistic', 'yellow'],   
    'Greedy_logistic': [G_logistic, {'prob_threshold':0.5}, 1, 'Greedy_logistic', 'red']   
}

# +
#plt.plot(reg_1[:tau])
#plt.plot(np.cumsum(rwd[:tau])/4.79)

plt.plot(1+reg_1)
plt.plot(1+np.cumsum(rwd))

# +
#x_ = logit(np.array(solver.prob.rewards_ev))
x_ = np.array(solver.prob.rewards_ev).flatten()
y_ = np.cumsum(solver.prob.rewards)/y.sum()

ci = np.array(solver.prob.uncertainty)

plt.plot(y_, expit(x_))
plt.fill_between(y_, expit(x_-ci), expit(x_+ci), alpha=.1)
# -

reg_list = {algo:[] for algo in algos_dict}
rwd_list = {algo:[] for algo in algos_dict}
#tau_list = {algo:[] for algo in algos_dict}
for trial_idx in trange(5):
            # print(prob.x)
    for algo, algo_setup in algos_dict.items():
        t_prob = deepcopy(prob)
        algo_func, algo_params, algo_N_trials, _, _ = algo_setup
                        # print(algo_func)
        if not trial_idx < algo_N_trials:
            continue
        solver = algo_func(t_prob, **algo_params)
        reg, rwd = solver.run(T)[:2]
                #act_list[algo].append(getattr(solver, "logistic_discover"))
                #act_list[algo].append(t_prob.actions)
                # print(prob.rewards)
        reg_list[algo].append(reg)
        rwd_list[algo].append(np.cumsum(rwd))
                #tau_list[algo].append(tau)

        #for algo in algos_dict:
        #algo_label = algos_dict[algo][3]
        #algo_style = algos_dict[algo][4]

alpha = 0
for algo in algos_dict:
    algo_label = algos_dict[algo][3]
    algo_style = algos_dict[algo][4]
    y_ = np.mean(reg_list[algo], axis = 0)
    y_ = y_ / np.power((np.ones(y_.shape[0])).cumsum(), alpha)
    ci = np.std(reg_list[algo], axis = 0)/ np.power((np.ones(y_.shape[0])).cumsum(), alpha)
    plt.plot(range(y_.shape[0]), y_,
                  algo_style, label=algo_label)
    plt.fill_between(range(y_.shape[0]), (y_-ci), (y_+ci), color = algo_style, alpha=.1)

    plt.legend()
