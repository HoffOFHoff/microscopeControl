# -*- coding: UTF-8 -*-

# apply the package we need
import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
# cv.namedWindow("focus_check1_img", cv.WINDOW_NORMAL)
train_dir = os.path.join("./Daniel_FocusTest_20220915_125656_10x_HFF_6.96umStep/")
train_image_names = os.listdir(train_dir)
# print(train_image_names)
capture_name=[]
capture_lap=[]
capture=[capture_name,capture_lap]

#laplician is a filter for kernal convolution method
min_lap = 10000
max_lap = 0
avg =0 
for i in range(0, len(train_image_names)):
    img = cv.imread(train_dir +"/" + train_image_names[i])
    capture_name.append(train_image_names[i])
    laplacian_var_i = cv.Laplacian(img, cv.CV_64F).var()
    capture_lap.append(laplacian_var_i)
    avg = avg+ laplacian_var_i
    if laplacian_var_i > max_lap:
        max_lap = laplacian_var_i
    if laplacian_var_i < min_lap:
        min_lap = laplacian_var_i

avg = avg/len(train_image_names)
print(min_lap)
# print(capture)
focus =[]
no_focus = []
focus_lap=[]
focus_img=[]
no_focus_lap=[]
no_focus_img=[]
for i in range(0, len(train_image_names)):
    img = cv.imread(train_dir +"/" + train_image_names[i])
    laplacian_var_i = cv.Laplacian(img, cv.CV_64F).var()
    if laplacian_var_i > avg:
        focus.append(train_image_names[i])
        focus_img.append(img)
        focus_lap.append(laplacian_var_i)
        # cv.imshow('focus', img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()


    if laplacian_var_i < avg:
        no_focus.append(train_image_names[i])
        no_focus_img.append(img)
        no_focus_lap.append(laplacian_var_i)

# focus_img = np.hstack(focus_img)
# cv.imshow("mutil_pic", focus_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

plt.figure('focus',figsize=(10, 10))
for i in range(0, len(focus)):
    plt.subplot(4,4,i+1),plt.imshow(focus_img[i])
    plt.title(focus_lap[i],fontsize=8)
    plt.xticks([]),plt.yticks([])

plt.figure('no_focus',figsize=(10, 10))
for i in range(0, len(focus)):
    plt.subplot(4,4,i+1),plt.imshow(no_focus_img[i])
    plt.title(no_focus_lap[i],fontsize=8)
    plt.xticks([]),plt.yticks([])
plt.show()
print('here is focus')
focus=[focus,focus_lap]
print(focus)
print('here is the no-focus one')
focus=[no_focus,no_focus_lap]
print(no_focus)
# #cv.imshow('laplacian', laplacian)

#REF https://www.youtube.com/watch?v=5YP7OoMhXbM&ab_channel=Pysource
#REF https://stackoverflow.com/questions/58231849/how-to-remove-blurriness-from-an-image-using-opencv-python-c