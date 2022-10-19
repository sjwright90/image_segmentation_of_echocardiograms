#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 20:26:54 2022

@author: Samsonite
"""
import imageio
import matplotlib.pyplot as plt
import os
import scipy.ndimage as ndi
import numpy as np
import matplotlib.animation as animation
#%%
img1 = imageio.imread('chest-220.dcm')
imageio.volread(uri, kwargs)


#%%

fig, ax = plt.subplots(3,3)
ax[0].imshow(img1)
plt.show()


#%%
x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x ** 2)

#%%
fig, ax = plt.subplots(1,3)
ax[0].plot(x,y)
ax[1].scatter(y,x)
ax[2].imshow(img1)

#%%
vol = imageio.volread('tcia-chest-ct-sample')
#%%
fig, ax = plt.subplots(nrows = 1, ncols = vol.shape[0])
for im in range(len(ax)):
    ax[im].imshow(vol[im], cmap = 'gray')
for axes in ax:
    axes.axis('off')
plt.show()
#%%
fig, ax = plt.subplots(nrows = vol.shape[1], ncols = 1)
d0, d1,d2 = vol.meta['sampling']
asp = d0/d2
for im in range(len(ax)):
    ax[im].imshow(vol[:,im, :], cmap = 'gray', aspect = asp)
for axes in ax:
    axes.axis('off')
plt.show()
#%%

#%%
vol_full = imageio.volread('SCD2001_010')















#%%
def plot_image(putin):
    plt.imshow(putin, cmap = 'gray')
    plt.axis('off')
    plt.show()


#%%
heart_vol = imageio.volread('SCD2001_000')
#%%
def load_brighten_makegif(folder = ''):
    heart_vol = imageio.volread(folder)
    heart_vol_g = ndi.median_filter(heart_vol, size = 3)
    
    heart_contr = np.empty((heart_vol_g.shape[0], 
                            heart_vol_g.shape[1], 
                            heart_vol_g.shape[2]))
    minval = heart_vol_g.min()
    maxval = heart_vol_g.max()
    for i in range(heart_vol_g.shape[0]):
        heart_hist = ndi.histogram(heart_vol_g[i], 
                                   min = minval,
                                   max = maxval,
                                   bins = maxval-minval + 1)
        cdf = heart_hist.cumsum()/heart_hist.sum()
        heart_contr[i] = cdf[heart_vol_g[i]] * 255
    
    
    ims = []
    fig, ax = plt.subplots()
    for im in range(heart_vol_g.shape[0]):
        ax.axis('off')
        h = ax.imshow(heart_vol_g[im], cmap = 'gray', animated = True)
        ims.append([h])
    
    ani = animation.ArtistAnimation(fig, ims, interval = 50, 
                                    blit = True, 
                                    repeat_delay = 1000)
    title = folder + '.gif'
    ani.save(title)
    return heart_vol, heart_vol_g, heart_contr
#%%
scd2001_001_vol, scd2001_001_vol_g, scd2001_001_contr = load_brighten_makegif(folder = 'SCD2001_001')
#%%

def mask_label(three_D_img, thresh = 0):
    '''input is three dimensional timeseries, where each slice is a 2d image
    and the first index is the time [n, x, y] where n is time and x,y is the
    2d image to display, thresh is an integer input to set the threshold for
    making the mask
    '''
    for i in range(three_D_img.shape[0]):
        if i == 0:
            labels = np.empty((three_D_img.shape[0], 
                              three_D_img.shape[1], 
                              three_D_img.shape[2]))
            overlay = np.empty((three_D_img.shape[0], 
                              three_D_img.shape[1], 
                              three_D_img.shape[2]))
            nlabels = np.empty(three_D_img.shape[0])
           
        mask = np.where(three_D_img[i] > thresh, 1, 0)
        mask_close = ndi.binary_closing(mask)
        labels_hold, nlabels[i] = ndi.label(mask_close)
        labels[i] = labels_hold
        mcommon = np.array([np.bincount(row).argmax() 
                                        for row in labels_hold])
        mcommon = np.bincount(mcommon).argmax()
        labels_hold = np.where(labels_hold != mcommon, labels_hold, np.nan)
        overlay[i] = labels_hold
        
    
    return labels, nlabels, overlay
#%%
def quick_input_plot(image, thresh = 0):
    x, y, x = mask_label(image, thresh = thresh)
    plt.imshow(x[0], cmap = 'rainbow')
    plt.show()

#%%
fig, ax = plt.subplots()
ax.imshow(smooth_003[0], cmap = 'gray')
ax.imshow(overlay_003[0], cmap = 'rainbow', alpha = .50)
fig.show()
#%%
def make_overlay_gif(three_D, overlay, save_as = ''):
    ims = []
    fig, ax = plt.subplots()
    for im in range(three_D.shape[0]):
        ax.axis('off')
        h = ax.imshow(three_D[im], cmap = 'gray', animated = True)
        z = ax.imshow(overlay[im], cmap = 'rainbow', animated = True, alpha = 0.5)
       
        ims.append([h, z])
    
    ani = animation.ArtistAnimation(fig, ims, interval = 50, 
                                    blit = True, 
                                    repeat_delay = 1000)
    title = save_as + '.gif'
    ani.save(title)
#%%
scd2001_001_g_labels, scd2001_001_g_nlabels, overlay = mask_label(scd2001_001_vol_g, 
                                                         thresh = 75)  
#%%
def find_boxes(labeled_threeD):
    bboxes = []
    box_lv = []
    lv_mask_all = np.empty((labeled_threeD.shape[0], 
                      labeled_threeD.shape[1], 
                      labeled_threeD.shape[2]))
    for i in range(labeled_threeD.shape[0]):
        lv_l = labeled_threeD[i][119,146]
        box_lv.append(lv_l)
        lv_mask = np.where(labeled_threeD[i] == lv_l, 1,0)
        lv_mask_all[i] = lv_mask
        bboxes.append( ndi.find_objects(lv_mask))
    return bboxes, lv_mask_all, box_lv
#%%

#%%
def calc_ejtc(orig, lv_mask):
    d0, d1, d2 = orig.meta.sampling
    dvoxel = d1*d2
    ts = np.zeros(orig.shape[0])
    for t in range(orig.shape[0]):
        nvoxel = ndi.sum(1, 
                         lv_mask[t])
        ts[t] = nvoxel * dvoxel
    plt.plot(ts)
    plt.show()
    edjfract = (ts.max() - ts.min())/ts.max()
    
    return edjfract, ts
    
#%%
boxes_scd_001, lv_mask_scd_001 = find_boxes(scd2001_001_g_labels.astype('uint8'))
#%%
def make_gif(three_D, save_as = ''):
    ims = []
    fig, ax = plt.subplots()
    for im in range(three_D.shape[0]):
        ax.axis('off')
        h = ax.imshow(three_D[im], cmap = 'rainbow', animated = True)
        ims.append([h])
    
    ani = animation.ArtistAnimation(fig, ims, interval = 50, 
                                    blit = True, 
                                    repeat_delay = 1000)
    title = save_as + '.gif'
    ani.save(title)
#%%
make_gif(scd2001_001_g_labels, save_as = '001_labels')
#%%
d0, d1, d2 = vol_005.meta.sampling
dvoxel = d1*d2
ts = np.zeros(vol_005.shape[0])
for t in range(vol_005.shape[0]):
    nvoxel = ndi.sum(1,
                     lv_005[t],
                     )
    ts[t] = nvoxel * dvoxel
#%%

#%%
ndi.find_objects()
ndi.gaussian_filter()

#%%

d0, d1, d2 = heart_vol.meta.sampling
asp_0_2 = d0/d2
asp_o_1 = d0/d1
#%%
heart_hist = ndi.histogram(heart_vol[0], min = 0, max = 255, bins = 256)
plt.plot(heart_hist)
#%%
heart_contr = np.empty((heart_vol.shape[0], heart_vol.shape[1], heart_vol.shape[2]))
for i in range(heart_vol.shape[0]):
    heart_hist = ndi.histogram(heart_vol[i], min = 0, max = 255, bins = 256)
    cdf = heart_hist.cumsum()/heart_hist.sum()
    heart_contr[i] = cdf[heart_vol[i]] * 255

#%%

ims = []
fig, ax = plt.subplots()
for im in range(heart_contr.shape[0]):
    h = ax.imshow(heart_contr[im], cmap = 'gray', animated = True)
    ims.append([h])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()
ani.save("movie.gif")


#%%
fig, ax = plt.subplots(nrows = 1, ncols = heart_vol.shape[0])
for im in range(len(ax)):
    ax[im].imshow(heart_vol[im], cmap = 'gray')
    ax[im].axis('off')
    plt.pause(3)

#%%
vol = imageio.volread('SCD2001_001')
vol_sm = ndi.gaussian_filter(vol, sigma = 0.3)
mask = np.where(vol_sm > 75, 1, 0)
mask_close = ndi.binary_closing(mask)
labels, nlabels = ndi.label(mask_close)

#%%
from pathlib import Path
paths = Path().glob('SCD2001_\d\d\d')
for path in paths:
    print(path)  
    
#%%
for root, dirs, files in os.walk('/Users/Samsonite/CourseraStrAlg/KNN/biomed_imaging'):
    for i in files:
        print(os.path.join(root, i))