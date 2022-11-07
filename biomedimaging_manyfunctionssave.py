#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 20:26:54 2022

@author: Samsonite
"""
# %%

# Import necessary packages
import imageio
import matplotlib.pyplot as plt
import os
import scipy.ndimage as ndi
import numpy as np
import matplotlib.animation as animation
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen
#%%
# get artist name for plot saving purposes
aritistusr = input("Enter artist name to save in image metadata: ")

# make writer for plot saving
gifwriter = animation.PillowWriter(fps = 15, \
        metadata=dict(artist = aritistusr), bitrate=1800)
# %%
# Pull directly from the link
url = "https://assets.datacamp.com/production/repositories/2085/datasets/fabaa1f1675549d624eb8f5d1bc94e0b11e30a8e/sunnybrook-cardiac-mr.zip"

resp = urlopen(url) # open the zipped url

myzip = ZipFile(BytesIO(resp.read())) # get zipped file

# Extract "SCD2001_005" folder from the zipped repository
# Other options are 001-010
for file in myzip.namelist():
    if file.startswith("SCD2001_005/"):
        myzip.extract(file, ".")
resp.close()

# %% 
# Simple plotting function for grayscale image
def plot_image(putin):
    plt.imshow(putin, cmap = 'gray')
    plt.show()
# %%
# Print directory contents of SCD2001_005
contents = sorted(os.listdir("SCD2001_005"))
print(contents)
# %%
# Open first dcm file
heart_vol = imageio.volread(f'SCD2001_005/{contents[0]}')
#%%
def load_brighten_makegif(dcmfile='', dirtosave=None):
    '''Load dicom file, bighted, make gif, and save output gif
    
    Parameters
    ----------
    dcmfile: string
        path to DICOM file to process

    dirtosave: string
        Default None, default will save file in same directory
        as dcmfile with same file name but in .gif format. Input specific
        file path and file name to save .gif file in that location.
        e.g. dirttosave = "~/Documents/images/filename.gif"
    
    Returns
    -------
    heart_vol: imageio.core.util.Array
        ImageIO object of raw data file

    heart_vol_g: numpy.ndarray
        Original image with median filter applied, will no longer
        contain DICOM metadata

    heart_cont: numpy.ndarray
        Filtered image with contrast brightening applied, will no
        longer contain DICOM metadata

    Prints gif output to console'''



    heart_vol = imageio.volread(dcmfile)

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
    if isinstance(dirtosave, str):
        if dirtosave.endswith(".gif"):
            title = dirtosave
        else:
            title = dirtosave + ".gif"
    else:
        title = dcmfile.replace(".dcm","") + '.gif'

    ani.save(title, writer=gifwriter)

    return heart_vol, heart_vol_g, heart_contr
# %%

def mask_label(three_D_img, thresh = 0):
    '''Label regions by brightness for time series image.

    Parametes
    ---------
    three_D_img: array-like
        3-D array of image pixels where the first axis is time
        if using 2-D image add length 1 3rd dimension as the first axis

    thresh: float or int
        Value to use for masking threshhold

    Output
    ------
    labels: array-like
        3-D array of same structure as input array, all pixels replaced
        by corresponding masked label

    nlabels: array-like
        1-D array of counts of number of labels per slice of 3-D image input

    overlay: array-like
        3-D array of same structure as input array. Pixels replaced by
        labeled value, but most common label replaced by NaN. Most common
        label will usually correspond to background so this return will
        be better for plotting, if most common is not the background
        use 'labels' for plotting
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

        # make binary mask using threshold
        mask = np.where(three_D_img[i] > thresh, 1, 0)

        # close small holes in mask
        mask_close = ndi.binary_closing(mask)

        # temp holding of current slice, store number of labels in slice
        labels_hold, nlabels[i] = ndi.label(mask_close)

        # store labeled slice
        labels[i] = labels_hold

        # calculate highest occurence
        lab, cnts = np.unique(labels_hold, return_counts=True)

        mcommon = lab[cnts.argmax()]

        # replace highest occurance with NaN
        labels_hold = np.where(labels_hold != mcommon, labels_hold, np.nan)

        # store filtered labels in overlay array
        overlay[i] = labels_hold
        
    
    return labels, nlabels, overlay
#%%
def quick_input_plot(image, thresh = 0):
    '''Make quick plot of an image
    
    Parameters
    ----------
    image: array-like
        Image to plot

    thresh: int or float
        Threshold cutoff for image

    Returns
    -------
    Plots labeled image to console, for 3-D image will only print first slice
    '''


    # utilize mask_label() function
    _, _, x = mask_label(image, thresh = thresh)
    plt.imshow(x[0], cmap = 'rainbow')
    plt.show()

#%%
def make_overlay_gif(three_D, overlay, save_as = None):
    '''Make gif of original image with labeled overlay
    
    Parameters
    ----------
    three_D: array-like
        3-D array of original image with time as first axis, if 2-D image
        add dummy time axis of length 1
    
    overlay: array-like
        3-D array of labels for original image, must be same dimensions as 
        three_D
    save_as: string
        Default None, file path to save gif as. Must be specified.

    Returns
    -------
    Saves resulting image to gif in given directory
    '''

    if save_as is None:
        raise ValueError("Must enter target string")
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

    if not save_as.endswith(".gif"):
        title = save_as + ".gif"
    else:
        title = save_as

    ani.save(title, writer = gifwriter)

# %%
def find_boxes(labeled_threeD, x_idx, y_idx):
    '''Mask specific region of interest from labeled image
    
    Parameters
    ----------
    labeled_threeD: array-like
        3-D array of timeseries labeled images where axis 0 is time and axis 1 and 2
        are the image. If 2-D image make dummy axis 0 of length 1

    x_idx: int
        Index along x-axis where field of interest is, axis 2 of 3-D array

    y_idx: int
        Index along y-axis where field of interest is, axis 1 of 3-D array

    Find x and y idx with plot_image() or imshow() function and manually
    locate region of interest

    Returns
    -------
    bboxes: list-like
        3-D list of tuples of bounding box slices at each time slice in the
        3-D image. Access slice by indexing into time series then into
        tuple item, e.g. bboxes[0][0] for slices in first image in
        time series. Plot slice with plot_image(image[i][bboxes[i][0]])

    lv_mask_all: array-like
        3-D array of dimensions of input array with mask including
        only the region of interest

    box_lv: list-like
        List of found label at each time slice in the 3-D image

    '''
    bboxes = []
    box_lv = []
    lv_mask_all = np.empty((labeled_threeD.shape[0], 
                      labeled_threeD.shape[1], 
                      labeled_threeD.shape[2]))
    for i in range(labeled_threeD.shape[0]):
        lv_l = labeled_threeD[i][y_idx,x_idx]
        box_lv.append(lv_l)
        lv_mask = np.where(labeled_threeD[i] == lv_l, 1,0)
        lv_mask_all[i] = lv_mask
        bboxes.append( ndi.find_objects(lv_mask))
    return bboxes, lv_mask_all, box_lv

# %%
def calc_ejtc(orig, lv_mask):
    '''Calculate the ejection fraction of ventrical for 3-D time series

    Parameters
    ----------
    orig: image.core.util.Array
        Original DICOM image with attached metadata

    lv_mask: array-like
        3-D array of same dimensions as "orig" with region of interest
        masked.

    Returns
    -------
    edjfract: float
        Normalized difference between maximum and minimum size of 
        masked object of interest in the image

    ts: array-like
        Size of object at each time slice

    For the current images this is a 2-D area calculation since the image
    is of a single depth slice over time, for 3_D volume would need
    image with 4 dimensional attributes, time-depth-x-y.
    '''

    # get sampling distances
    d0, d1, d2 = orig.meta.sampling

    # calculate area of each pixel
    dvoxel = d1*d2

    ts = np.zeros(orig.shape[0])

    # iterate over images, sum volume in each
    for t in range(orig.shape[0]):
        nvoxel = ndi.sum(1, 
                         lv_mask[t])
        ts[t] = nvoxel * dvoxel # number pixels times area per pixel
    
    # plot graph of change in area over time
    plt.plot(ts)
    plt.show()
    edjfract = (ts.max() - ts.min())/ts.max()
    
    return edjfract, ts
# %%
