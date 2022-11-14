# biomed_imaging
A series of functions to import, transform, manipulate, and plot DICOM medical imaging files. This repository uses MRIs of the heart and isolates the left ventrical to calculate ejection fraction, but the functions are generalized enough to other manual image segmentation processes for working with time series images. 

Data was sourced from the [Biomedical Image Analysis in Python](https://app.datacamp.com/learn/courses/biomedical-image-analysis-in-python) course on DataCamp, which consisted of a single patient's data from the [Cardiac Atlas Project](http://www.cardiacatlas.org/studies/sunnybrook-cardiac-data/). Follow data camp link to access specific data used in this script, or download the zip file directly [here](https://assets.datacamp.com/production/repositories/2085/datasets/fabaa1f1675549d624eb8f5d1bc94e0b11e30a8e/sunnybrook-cardiac-mr.zip). Most of the workflow was also adapted from that tutorial, though combined into more generalized functions.

Examples of output plots for one of the time series images is shown below:

<p align="center">
 <img src="/DCM2001_005.gif" height="300" width="460"/>
    <br>
    <em>Figure 1: Gif of contrast highlighted heart image</em>
</p>

<p align="center">
 <img src="/overlaygif_05.gif" height="300" width="460"/>
    <br>
    <em>Figure 2: Gif of label overlay on heart image highlighting regions within the heart</em>
</p>

