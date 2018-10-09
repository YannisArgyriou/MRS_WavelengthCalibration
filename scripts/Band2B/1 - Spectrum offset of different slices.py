
# coding: utf-8

# # Spectrum offset of different slices
# Script written by Ioannis Argyriou (Institute of Astronomy, KU Leuven)  
# Creation date: October 9th 2018 
# 
# This notebook computes the pixel offset of the dichroic transmission, seen by the MRS, in the different IFU slices.

# ### > Import modules

# In[1]:

import funcs
import mrsobs
from distortionMaps import d2cMapping

import numpy as np
from datetime import date
from astropy.io import fits
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib notebook')

import warnings
warnings.simplefilter('ignore')


# ### > Set directories

# In[2]:

# USER:
user = "yannis" # "yannis"

# Set work directory
# *** Change these to more general folders like"./data_for_notebook_test/" that we can all have
if user == "alvaro":
    workDir = "./"
    
if user == "yannis":
    workDir  = "/Users/ioannisa/Desktop/python/miri_devel/"

cdpDir   = workDir+"cdp_data/"
lvl2path = workDir+"FM_data/LVL2/"


# ### > Give analysis inputs

# In[6]:

# inputs
band = '2B'                     # spectral band under investigation
ref_slice = 10                  # slice number (equivalent to beta position)
alpha_pos = 0.                  # along-slice position, [arcsec]


# ### > Import MRS observations

# In[7]:

# top: xconfig 2BxC image, bottom: nominal 1B/2B detector image
cross_config = mrsobs.MIRI_internal_calibration_source(lvl2path,'2BxC',campaign='FM',output='img')
nomin_config = mrsobs.MIRI_internal_calibration_source(lvl2path,'2B',campaign='FM',output='img')
dichroic_transmission_img = cross_config/nomin_config

# plot
fig,axs = plt.subplots(1,3,figsize=(12,4))
axs[0].imshow(cross_config)
axs[1].imshow(nomin_config)
axs[2].imshow(dichroic_transmission_img,vmin=0,vmax=1.2)
for plot in range(3):
    axs[plot].set_xlabel('Detector x-coordinate [pixel]',fontsize=12)
    axs[plot].set_ylabel('Detector y-coordinate [pixel]',fontsize=12)
    axs[plot].tick_params(axis='both',labelsize=12)
axs[0].set_title('2AxB observation',fontsize=12)
axs[1].set_title('1A/2A observation',fontsize=12)
axs[2].set_title('Dichroic transmission',fontsize=12)
plt.tight_layout()


# ### > Determine spectrum offset between the central trace of each MRS IFU slice

# In[8]:

# load distortion maps
d2cMaps   = d2cMapping(band,cdpDir)
sliceMap  = d2cMaps['sliceMap']
lambdaMap = d2cMaps['lambdaMap']
alphaMap  = d2cMaps['alphaMap']
nslices   = d2cMaps['nslices']
det_dims  = (1024,1032)


# In[9]:

plt.close('all')
ypos_ref,xpos_ref = funcs.detpixel_trace(band,d2cMaps,sliceID=ref_slice,alpha_pos=alpha_pos)
sci_fm_data_ref = dichroic_transmission_img[ypos_ref,xpos_ref][10:-10]

# create a finer grid
step = 0.2
fine_grid = np.arange(10,1013+step,step)
sci_fm_data_ref_fine = interp1d(10+np.arange(len(sci_fm_data_ref)),sci_fm_data_ref)(fine_grid)
# sci_fm_data_ref_fine = savgol_filter(sci_fm_data_ref_fine,21,3)

pix_offsets = []
offsets = np.arange(1,200)
wider_offsets = np.arange(-200,200)
plot = True
for islice in range(1,nslices+1):
# for islice in [13]:
    if islice == 10:
        pix_offset = 0.
        pix_offsets.append(round(pix_offset,2))
        continue
    ypos,xpos = funcs.detpixel_trace(band,d2cMaps,sliceID=islice,alpha_pos=alpha_pos)
    sci_fm_data = dichroic_transmission_img[ypos,xpos][10:-10]

    sci_fm_data_fine = interp1d(10+np.arange(len(sci_fm_data)),sci_fm_data)(fine_grid)
#     sci_fm_data_fine = savgol_filter(sci_fm_data_fine,21,3)
    
    if (lambdaMap[ypos_ref,xpos_ref][len(ypos)/2] - lambdaMap[ypos,xpos][len(ypos)/2]) > 0. :
        residuals = []
        for offset in offsets:
            residuals.append(np.sum(((sci_fm_data_fine[offset:]-sci_fm_data_ref_fine[:-offset])[~np.isnan(sci_fm_data_fine[offset:]-sci_fm_data_ref_fine[:-offset])])[:-300]**2))
        residuals = np.array(residuals)

        popt     = np.polyfit(offsets,residuals,3)
        poly     = np.poly1d(popt)
        
        pix_offset = wider_offsets[np.argmin(poly(wider_offsets))]*step
        
        if plot is True:
            fig,axs = plt.subplots(1,3,figsize=(12,4))
            axs[0].plot(fine_grid,sci_fm_data_fine,label='Slice {}'.format(islice))
            axs[0].plot(fine_grid,sci_fm_data_ref_fine,label='Ref slice: {}'.format(ref_slice))
            axs[1].plot(offsets*step,residuals,'bo')
            axs[1].plot(wider_offsets*step,poly(wider_offsets),'r')
            axs[1].vlines(pix_offset,-3,5,'k',linestyle='dashed')
            axs[2].plot(fine_grid[:len(sci_fm_data_fine[np.argmin(poly(offsets)):])],sci_fm_data_fine[np.argmin(poly(offsets)):],label='Slice {}'.format(islice))
            axs[2].plot(fine_grid,sci_fm_data_ref_fine,label='Ref slice: {}'.format(ref_slice))
            for iplot in [0,2]:
                axs[iplot].set_xlabel('X-coordinate [pix]')
                axs[iplot].set_ylabel('Transmission')
                axs[iplot].legend(loc='lower right')
            axs[1].set_xlabel('Pixel offset [pix]')
            axs[1].set_ylabel(r'$\chi$$^2$')
            plt.tight_layout()
    elif (lambdaMap[ypos_ref,xpos_ref][len(ypos)/2] - lambdaMap[ypos,xpos][len(ypos)/2]) < 0. :
        residuals = []
        for offset in offsets:
            residuals.append(np.sum(((sci_fm_data_ref_fine[offset:]-sci_fm_data_fine[:-offset])[~np.isnan(sci_fm_data_ref_fine[offset:]-sci_fm_data_fine[:-offset])])[:-300]**2))
        residuals = np.array(residuals)

        popt     = np.polyfit(offsets,residuals,3)
        poly     = np.poly1d(popt)
        
        pix_offset = wider_offsets[np.argmin(poly(wider_offsets))]*step
        
        if plot is True:
            fig,axs = plt.subplots(1,2,figsize=(12,4))
            axs[0].plot(fine_grid,sci_fm_data_fine,label='Slice {}'.format(islice))
            axs[0].plot(fine_grid,sci_fm_data_ref_fine,label='Ref slice: {}'.format(ref_slice))
            axs[1].plot(offsets*step,residuals,'bo')
            axs[1].plot(wider_offsets*step,poly(wider_offsets),'r')
            axs[1].vlines(pix_offset,-3,5,'k',linestyle='dashed')
#             axs[2].plot(fine_grid[:len(sci_fm_data_fine[np.argmin(poly(wider_offsets)):])],sci_fm_data_fine[np.argmin(poly(wider_offsets)):],label='Slice {}'.format(islice))
#             axs[2].plot(fine_grid,sci_fm_data_ref_fine,label='Ref slice: {}'.format(ref_slice))
            for iplot in [0]:
                axs[iplot].set_xlabel('X-coordinate [pix]')
                axs[iplot].set_ylabel('Transmission')
                axs[iplot].legend(loc='lower right')
            axs[1].set_xlabel('Pixel offset [pix]')
            axs[1].set_ylabel(r'$\chi$$^2$')
            plt.tight_layout()
    pix_offsets.append(round(pix_offset,2))
    print 'Slice {}'.format(islice)
    print 'Pixel offset: {}pix'.format(pix_offsets[islice-1])


# In[12]:

# save output
save_file = open('data/Band'+str(band)+'_Dichroic_refslice'+str(ref_slice)+'_alpha'+str(alpha_pos)+'_reloffsets.txt', 'w')
save_file.write('# Band '+str(band)+'   Dichroic   RefSlice '+str(ref_slice)+'\n')
save_file.write('# Distortion map version 06.04.00 - date '+str(date.today())+'\n')
save_file.write('# Trace (isoalpha): Take pixel trace along specified slice, specified alpha position trace is built by taking the pixel in every detector row with alpha value closest to the one specified \n')
save_file.write('# xpos[i] = np.argmin(alpha_img[i,:])+funcs.find_nearest(alpha_img[i,:][(slice_img[i,:]!=0)],alpha_pos)\n')
save_file.write('# Band'+str(band)+'_Dichroic_refslice'+str(ref_slice)+'_alpha'+str(alpha_pos)+'\n')
save_file.write('\n')
save_file.write('# Slice Nr.  Pixel offset (pix)\n')

for zzz in range(1,nslices+1):
    save_file.write(str(zzz)+'  '+str(pix_offsets[zzz-1])+'\n')
save_file.close()


# In[ ]:



