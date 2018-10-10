
# coding: utf-8

# # MIRI FM etalon line identification
# 
# This notebook is used to save all fitted centers and properties of fitted etalon lines observed with the MIRI MRS.

# In[1]:

# import modules
import funcs
import mrsobs
from distortionMaps import d2cMapping

import numpy as np
import pandas as pd
from datetime import date
import scipy.interpolate as scp_interpolate
from scipy.optimize import curve_fit
from astropy.io import fits

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib notebook')


# In[2]:

# > Set directories
user = "yannis" # "yannis"
if user == "alvaro":
    workDir = "./"
if user == "yannis":
    workDir  = "/Users/ioannisa/Desktop/python/miri_devel/"
cdpDir   = workDir+"cdp_data/"
lvl2path = workDir+"FM_data/LVL2/"
MRSWaveCalDir = workDir+"MRSWaveCal/"
FTSlinefits   = MRSWaveCalDir+"FTS_ET_linefits/"


# In[3]:

# give analysis inputs
band = '2C'                     # spectral band under investigation

# load distortion maps
d2cMaps   = d2cMapping(band,cdpDir)
sliceMap  = d2cMaps['sliceMap']
lambdaMap = d2cMaps['lambdaMap']
alphaMap  = d2cMaps['alphaMap']
nslices   = d2cMaps['nslices']
det_dims  = (1024,1032)

# To account for the problematic lines at the top of the distorion maps
## WARNING: This should only be done for band 1C (and only if the distortion CDP version 06.04.00 is used, not the CDP7 version).
if band == "1C":
    d2cMaps['sliceMap'][0,:] = d2cMaps['sliceMap'][1,:]
    d2cMaps['alphaMap'][0,:] = d2cMaps['alphaMap'][1,:]
    d2cMaps['sliceMap'][-1,:] = d2cMaps['sliceMap'][-2,:]
    d2cMaps['alphaMap'][-1,:] = d2cMaps['alphaMap'][-2,:]


# In[4]:

# Load etalon MRS and RAL FTS data files and data, and subtract BKG
# Etalon_1A FM data
sci_etalon1A_img,bkg_etalon1A_img = mrsobs.FM_MTS_800K_BB_extended_source_through_etalon(lvl2path,band,etalon='ET1A',output='img')
# Etalon_1B FM data
sci_etalon1B_img,bkg_etalon1B_img = mrsobs.FM_MTS_800K_BB_extended_source_through_etalon(lvl2path,band,etalon='ET1B',output='img')

etalon1A_img = sci_etalon1A_img-bkg_etalon1A_img
etalon1B_img = sci_etalon1B_img-bkg_etalon1B_img


# ### Perform analysis for all alphas in all slices

# In[5]:

nslices = d2cMaps['nslices']

for islice in range(1,nslices+1):
    print "Slice {}".format(islice)
    # open files to store information
    save_et1a = open('data/Band'+str(band)+'_ET1A_slice'+str(islice)+'.txt', 'w')
    save_et1a.write('# Band '+str(band)+'   Etalon 1A   Slice '+str(islice)+'\n')
    save_et1a.write('# Distortion map version 06.04.00 - date '+str(date.today())+'\n')
    save_et1a.write('# Trace (isoalpha): Take pixel trace along specified slice, specified alpha position trace is built by taking the pixel in every detector row with alpha value closest to the one specified \n')
    save_et1a.write('# xpos[i] = np.argmin(alpha_img[i,:])+funcs.find_nearest(alpha_img[i,:][(slice_img[i,:]!=0)],alpha_pos)\n')
    save_et1a.write('#    alpha       x  y      center         FWHM          skewness\n')

    save_et1b = open('data/Band'+str(band)+'_ET1B_slice'+str(islice)+'.txt', 'w')
    save_et1b.write('# Band '+str(band)+'   Etalon 1B   Slice '+str(islice)+'\n')
    save_et1b.write('# Distortion map version 06.04.00 - date '+str(date.today())+'\n')
    save_et1b.write('# Trace (isoalpha): Take pixel trace along specified slice, specified alpha position trace is built by taking the pixel in every detector row with alpha value closest to the one specified \n')
    save_et1b.write('# xpos[i] = np.argmin(alpha_img[i,:])+funcs.find_nearest(alpha_img[i,:][(slice_img[i,:]!=0)],alpha_pos)\n')
    save_et1b.write('#    alpha       x  y      center         FWHM          skewness\n')
    
    # Traces for each isoalpha are identified based on the following criteria:
    alpha_high,alpha_low,thres_e1a,min_dist_e1a,sigma0_e1a,thres_e1b,min_dist_e1b,sigma0_e1b = funcs.etalon_line_params(band)
    
    # list of alpha positions to fill slice
    alphas_inslice = funcs.slice_alphapositions(band,d2cMaps,sliceID=islice)
    alphas_inslice = np.append(0,alphas_inslice)  # To include alpha=0, and put it at the beginning of the array

    inds_al = np.where((alphas_inslice < alpha_high) & (alphas_inslice > alpha_low))  # we cut to this range to avoid issues with the edges of the slices.
    alphas_inslice = alphas_inslice[inds_al]

    for alpha_pos in alphas_inslice:
        if alpha_pos == alphas_inslice[0]:
            print 'Along-slice position:'
        print  '                     {} arcsec'.format(round(alpha_pos,2))
        # Take pixel trace along specified slice, specified alpha position trace is built by taking the pixel in every detector row with alpha value closest to the one specified
        ypos,xpos = funcs.detpixel_trace(band,d2cMaps,sliceID=islice,alpha_pos=alpha_pos)
        if band == '1C':
            xpos[0] = xpos[1]
            xpos[-1] = xpos[-2]

        # Choose data regions
        #--FM data
        etalon1A_fm_data = etalon1A_img[ypos,xpos]
        etalon1B_fm_data = etalon1B_img[ypos,xpos]

        # Determine etalon peaks
        #--FM Etalon_1A data
        if user == 'yannis':
            etalon1A_fm_data[np.isnan(etalon1A_fm_data)] = -1
            FMetalon1A_peaks = funcs.find_peaks(etalon1A_fm_data,thres=thres_e1a,min_dist=min_dist_e1a)
            FMetalon1A_peaks = FMetalon1A_peaks[(FMetalon1A_peaks>1) & (FMetalon1A_peaks<1022)]
        if user == 'alvaro':
            picos_1A, y_pixs_1A = funcs.find_max(etalon1A_fm_data,np.arange(1,1025,1.),maxcut_1a,toler_1a) # maxcut, wavel_toler
            picos_inds_1A = y_pixs_1A-1
            FMetalon1A_peaks = picos_inds_1A.astype(int)
        etalon1A_fm_data[(etalon1A_fm_data == -1)] = np.nan
        etalon1A_fm_data_noNaN = etalon1A_fm_data.copy()
        etalon1A_fm_data_noNaN[np.isnan(etalon1A_fm_data)] = 0.

        #--FM Etalon_1B data
        if user == 'yannis':
            etalon1B_fm_data[np.isnan(etalon1B_fm_data)] = -1
            FMetalon1B_peaks = funcs.find_peaks(etalon1B_fm_data,thres=thres_e1b,min_dist=min_dist_e1b)
            FMetalon1B_peaks = FMetalon1B_peaks[(FMetalon1B_peaks>1) & (FMetalon1B_peaks<1022)]
        if user == 'alvaro':
            picos_1B, y_pixs_1B = funcs.find_max(etalon1B_fm_data,np.arange(1,1025,1.),maxcut_1b,toler_1b) # maxcut, wavel_toler
            picos_inds_1B = y_pixs_1B-1
            FMetalon1B_peaks = picos_inds_1B.astype(int)
        etalon1B_fm_data[(etalon1B_fm_data == -1)] = np.nan
        etalon1B_fm_data_noNaN = etalon1B_fm_data.copy()
        etalon1B_fm_data_noNaN[np.isnan(etalon1B_fm_data)] = 0.

        # Fit Etalon_1A lines in FM data
        FMetalon1A_fitparams,FMetalon1A_fiterrors,ET1A_fitting_flag,range_ini,range_fin = funcs.fit_etalon_lines(np.arange(det_dims[0]),etalon1A_fm_data_noNaN,FMetalon1A_peaks,fit_func='skewed_voight',sigma0=sigma0_e1a,f0=0.5,a0=0.1)
        linecenter_ET1A = funcs.get_linecenter(FMetalon1A_fitparams,ET1A_fitting_flag)
        linefwhm_ET1A   = funcs.get_FWHM(FMetalon1A_fitparams,ET1A_fitting_flag)
        lineskew_ET1A   = funcs.get_skewness(FMetalon1A_fitparams,ET1A_fitting_flag)

        # Fit Etalon_1B lines in FM data
        FMetalon1B_fitparams,FMetalon1B_fiterrors,ET1B_fitting_flag,range_ini,range_fin = funcs.fit_etalon_lines(np.arange(det_dims[0]),etalon1B_fm_data_noNaN,FMetalon1B_peaks,fit_func='skewed_voight',sigma0=sigma0_e1b,f0=0.5,a0=0.1)
        linecenter_ET1B = funcs.get_linecenter(FMetalon1B_fitparams,ET1B_fitting_flag)
        linefwhm_ET1B   = funcs.get_FWHM(FMetalon1B_fitparams,ET1B_fitting_flag)
        lineskew_ET1B   = funcs.get_skewness(FMetalon1B_fitparams,ET1B_fitting_flag)

        # SAVES fits of FM - ETALON 1A
        for zzz in range(0,np.size(FMetalon1A_peaks),1):
            save_et1a.write(str(alpha_pos)+'  '+str(xpos[FMetalon1A_peaks[zzz]])+'  '+str(FMetalon1A_peaks[zzz])+'  '+str(linecenter_ET1A[zzz])+'  '+str(linefwhm_ET1A[zzz])+'  '+str(lineskew_ET1A[zzz])+'\n')

        # SALVA fits of FM - ETALON 1B
        for zzz in range(0,np.size(FMetalon1B_peaks),1):
            save_et1b.write(str(alpha_pos)+'  '+str(xpos[FMetalon1B_peaks[zzz]])+'  '+str(FMetalon1B_peaks[zzz])+'  '+str(linecenter_ET1B[zzz])+'  '+str(linefwhm_ET1B[zzz])+'  '+str(lineskew_ET1B[zzz])+'\n')

    save_et1a.close()
    save_et1b.close()


# In[ ]:



