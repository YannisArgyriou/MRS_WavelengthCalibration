
# coding: utf-8

# # MRS Band 1B ref point fit
# Script written by Ioannis Argyriou (Institute of Astronomy, KU Leuven), with updates from Alvaro Labiano (CAB)  
# Creation date 7th October 2018  
# 
# This notebook compare the fits of the reference point by Yannis and Alvaro

# ### > Import modules

# In[1]:

import funcs
import mrsobs
from distortionMaps import d2cMapping

import numpy as np
from scipy import signal
from datetime import date
import scipy.interpolate as scp_interpolate
from scipy.optimize import curve_fit
import scipy.special as sp
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
MRSWaveCalDir = workDir+"MRSWaveCal/"


# ### > Give analysis inputs

# In[3]:

# inputs
band = '1B'                     # spectral band under investigation
lamblower,lambupper = 5.64,6.67   # nominal wavelength range of band 1C
islice    = 10                  # slice number (equivalent to beta position)
alpha_pos = 0.                  # along-slice position, [arcsec]


# ## Start analysis

# A reference wavelength/pixel pair is determined using the long-wave-pass (SWP) filter transmission. By comparing the transmission of the filter determined from the lab and with the MRS, a reference wavelength can be matched to a reference pixel on the MRS detector. Let's look at the lab and MRS SWP filter transmission.
# 
# During commissioning the reference point will be obtained from the spectral lines of a celestial calibrator.

# ### > Import filter transmission data

# In[4]:

# Read the measured transmission curves from the dat files
# first colum is wavelength [micrometer]
# second and third columns are room temperature transmissions
# fourth column is 35K transmission
SWPwvnr,SWPtransm = np.genfromtxt(MRSWaveCalDir + "swp_filter.txt", skip_header = 15, skip_footer=1, usecols=(0,3), delimiter = '',unpack='True')
SWPwave = 10000./SWPwvnr
SWPtransm = SWPtransm/100. # convert percentage to decimal
sel = (SWPwave>=lamblower) & (SWPwave<=lambupper)

# plot transmissions
plt.figure(figsize=(12,6))
plt.plot(SWPwave[sel],SWPtransm[sel])
plt.xlim(lamblower,lambupper)
plt.ylim(0.65,0.825)
plt.title('SWP filter transmission (lab data)',fontsize=20)
plt.xlabel('Wavelength [micron]',fontsize=20)
plt.ylabel('Transmission',fontsize=20)
plt.tick_params(axis='both',labelsize=20)
plt.tight_layout()


# ## > Import MRS observations of filter

# In[5]:

# Import MRS observations
# -->MRS detector images processed with DHAS 9.4.4 miri_sloper only
# swp_filter_img: SWP filter extended obs (SWP transm x 800K BB), ext_source_img: 800K BB extended source config

swp_filter_img,ext_source_img,bkg_img = mrsobs.FM_MTS_800K_BB_MRS_OPT_08(lvl2path,wp_filter='SWP',output='img')
swp_transmission_img = (swp_filter_img-bkg_img)/(ext_source_img-bkg_img)

# plot
fig,axs = plt.subplots(1,3,figsize=(12,4))
axs[0].imshow(swp_filter_img)
axs[1].imshow(ext_source_img)
axs[2].imshow(swp_transmission_img,vmin=0,vmax=1.4)
for plot in range(3):
    axs[plot].set_xlabel('Detector x-coordinate [pixel]',fontsize=12)
    axs[plot].set_ylabel('Detector y-coordinate [pixel]',fontsize=12)
    axs[plot].tick_params(axis='both',labelsize=12)
axs[0].set_title('SWP filter observation',fontsize=12)
axs[1].set_title('Extended source observation',fontsize=12)
axs[2].set_title('SWP filter transmission',fontsize=12)
plt.tight_layout()


# ## > Perform wavelength calibration analysis on selected position

# In[6]:

# load distortion maps
d2cMaps   = d2cMapping(band,cdpDir)
sliceMap  = d2cMaps['sliceMap']
lambdaMap = d2cMaps['lambdaMap']
alphaMap  = d2cMaps['alphaMap']
nslices   = d2cMaps['nslices']
det_dims  = (1024,1032)


# In[7]:

# take pixel trace along specified slice, specified alpha position
## ******* AL: Shall we use interpolation between closest alpha values? <-- This should result in a pixel to sub-pixel order improvement on the determination of the wavelength-pixel reference
ypos,xpos = funcs.detpixel_trace(band,d2cMaps,sliceID=islice,alpha_pos=alpha_pos)

# plot trace
valid_img = np.zeros(det_dims)
valid_img[ypos,xpos] = 1.

fig,axs = plt.subplots(1,2,figsize=(12,4))
axs[0].imshow(sliceMap,cmap='gray')
axs[0].imshow(valid_img,alpha=0.6,cmap='gray')
axs[0].set_title('Pixel trace through MRS slice',fontsize=12)
axs[0].set_xlabel('Detector x-coordinate [pixel]',fontsize=12)
axs[0].set_ylabel('Detector y-coordinate [pixel]',fontsize=12)
axs[1].plot(swp_transmission_img[ypos,xpos])
axs[1].set_title('SWP filter transmission (MRS data)',fontsize=12)
axs[1].set_xlabel('Detector y-coordinate [pix]',fontsize=12)
axs[1].set_ylabel('Transmission',fontsize=12)
for plot in range(2): axs[plot].tick_params(axis='both',labelsize=12)
plt.tight_layout()


# # > Yannis fit (spline) of the reference point

# Below we compare the lab and MRS determined filter transmissions. Since the steep gradient part of the transmission in the lab data shows erratic changes of slope (compared to the MRS data), rather than defining a cut-off on the steep gradient part of the curve, we determine the reference wavelength/pixel pair at the location where the filter transmission flattens out, i.e. where the gradient of the curve is zero. A spline is fitted through the MRS data to remove the small amplitude high-frequency noise in the data.  
#   
# The result of the following analysis is the definition of a reference wavelength/pixel pair.

# In[8]:

### > Yannis fit (spline) of the reference point
# load spectrum from desired location and carry-out analysis
sci_fm_data = swp_transmission_img[ypos,xpos]
sci_fm_data[np.isnan(sci_fm_data)] = 0

fig,axs = plt.subplots(1,2,figsize=(12,5))
axs[0].plot(SWPwave[sel],SWPtransm[sel])
axs[0].set_xlim(lamblower,lambupper)
axs[0].set_xlabel('Wavelength [micron]',fontsize=12)
axs[0].set_ylabel('Transmission',fontsize=12)
axs[0].set_title('SWP filter transmission (lab data)',fontsize=12)
axs[1].plot(sci_fm_data[np.nonzero(sci_fm_data)])
axs[1].plot(signal.savgol_filter(sci_fm_data[np.nonzero(sci_fm_data)],21,3),'r')
axs[1].set_xlim(-40,1064)
axs[1].set_xlabel('Y-coordinate [pix]',fontsize=20)
axs[1].set_title('SWP filter transmission (MRS data)',fontsize=12)
for plot in range(2): axs[plot].tick_params(axis='both',labelsize=12)
plt.tight_layout()

SWPgrad = np.gradient(SWPtransm[sel],SWPwave[sel])
SWPsigns = np.sign(SWPgrad)
SWPsigns[:funcs.find_nearest(SWPwave[sel],6.44)] = 0

sci_fm_data_grad = np.gradient(signal.savgol_filter(sci_fm_data[np.nonzero(sci_fm_data)],21,3))
sci_fm_data_signs = np.sign(sci_fm_data_grad)
sci_fm_data_signs[:800] = 0

SWPzerocrossing = np.where(np.abs(np.diff(SWPsigns)) == 2)[0]
sci_fm_data_zerocrossing = np.where(np.abs(np.diff(sci_fm_data_signs)) == 2)[0]

x0 = SWPwave[sel][SWPzerocrossing[0]]
x1 = SWPwave[sel][SWPzerocrossing[0]+1]
y0 = SWPgrad[SWPzerocrossing[0]]
y1 = SWPgrad[SWPzerocrossing[0]+1]
a = (y1-y0)/(x1-x0)
cutofflamb = (-y0/a) + x0

x0 = sci_fm_data_zerocrossing[0]
x1 = np.arange(len(sci_fm_data))[sci_fm_data_zerocrossing[0]+1]
y0 = sci_fm_data_grad[sci_fm_data_zerocrossing[0]]
y1 = sci_fm_data_grad[sci_fm_data_zerocrossing[0]+1]
a = (y1-y0)/(x1-x0)
cutoffpix = (-y0/a) + x0

# plot transmissions
fig,axs = plt.subplots(2,1,figsize=(12,10))
axs[0].plot(SWPwave[sel],np.gradient(SWPtransm[sel],SWPwave[sel]),'b')
axs[0].hlines(0,lamblower,lambupper)
axs[0].vlines(cutofflamb,-0.001,0.004,linestyle='dashed',label='reference point')
axs[0].set_xlim(lamblower,lambupper)
axs[1].plot(np.gradient(signal.savgol_filter(sci_fm_data[np.nonzero(sci_fm_data)],21,3)),'r')
axs[1].hlines(0,0,1024)
axs[1].vlines(cutoffpix,-0.004,0.008,linestyle='dashed',label='reference point')
axs[1].set_xlim(0,1023)
axs[0].set_xlabel('Wavelength [micron]',fontsize=12)
axs[0].set_ylabel('INTA SWP transm gradient',fontsize=12)
axs[1].set_xlabel('Y-coordinate [pix]',fontsize=12)
axs[1].set_ylabel('RAL SWP transm gradient',fontsize=12)
for plot in range(2):
    axs[plot].tick_params(axis='both',labelsize=12)
    axs[plot].legend(loc='upper right',fontsize=12)
plt.tight_layout()

print 'Reference wavelength: {} micron'.format(round(cutofflamb,3))
print 'Reference pixel: {} pix'.format(round(cutoffpix,3))


# ### > Alvaro's fit of the reference point

# In[9]:

## BASED ON 1C_comparafiltros_lam_to_coordy_allcols.py

from scipy import interpolate
from scipy import optimize
from scipy import stats

normaliz = 1 # 100    
guess = [150, 800, 1.03]
bound_val = 0 #np.nan

# Fit region
fmpixlims = [5,400]
fmwavlims = [6.515,6.9]  #values used for original wavecal, using zemax wavelengths

intawavlims = [fmwavlims[0]-0.08,fmwavlims[1]-0.05]
# Fit centers
lam_ref_val = 6.626905235  #(taken from SWP_INTA.dat) for 0.45815491
flx_ref_val = 0.45815491

### FM data
sci_fm_data = Swp_transmission_img[ypos,xpos]
#sci_fm_data[np.isnan(sci_fm_data)] = 0   ## NANS????
yFM_orig = sci_fm_data
xFM_pixl = np.array(range(1,yFM_orig.size+1))        # In pixels
# print sci_fm_data.shape, yFM_orig.shape, xFM_pixl.shape

# Cut the spectra to the region of interest:
inds_xFM = np.arange(fmpixlims[0],fmpixlims[1])   #inds_xFM = np.where((xFM_orig > fmwavlims[0]) & (xFM_orig < fmwavlims[1]))
xFM_cut = xFM_pixl[inds_xFM]
yFM_cut = yFM_orig[inds_xFM]

# Use only rows of FM where flux is NOT nan:
nans_yfm = np.isnan(yFM_cut)
inds_yfm = np.where(nans_yfm == False)
xFM = xFM_cut[inds_yfm]
yFM = yFM_cut[inds_yfm]


### INTA data
wINTA_orig = SWPwave
yINTA_orig = SWPtransm
inds_winta = np.where((wINTA_orig > intawavlims[0]) & (wINTA_orig < intawavlims[1]))
wINTA = wINTA_orig[inds_winta]
yINTA = yINTA_orig[inds_winta]             


yINTA = yINTA/normaliz

# Plot original data
fig,axs = plt.subplots(1,2,figsize=(12,6))
axs[0].plot(wINTA,yINTA)
axs[0].set_xlim(intawavlims[0]-0.1,intawavlims[1]+0.1)
axs[0].tick_params(axis='both',labelsize=20)
axs[0].set_xlabel('Wavelength [micron]',fontsize=20)
axs[0].set_ylabel('Transmission',fontsize=20)
axs[0].set_title('SWP - INTA data')
axs[1].plot(xFM,yFM)
#axs[1].plot(spl(np.arange(len(sci_fm_data))),'r')
axs[1].set_xlim(fmpixlims[0]-10,fmpixlims[1]+10)
axs[1].set_xlabel('Y-coordinate [pix]',fontsize=20)
axs[1].tick_params(axis='both',labelsize=20)
axs[1].set_title('SWP - FM data')
plt.tight_layout()


## Fits one curve with the other
def FM_T2(zz,a,b,c):  # analoga a f2, en su ejemplo
    xINTAp = a + (wINTA - lam_ref_val) * b  # nuevo eje de lambdas del FM  
#     print a,b, c
    #print "xFMp=" ,xFMp
    yINTAinterpol = interpolate.interp1d(xINTAp,yINTA,bounds_error=False,fill_value=bound_val,kind='cubic')
    yINTAp = yINTAinterpol(zz) 
#     print zz.size, xINTAp.size, yINTAp.size # print size(zz), size(xINTAp), size(yINTAp)
#     print "MINFM, MAXFM, MININTA, MAXINTA =", min(xFM), max(xFM), min(wINTA), max(wINTA) #, ifusli+xcol   ######
    y_fut =  yINTAp * c # +d
    return y_fut        

                    
params, params_covariance = optimize.curve_fit(FM_T2,xFM, yFM, guess)
print "covar", params_covariance
if str(params_covariance) == 'inf':
    print "params", params[0], '+/-', str(params_covariance), params[1], '+/-', str(params_covariance), params[2], '+/-', str(params_covariance)
    params_stddev = ['inf','inf','inf']
    tit2 = 'inf'
    tit4 = 'inf'
    tit6 = 'inf'
else:
    print "params", params[0], '+/-', params_covariance[0,0]**0.5, params[1], '+/-', params_covariance[1,1]**0.5, params[2], '+/-', params_covariance[2,2]**0.5
    # matrix of fit covariances, (the diagonals of which are the squares of the 1 sigma uncertainties on the fit coefficients)
    params_stddev = [params_covariance[0,0]**0.5,params_covariance[1,1]**0.5, params_covariance[2,2]**0.5]
    tit2 = str("%.5f" % params_stddev[0])
    tit4 = str("%.3e" % params_stddev[1])
    tit6 = str("%.5f" % params_stddev[2])

new_xINTA = params[0] + params[1] * (wINTA - lam_ref_val )
new_yINTA = yINTA * params[2]

print params[0] + params[1] * (np.array([6.559527714,6.626905235 ,6.677796327]) - lam_ref_val) #+ params[3] * (np.array([7.65,8.06,8.17]) - lam_ref_val)**2. 
print params[2]*(np.array([0.201256,0.45815491,0.70766796]))


# In[10]:

#Plot result 
plot_xlims = [-50,350]
plot_ylims = [0,1]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xFM, yFM, 'b-', label="FM data")
ax.plot(guess[0] + guess[1] * (wINTA - lam_ref_val), yINTA*guess[2], 'g', ls='dotted', label="Initial guess")
ax.plot(xFM, FM_T2(xFM, *params), 'r', label="resulting fit")
#ax.set_title(tit0)
ax.set_xlim(plot_xlims[0],plot_xlims[1])
ax.set_ylim(plot_ylims[0],plot_ylims[1])
ax.set_yticks(np.arange(0,11)*0.1)
ax.legend(loc=4)
ax.set_xlabel('Y-pixel')
ax.set_ylabel('Throughput')

print "\n"
print lam_ref_val, params[0], "Alvaro's reference point"
pix_INTA = params[0] + params[1] * (wINTA - lam_ref_val )
print pix_INTA[169],wINTA[169], "Yannis reference pixel with Alvaro's conversion"
pix_Yannis = params[0] + params[1] * (cutofflamb - lam_ref_val )
print cutofflamb, pix_Yannis, "Yannis reference wavelength using Alvaro's conversion"


# In[9]:

print 'YANNIS Reference wavelength: {} micron'.format(round(cutofflamb,3))
print 'YANNIS Reference pixel: {} pix'.format(round(cutoffpix,3))
print 'The result seems very senstive to spl.set_smoothing_factor'
print 'using spl.set_smoothing_factor(0.0005) gives results similar to Alvaros conversion'

yan_ref = np.array([cutofflamb,cutoffpix])
# alo_ref = np.array([lam_ref_val, params[0]])
print yan_ref
# print alo_ref


# In[10]:

# save output
save_file = open('data/Band'+str(band)+'_SWP_refslice'+str(islice)+'_alpha'+str(alpha_pos)+'_refpoint_'+user+'.txt', 'w')
save_file.write('# Band '+str(band)+'   Filter SWP   RefSlice '+str(islice)+'\n')
save_file.write('# Distortion map version 06.04.00 - date '+str(date.today())+'\n')
save_file.write('# Trace (isoalpha): Take pixel trace along specified slice, specified alpha position trace is built by taking the pixel in every detector row with alpha value closest to the one specified \n')
save_file.write('# xpos[i] = np.argmin(alpha_img[i,:])+funcs.find_nearest(alpha_img[i,:][(slice_img[i,:]!=0)],alpha_pos)\n')
save_file.write('# Band'+str(band)+'_SWP_refslice'+str(islice)+'_alpha'+str(alpha_pos)+'\n')
save_file.write('\n')
save_file.write('# Reference Wavelength  Reference pixel (pix)\n')

if user == 'yannis':
    save_file.write(str(yan_ref[0])+'  '+str(yan_ref[1]))
elif user == 'alvaro':
    save_file.write(str(alo_ref[0])+'  '+str(alo_ref[1]))
save_file.close()


# ## IDEA FOR UNCERTAINTIES:
# 
# Run mMontecarlo on Alvaro's fit (Can it be done for Yannis's too? 
# 
# The idea is to create many (1000?) curves varying the throughput and wavelength  (or pixel) data points within their error bars (half a pixel?), rerun the fit for all of them, and check the variations

# In[ ]:



