
# coding: utf-8

# # Update the DISTORTION CDP

# In[1]:

from distortionMaps import d2cMapping

import datetime
import numpy as np
from astropy.io import fits
import numpy.lib.recfunctions as rec
from scipy.interpolate import interp1d


# In[2]:

# directories

# USER:
user = "yannis" # "alvaro","yannis"
# Set work directory
if user == "alvaro":
    workDir = "./"
if user == "yannis":
    workDir  = "/Users/ioannisa/Desktop/python/miri_devel/"

datadir = "data/"
cdpDir  = workDir+"cdp_data/"
outDir  = cdpDir+"CDP7/"


# # Band 1A/2A

# In[4]:

miri_setup = 'IFUSHORT_12SHORT'
# load old distortion cdp fits file
old_distortion_cdp = fits.open(outDir+'MIRI_FM_MIR{}_DISTORTION_7B.05.00.fits'.format(miri_setup))

# change headers
old_distortion_cdp[0].header['DATE']     = datetime.datetime.utcnow().isoformat()
old_distortion_cdp[0].header['FILENAME'] = 'MIRI_FM_MIR{}_DISTORTION_7B.06.00.fits'.format(miri_setup)
old_distortion_cdp[0].header['VERSION']  = '7B.XX.XX'
old_distortion_cdp[0].header['ORIGFILE'] = 'MIRI_FM_MIR{}_DISTORTION_7B.05.00.fits'.format(miri_setup)
old_distortion_cdp[0].header.add_history("- updated wavelength extension Channel 1C -> DOCUMENT:")
old_distortion_cdp[0].header.add_history("- MIRI-TN-00004-ETH, Alvaro Labiano and Ioannis Argyriou")

# update the wavelength solution
for band in ['2A']:
    if band == '1A': etal='ET1A'
    elif band == '2A': etal='ET1B'
    
    # number of slices in band
    d2cMaps  = d2cMapping(band,cdpDir)
    sliceMap = d2cMaps['sliceMap']
    alphaMap = d2cMaps['alphaMap']
    nslices  = d2cMaps['nslices']
    det_dims  = (1024,1032)

    # Create arrays of variables
    VAR1 = np.zeros(nslices)
    for islice in range(1,nslices+1):
        # compute slice reference x-position
        alpha_img = np.zeros(det_dims)
        alpha_img[(sliceMap == 100*int(band[0])+islice)] = alphaMap[(sliceMap == 100*int(band[0])+islice)]
        x_coords = np.nonzero(alpha_img[512,:])[0]
        alphas = alpha_img[512,:][x_coords]
        xs = interp1d(alphas,x_coords)(0.)

        VAR1[islice-1] = xs

    VAR2 = {}
    for var in range(25):
        VAR2[str(var)] = np.zeros(nslices)
        for islice in range(1,nslices+1):
            # load new wavelength solution
            wavsolution_file   = 'data/Band'+str(band)+'_ET'+ etal[-2:] +'_slice'+str(islice)+'_coeffs.txt'
            slice_wavcoeffs = np.loadtxt(wavsolution_file,unpack=True, skiprows = 5)

            VAR2[str(var)][islice-1] = slice_wavcoeffs[var]

    Lambda_CHX_new = np.rec.array([VAR1,VAR2['0'],VAR2['1'],VAR2['2'],VAR2['3'],VAR2['4'],VAR2['5'],VAR2['6'],
                  VAR2['7'],VAR2['8'],VAR2['9'],VAR2['10'],VAR2['11'],VAR2['12'],VAR2['13'],VAR2['14'],
                  VAR2['15'],VAR2['16'],VAR2['17'],VAR2['18'],VAR2['19'],VAR2['20'],VAR2['21'],VAR2['22'],VAR2['23'],VAR2['24']],
                  dtype=[('VAR1', 'float64'), ('VAR2_0_0', 'float64'), ('VAR2_0_1', 'float64'), ('VAR2_0_2', 'float64'), ('VAR2_0_3', 'float64'), ('VAR2_0_4', 'float64'), 
                         ('VAR2_1_0', 'float64'), ('VAR2_1_1', 'float64'), ('VAR2_1_2', 'float64'), ('VAR2_1_3', 'float64'), ('VAR2_1_4', 'float64'),
                         ('VAR2_2_0', 'float64'), ('VAR2_2_1', 'float64'), ('VAR2_2_2', 'float64'), ('VAR2_2_3', 'float64'), ('VAR2_2_4', 'float64'),
                         ('VAR2_3_0', 'float64'), ('VAR2_3_1', 'float64'), ('VAR2_3_2', 'float64'), ('VAR2_3_3', 'float64'), ('VAR2_3_4', 'float64'),
                         ('VAR2_4_0', 'float64'), ('VAR2_4_1', 'float64'), ('VAR2_4_2', 'float64'), ('VAR2_4_3', 'float64'), ('VAR2_4_4', 'float64')])

    # update corresponding fits extensions
    old_distortion_cdp['Lambda_CH{}'.format(band[0])].data = Lambda_CHX_new

    # save output
    old_distortion_cdp.writeto(outDir + "MIRI_FM_MIR{}_DISTORTION_7B.06.00.fits".format(miri_setup),overwrite=True)


# # Band 1B/2B

# In[5]:

miri_setup = 'IFUSHORT_12MEDIUM'
# load old distortion cdp fits file
old_distortion_cdp = fits.open(outDir+'MIRI_FM_MIR{}_DISTORTION_7B.05.00.fits'.format(miri_setup))

# change headers
old_distortion_cdp[0].header['DATE']     = datetime.datetime.utcnow().isoformat()
old_distortion_cdp[0].header['FILENAME'] = 'MIRI_FM_MIR{}_DISTORTION_7B.06.00.fits'.format(miri_setup)
old_distortion_cdp[0].header['VERSION']  = '7B.XX.XX'
old_distortion_cdp[0].header['ORIGFILE'] = 'MIRI_FM_MIR{}_DISTORTION_7B.05.00.fits'.format(miri_setup)
old_distortion_cdp[0].header.add_history("- updated wavelength extension Channel 1C -> DOCUMENT:")
old_distortion_cdp[0].header.add_history("- MIRI-TN-00004-ETH, Alvaro Labiano and Ioannis Argyriou")

# update the wavelength solution
for band in ['1B','2B']:
    if band == '1B': etal='ET1A'
    elif band == '2B': etal='ET1B'
    
    # number of slices in band
    d2cMaps  = d2cMapping(band,cdpDir)
    sliceMap = d2cMaps['sliceMap']
    alphaMap = d2cMaps['alphaMap']
    nslices  = d2cMaps['nslices']

    # Create arrays of variables
    VAR1 = np.zeros(nslices)
    for islice in range(1,nslices+1):
        # compute slice reference x-position
        alpha_img = np.zeros(det_dims)
        alpha_img[(sliceMap == 100*int(band[0])+islice)] = alphaMap[(sliceMap == 100*int(band[0])+islice)]
        x_coords = np.nonzero(alpha_img[512,:])[0]
        alphas = alpha_img[512,:][x_coords]
        xs = interp1d(alphas,x_coords)(0.)

        VAR1[islice-1] = xs

    VAR2 = {}
    for var in range(25):
        VAR2[str(var)] = np.zeros(nslices)
        for islice in range(1,nslices+1):
            # load new wavelength solution
            wavsolution_file   = 'data/Band'+str(band)+'_ET'+ etal[-2:] +'_slice'+str(islice)+'_coeffs.txt'
            slice_wavcoeffs = np.loadtxt(wavsolution_file,unpack=True, skiprows = 5)

            VAR2[str(var)][islice-1] = slice_wavcoeffs[var]

    Lambda_CHX_new = np.rec.array([VAR1,VAR2['0'],VAR2['1'],VAR2['2'],VAR2['3'],VAR2['4'],VAR2['5'],VAR2['6'],
                  VAR2['7'],VAR2['8'],VAR2['9'],VAR2['10'],VAR2['11'],VAR2['12'],VAR2['13'],VAR2['14'],
                  VAR2['15'],VAR2['16'],VAR2['17'],VAR2['18'],VAR2['19'],VAR2['20'],VAR2['21'],VAR2['22'],VAR2['23'],VAR2['24']],
                  dtype=[('VAR1', 'float64'), ('VAR2_0_0', 'float64'), ('VAR2_0_1', 'float64'), ('VAR2_0_2', 'float64'), ('VAR2_0_3', 'float64'), ('VAR2_0_4', 'float64'), 
                         ('VAR2_1_0', 'float64'), ('VAR2_1_1', 'float64'), ('VAR2_1_2', 'float64'), ('VAR2_1_3', 'float64'), ('VAR2_1_4', 'float64'),
                         ('VAR2_2_0', 'float64'), ('VAR2_2_1', 'float64'), ('VAR2_2_2', 'float64'), ('VAR2_2_3', 'float64'), ('VAR2_2_4', 'float64'),
                         ('VAR2_3_0', 'float64'), ('VAR2_3_1', 'float64'), ('VAR2_3_2', 'float64'), ('VAR2_3_3', 'float64'), ('VAR2_3_4', 'float64'),
                         ('VAR2_4_0', 'float64'), ('VAR2_4_1', 'float64'), ('VAR2_4_2', 'float64'), ('VAR2_4_3', 'float64'), ('VAR2_4_4', 'float64')])

    # update corresponding fits extensions
    old_distortion_cdp['Lambda_CH{}'.format(band[0])].data = Lambda_CHX_new

    # save output
    old_distortion_cdp.writeto(outDir + "MIRI_FM_MIR{}_DISTORTION_7B.06.00.fits".format(miri_setup),overwrite=True)


# # Band 1C/2C

# In[6]:

miri_setup = 'IFUSHORT_12LONG'
# load old distortion cdp fits file
old_distortion_cdp = fits.open(outDir+'MIRI_FM_MIR{}_DISTORTION_7B.05.00.fits'.format(miri_setup))

# change headers
old_distortion_cdp[0].header['DATE']     = datetime.datetime.utcnow().isoformat()
old_distortion_cdp[0].header['FILENAME'] = 'MIRI_FM_MIR{}_DISTORTION_7B.06.00.fits'.format(miri_setup)
old_distortion_cdp[0].header['VERSION']  = '7B.XX.XX'
old_distortion_cdp[0].header['ORIGFILE'] = 'MIRI_FM_MIR{}_DISTORTION_7B.05.00.fits'.format(miri_setup)
old_distortion_cdp[0].header.add_history("- updated wavelength extension Channel 1C -> DOCUMENT:")
old_distortion_cdp[0].header.add_history("- MIRI-TN-00004-ETH, Alvaro Labiano and Ioannis Argyriou")

# update the wavelength solution
for band in ['1C','2C']:
    if band == '1C': etal='ET1A'
    elif band == '2C': etal='ET1B'
    
    # number of slices in band
    d2cMaps  = d2cMapping(band,cdpDir)
    sliceMap = d2cMaps['sliceMap']
    alphaMap = d2cMaps['alphaMap']
    nslices  = d2cMaps['nslices']

    # Create arrays of variables
    VAR1 = np.zeros(nslices)
    for islice in range(1,nslices+1):
        # compute slice reference x-position
        alpha_img = np.zeros(det_dims)
        alpha_img[(sliceMap == 100*int(band[0])+islice)] = alphaMap[(sliceMap == 100*int(band[0])+islice)]
        x_coords = np.nonzero(alpha_img[512,:])[0]
        alphas = alpha_img[512,:][x_coords]
        xs = interp1d(alphas,x_coords)(0.)

        VAR1[islice-1] = xs

    VAR2 = {}
    for var in range(25):
        VAR2[str(var)] = np.zeros(nslices)
        for islice in range(1,nslices+1):
            # load new wavelength solution
            wavsolution_file   = 'data/Band'+str(band)+'_ET'+ etal[-2:] +'_slice'+str(islice)+'_coeffs.txt'
            slice_wavcoeffs = np.loadtxt(wavsolution_file,unpack=True, skiprows = 5)

            VAR2[str(var)][islice-1] = slice_wavcoeffs[var]

    Lambda_CHX_new = np.rec.array([VAR1,VAR2['0'],VAR2['1'],VAR2['2'],VAR2['3'],VAR2['4'],VAR2['5'],VAR2['6'],
                  VAR2['7'],VAR2['8'],VAR2['9'],VAR2['10'],VAR2['11'],VAR2['12'],VAR2['13'],VAR2['14'],
                  VAR2['15'],VAR2['16'],VAR2['17'],VAR2['18'],VAR2['19'],VAR2['20'],VAR2['21'],VAR2['22'],VAR2['23'],VAR2['24']],
                  dtype=[('VAR1', 'float64'), ('VAR2_0_0', 'float64'), ('VAR2_0_1', 'float64'), ('VAR2_0_2', 'float64'), ('VAR2_0_3', 'float64'), ('VAR2_0_4', 'float64'), 
                         ('VAR2_1_0', 'float64'), ('VAR2_1_1', 'float64'), ('VAR2_1_2', 'float64'), ('VAR2_1_3', 'float64'), ('VAR2_1_4', 'float64'),
                         ('VAR2_2_0', 'float64'), ('VAR2_2_1', 'float64'), ('VAR2_2_2', 'float64'), ('VAR2_2_3', 'float64'), ('VAR2_2_4', 'float64'),
                         ('VAR2_3_0', 'float64'), ('VAR2_3_1', 'float64'), ('VAR2_3_2', 'float64'), ('VAR2_3_3', 'float64'), ('VAR2_3_4', 'float64'),
                         ('VAR2_4_0', 'float64'), ('VAR2_4_1', 'float64'), ('VAR2_4_2', 'float64'), ('VAR2_4_3', 'float64'), ('VAR2_4_4', 'float64')])

    # update corresponding fits extensions
    old_distortion_cdp['Lambda_CH{}'.format(band[0])].data = Lambda_CHX_new

    # save output
    old_distortion_cdp.writeto(outDir + "MIRI_FM_MIR{}_DISTORTION_7B.06.00.fits".format(miri_setup),overwrite=True)


# In[ ]:



