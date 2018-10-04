# -*- coding: utf-8 -*-
"""

File paths to mrs observations.

:History:

Created on Thu Mar 01 10:58:50 2018

@author: Ioannis Argyriou (KUL, ioannis.argyriou@kuleuven.be)
"""
# some trivia
allbands = ['1A','1B','1C','2A','2B','2C','3A','3B','3C','4A','4B','4C']
allchannels = ['1','2','3','4']
allsubchannels = ['A','B','C']

Etalons_names = ['ET_1A','ET_1B','ET_2A','ET_2B']

# filepaths
#-- FM campaign
def FM_MTS_BB_extended_source(lvl2path,band,bb_temp=None,output='img'):
    # From FM test MRS_RAD_04 (MRS spectrophotometric performance)
    if bb_temp == '800K':
        sci_imgs = {"1A":lvl2path +'FM1T00011282/MIRFM1T00011282_1_495_SE_2011-05-31T02h15m32_LVL2.fits',
                    "1B":lvl2path +'FM1T00011283/MIRFM1T00011283_1_495_SE_2011-05-31T03h12m30_LVL2.fits',
                    "1C":lvl2path +'FM1T00011284/MIRFM1T00011284_1_495_SE_2011-05-31T04h09m25_LVL2.fits',
                    "2A":lvl2path +'FM1T00011282/MIRFM1T00011282_1_495_SE_2011-05-31T02h15m32_LVL2.fits',
                    "2B":lvl2path +'FM1T00011283/MIRFM1T00011283_1_495_SE_2011-05-31T03h12m30_LVL2.fits',
                    "2C":lvl2path +'FM1T00011284/MIRFM1T00011284_1_495_SE_2011-05-31T04h09m25_LVL2.fits',
                    "3A":lvl2path +'FM1T00011282/MIRFM1T00011282_1_494_SE_2011-05-31T02h15m02_LVL2.fits',
                    "3B":lvl2path +'FM1T00011283/MIRFM1T00011283_1_494_SE_2011-05-31T03h11m59_LVL2.fits',
                    "3C":lvl2path +'FM1T00011284/MIRFM1T00011284_1_494_SE_2011-05-31T04h08m55_LVL2.fits',
                    "4A":lvl2path +'FM1T00011282/MIRFM1T00011282_1_494_SE_2011-05-31T02h15m02_LVL2.fits',
                    "4B":lvl2path +'FM1T00011283/MIRFM1T00011283_1_494_SE_2011-05-31T03h11m59_LVL2.fits',
                    "4C":lvl2path +'FM1T00011284/MIRFM1T00011284_1_494_SE_2011-05-31T04h08m55_LVL2.fits'}

        bkg_imgs = {"1A":lvl2path +'FM1T00011285/MIRFM1T00011285_1_495_SE_2011-05-31T05h06m47_LVL2.fits',
                    "1B":lvl2path +'FM1T00011286/MIRFM1T00011286_1_495_SE_2011-05-31T06h03m43_LVL2.fits',
                    "1C":lvl2path +'FM1T00011287/MIRFM1T00011287_1_495_SE_2011-05-31T07h00m44_LVL2.fits',
                    "2A":lvl2path +'FM1T00011285/MIRFM1T00011285_1_495_SE_2011-05-31T05h06m47_LVL2.fits',
                    "2B":lvl2path +'FM1T00011286/MIRFM1T00011286_1_495_SE_2011-05-31T06h03m43_LVL2.fits',
                    "2C":lvl2path +'FM1T00011287/MIRFM1T00011287_1_495_SE_2011-05-31T07h00m44_LVL2.fits',
                    "3A":lvl2path +'FM1T00011285/MIRFM1T00011285_1_494_SE_2011-05-31T05h06m17_LVL2.fits',
                    "3B":lvl2path +'FM1T00011286/MIRFM1T00011286_1_494_SE_2011-05-31T06h03m14_LVL2.fits',
                    "3C":lvl2path +'FM1T00011287/MIRFM1T00011287_1_494_SE_2011-05-31T07h00m15_LVL2.fits',
                    "4A":lvl2path +'FM1T00011285/MIRFM1T00011285_1_494_SE_2011-05-31T05h06m17_LVL2.fits',
                    "4B":lvl2path +'FM1T00011286/MIRFM1T00011286_1_494_SE_2011-05-31T06h03m14_LVL2.fits',
                    "4C":lvl2path +'FM1T00011287/MIRFM1T00011287_1_494_SE_2011-05-31T07h00m15_LVL2.fits'}
    elif bb_temp == '600K':
        sci_imgs = {"1A":lvl2path +'FM1T00011982/MIRFM1T00011982_1_495_SE_2011-06-19T03h57m48_LVL2.fits',
                    "1B":lvl2path +'FM1T00011984/MIRFM1T00011984_1_495_SE_2011-06-19T05h55m41_LVL2.fits',
                    "1C":lvl2path +'FM1T00011986/MIRFM1T00011986_1_495_SE_2011-06-19T07h53m34_LVL2.fits',
                    "2A":lvl2path +'FM1T00011982/MIRFM1T00011982_1_495_SE_2011-06-19T03h57m48_LVL2.fits',
                    "2B":lvl2path +'FM1T00011984/MIRFM1T00011984_1_495_SE_2011-06-19T05h55m41_LVL2.fits',
                    "2C":lvl2path +'FM1T00011986/MIRFM1T00011986_1_495_SE_2011-06-19T07h53m34_LVL2.fits',
                    "3A":lvl2path +'FM1T00011982/MIRFM1T00011982_1_494_SE_2011-06-19T03h57m18_LVL2.fits',
                    "3B":lvl2path +'FM1T00011984/MIRFM1T00011984_1_494_SE_2011-06-19T05h55m11_LVL2.fits',
                    "3C":lvl2path +'FM1T00011986/MIRFM1T00011986_1_494_SE_2011-06-19T07h53m05_LVL2.fits',
                    "4A":lvl2path +'FM1T00011982/MIRFM1T00011982_1_494_SE_2011-06-19T03h57m18_LVL2.fits',
                    "4B":lvl2path +'FM1T00011984/MIRFM1T00011984_1_494_SE_2011-06-19T05h55m11_LVL2.fits',
                    "4C":lvl2path +'FM1T00011986/MIRFM1T00011986_1_494_SE_2011-06-19T07h53m05_LVL2.fits'}

        bkg_imgs = {"1A":lvl2path +'FM1T00011983/MIRFM1T00011983_1_495_SE_2011-06-19T04h57m29_LVL2.fits',
                    "1B":lvl2path +'FM1T00011985/MIRFM1T00011985_1_495_SE_2011-06-19T06h55m53_LVL2.fits',
                    "1C":lvl2path +'FM1T00011987/MIRFM1T00011987_1_495_SE_2011-06-19T08h50m05_LVL2.fits',
                    "2A":lvl2path +'FM1T00011983/MIRFM1T00011983_1_495_SE_2011-06-19T04h57m29_LVL2.fits',
                    "2B":lvl2path +'FM1T00011985/MIRFM1T00011985_1_495_SE_2011-06-19T06h55m53_LVL2.fits',
                    "2C":lvl2path +'FM1T00011987/MIRFM1T00011987_1_495_SE_2011-06-19T08h50m05_LVL2.fits',
                    "3A":lvl2path +'FM1T00011983/MIRFM1T00011983_1_494_SE_2011-06-19T04h57m00_LVL2.fits',
                    "3B":lvl2path +'FM1T00011985/MIRFM1T00011985_1_494_SE_2011-06-19T06h55m23_LVL2.fits',
                    "3C":lvl2path +'FM1T00011987/MIRFM1T00011987_1_494_SE_2011-06-19T08h49m36_LVL2.fits',
                    "4A":lvl2path +'FM1T00011983/MIRFM1T00011983_1_494_SE_2011-06-19T04h57m00_LVL2.fits',
                    "4B":lvl2path +'FM1T00011985/MIRFM1T00011985_1_494_SE_2011-06-19T06h55m23_LVL2.fits',
                    "4C":lvl2path +'FM1T00011987/MIRFM1T00011987_1_494_SE_2011-06-19T08h49m36_LVL2.fits'}
    elif bb_temp == '400K':
        sci_imgs = {"1A":lvl2path +'FM1T00011819/MIRFM1T00011819_1_495_SE_2011-06-17T14h32m35_LVL2.fits',
                    "1B":lvl2path +'FM1T00011822/MIRFM1T00011822_1_495_SE_2011-06-17T16h41m44_LVL2.fits',
                    "1C":lvl2path +'FM1T00011817/MIRFM1T00011817_1_495_SE_2011-06-17T12h27m28_LVL2.fits',
                    "2A":lvl2path +'FM1T00011819/MIRFM1T00011819_1_495_SE_2011-06-17T14h32m35_LVL2.fits',
                    "2B":lvl2path +'FM1T00011822/MIRFM1T00011822_1_495_SE_2011-06-17T16h41m44_LVL2.fits',
                    "2C":lvl2path +'FM1T00011817/MIRFM1T00011817_1_495_SE_2011-06-17T12h27m28_LVL2.fits',
                    "3A":lvl2path +'FM1T00011819/MIRFM1T00011819_1_494_SE_2011-06-17T14h32m05_LVL2.fits',
                    "3B":lvl2path +'FM1T00011822/MIRFM1T00011822_1_494_SE_2011-06-17T16h41m14_LVL2.fits',
                    "3C":lvl2path +'FM1T00011817/MIRFM1T00011817_1_494_SE_2011-06-17T12h26m58_LVL2.fits',
                    "4A":lvl2path +'FM1T00011819/MIRFM1T00011819_1_494_SE_2011-06-17T14h32m05_LVL2.fits',
                    "4B":lvl2path +'FM1T00011822/MIRFM1T00011822_1_494_SE_2011-06-17T16h41m14_LVL2.fits',
                    "4C":lvl2path +'FM1T00011817/MIRFM1T00011817_1_494_SE_2011-06-17T12h26m58_LVL2.fits'}

        bkg_imgs = {"1A":lvl2path +'FM1T00011820/MIRFM1T00011820_1_495_SE_2011-06-17T15h29m07_LVL2.fits',
                    "1B":lvl2path +'FM1T00011823/MIRFM1T00011823_1_495_SE_2011-06-17T17h39m40_LVL2.fits',
                    "1C":lvl2path +'FM1T00011818/MIRFM1T00011818_1_495_SE_2011-06-17T13h34m18_LVL2.fits',
                    "2A":lvl2path +'FM1T00011820/MIRFM1T00011820_1_495_SE_2011-06-17T15h29m07_LVL2.fits',
                    "2B":lvl2path +'FM1T00011823/MIRFM1T00011823_1_495_SE_2011-06-17T17h39m40_LVL2.fits',
                    "2C":lvl2path +'FM1T00011818/MIRFM1T00011818_1_495_SE_2011-06-17T13h34m18_LVL2.fits',
                    "3A":lvl2path +'FM1T00011820/MIRFM1T00011820_1_494_SE_2011-06-17T15h28m37_LVL2.fits',
                    "3B":lvl2path +'FM1T00011823/MIRFM1T00011823_1_494_SE_2011-06-17T17h39m10_LVL2.fits',
                    "3C":lvl2path +'FM1T00011818/MIRFM1T00011818_1_494_SE_2011-06-17T13h33m48_LVL2.fits',
                    "4A":lvl2path +'FM1T00011820/MIRFM1T00011820_1_494_SE_2011-06-17T15h28m37_LVL2.fits',
                    "4B":lvl2path +'FM1T00011823/MIRFM1T00011823_1_494_SE_2011-06-17T17h39m10_LVL2.fits',
                    "4C":lvl2path +'FM1T00011818/MIRFM1T00011818_1_494_SE_2011-06-17T13h33m48_LVL2.fits'}
    if output == 'filename':
        return sci_imgs[band],bkg_imgs[band]
    elif output == 'img':
        from astropy.io import fits
        hdulist_sci,hdulist_bkg = fits.open(sci_imgs[band]), fits.open(bkg_imgs[band])
        sci_data,bkg_data = hdulist_sci[0].data[0,:,:],hdulist_bkg[0].data[0,:,:]
        hdulist_sci.close() ; hdulist_bkg.close()
        return sci_data,bkg_data
    elif output == 'img_error':
        from astropy.io import fits
        hdulist_sci,hdulist_bkg = fits.open(sci_imgs[band]), fits.open(bkg_imgs[band])
        sci_data,bkg_data = hdulist_sci[0].data[1,:,:],hdulist_bkg[0].data[1,:,:]
        hdulist_sci.close() ; hdulist_bkg.close()
        return sci_data,bkg_data

def MIRI_internal_calibration_source(lvl2path,band,campaign=None,output='img'):
    # MRS_RAD_11 (MRS Calibration Source)
    """
    * 800K BB source
    * observed in different test campaigns
    * CCC closed during internal calibration source observation (no background observations)
    """
    if campaign == 'FM':
        sci_imgs = {"1A":lvl2path +'FM1T00012668/MIRFM1T00012668_1_495_SE_2011-07-06T17h08m40_LVL2.fits',
                    "1B":lvl2path +'FM1T00012668/MIRFM1T00012668_5_495_SE_2011-07-06T17h32m15_LVL2.fits',
                    "1C":lvl2path +'FM1T00012668/MIRFM1T00012668_9_495_SE_2011-07-06T17h55m46_LVL2.fits',
                    "2A":lvl2path +'FM1T00012668/MIRFM1T00012668_1_495_SE_2011-07-06T17h08m40_LVL2.fits',
                    "2B":lvl2path +'FM1T00012668/MIRFM1T00012668_5_495_SE_2011-07-06T17h32m15_LVL2.fits',
                    "2C":lvl2path +'FM1T00012668/MIRFM1T00012668_9_495_SE_2011-07-06T17h55m46_LVL2.fits',
                    "3A":lvl2path +'FM1T00012668/MIRFM1T00012668_1_494_SE_2011-07-06T17h08m24_LVL2.fits',
                    "3B":lvl2path +'FM1T00012668/MIRFM1T00012668_5_494_SE_2011-07-06T17h31m59_LVL2.fits',
                    "3C":lvl2path +'FM1T00012668/MIRFM1T00012668_9_494_SE_2011-07-06T17h55m30_LVL2.fits',
                    "4A":lvl2path +'FM1T00012668/MIRFM1T00012668_1_494_SE_2011-07-06T17h08m24_LVL2.fits',
                    "4B":lvl2path +'FM1T00012668/MIRFM1T00012668_5_494_SE_2011-07-06T17h31m59_LVL2.fits',
                    "4C":lvl2path +'FM1T00012668/MIRFM1T00012668_9_494_SE_2011-07-06T17h55m30_LVL2.fits',
                    "2AxB":lvl2path +'FM1T00012668/MIRFM1T00012668_4_495_SE_2011-07-06T17h26m25_LVL2.fits',
                    "2AxC":lvl2path +'FM1T00012668/MIRFM1T00012668_7_495_SE_2011-07-06T17h44m11_LVL2.fits',
                    "2BxA":lvl2path +'FM1T00012668/MIRFM1T00012668_2_495_SE_2011-07-06T17h14m30_LVL2.fits',
                    "2BxC":lvl2path +'FM1T00012668/MIRFM1T00012668_8_495_SE_2011-07-06T17h50m01_LVL2.fits',
                    "2CxA":lvl2path +'FM1T00012668/MIRFM1T00012668_3_495_SE_2011-07-06T17h20m16_LVL2.fits',
                    "2CxB":lvl2path +'FM1T00012668/MIRFM1T00012668_6_495_SE_2011-07-06T17h38m00_LVL2.fits',
                    "3AxB":lvl2path +'FM1T00012668/MIRFM1T00012668_4_494_SE_2011-07-06T17h26m09_LVL2.fits',
                    "3AxC":lvl2path +'FM1T00012668/MIRFM1T00012668_7_494_SE_2011-07-06T17h43m55_LVL2.fits',
                    "3BxA":lvl2path +'FM1T00012668/MIRFM1T00012668_2_494_SE_2011-07-06T17h14m14_LVL2.fits',
                    "3BxC":lvl2path +'FM1T00012668/MIRFM1T00012668_8_494_SE_2011-07-06T17h49m45_LVL2.fits',
                    "3CxA":lvl2path +'FM1T00012668/MIRFM1T00012668_3_494_SE_2011-07-06T17h19m59_LVL2.fits',
                    "3CxB":lvl2path +'FM1T00012668/MIRFM1T00012668_6_494_SE_2011-07-06T17h37m45_LVL2.fits',
                    "4AxB":lvl2path +'FM1T00012668/MIRFM1T00012668_2_494_SE_2011-07-06T17h14m14_LVL2.fits',
                    "4AxC":lvl2path +'FM1T00012668/MIRFM1T00012668_3_494_SE_2011-07-06T17h19m59_LVL2.fits',
                    "4BxA":lvl2path +'FM1T00012668/MIRFM1T00012668_4_494_SE_2011-07-06T17h26m09_LVL2.fits',
                    "4BxC":lvl2path +'FM1T00012668/MIRFM1T00012668_6_494_SE_2011-07-06T17h37m45_LVL2.fits',
                    "4CxA":lvl2path +'FM1T00012668/MIRFM1T00012668_7_494_SE_2011-07-06T17h43m55_LVL2.fits',
                    "4CxB":lvl2path +'FM1T00012668/MIRFM1T00012668_8_494_SE_2011-07-06T17h49m45_LVL2.fits'}
    elif campaign == 'OTIS':
        sci_imgs = {"1A":lvl2path +'MIRV00331001001P0000000002103_1_495_SE_2017-08-25T19h09m24_LVL2.fits',
                    "1B":lvl2path +'',
                    "1C":lvl2path +'',
                    "2A":lvl2path +'MIRV00331001001P0000000002103_1_495_SE_2017-08-25T19h09m24_LVL2.fits',
                    "2B":lvl2path +'',
                    "2C":lvl2path +'',
                    "3A":lvl2path +'',
                    "3B":lvl2path +'',
                    "3C":lvl2path +'',
                    "4A":lvl2path +'',
                    "4B":lvl2path +'',
                    "4C":lvl2path +''}
    elif campaign == 'CV3':
        sci_imgs = {"1A":lvl2path +'MIRM33541-A-A-8MA-6019093539_1_495_SE_2016-01-19T09h59m18_LVL2.fits',
                    "1B":lvl2path +'',
                    "1C":lvl2path +'',
                    "2A":lvl2path +'MIRM33541-A-A-8MA-6019093539_1_495_SE_2016-01-19T09h59m18_LVL2.fits',
                    "2B":lvl2path +'',
                    "2C":lvl2path +'',
                    "3A":lvl2path +'',
                    "3B":lvl2path +'',
                    "3C":lvl2path +'',
                    "4A":lvl2path +'',
                    "4B":lvl2path +'',
                    "4C":lvl2path +''}
    elif campaign == 'CV1':
        sci_imgs = {"1A":lvl2path +'MIRM33512-A-A-8MA-3291020639_1_495_SE_2013-10-18T02h38m56_LVL2.fits',
                    "1B":lvl2path +'',
                    "1C":lvl2path +'',
                    "2A":lvl2path +'MIRM33512-A-A-8MA-3291020639_1_495_SE_2013-10-18T02h38m56_LVL2.fits',
                    "2B":lvl2path +'',
                    "2C":lvl2path +'',
                    "3A":lvl2path +'',
                    "3B":lvl2path +'',
                    "3C":lvl2path +'',
                    "4A":lvl2path +'',
                    "4B":lvl2path +'',
                    "4C":lvl2path +''}
    if output == 'filename':
        return sci_imgs[band]
    elif output == 'img':
        from astropy.io import fits
        hdulist_sci = fits.open(sci_imgs[band])
        sci_data = hdulist_sci[0].data[0,:,:]
        hdulist_sci.close()
        return sci_data

def FM_MTS_800K_BB_extended_source_through_etalon(lvl2path,band,etalon=None,output='img'):
    # MRS_OPT_08 (MRS wavelength characterization)
    if etalon == 'ET1A':
        sci_imgs = {"1A":lvl2path +'FM1T00010937/MIRFM1T00010937_1_495_SE_2011-05-19T22h24m06_LVL2.fits',
                    "1B":lvl2path +'FM1T00010939/MIRFM1T00010939_1_495_SE_2011-05-20T00h04m44_LVL2.fits',
                    "1C":lvl2path +'FM1T00010843/MIRFM1T00010843_1_495_SE_2011-05-18T00h47m06_LVL2.fits',
                    "2A":lvl2path +'FM1T00010937/MIRFM1T00010937_1_495_SE_2011-05-19T22h24m06_LVL2.fits',
                    "2B":lvl2path +'FM1T00010939/MIRFM1T00010939_1_495_SE_2011-05-20T00h04m44_LVL2.fits',
                    "2C":lvl2path +'FM1T00010843/MIRFM1T00010843_1_495_SE_2011-05-18T00h47m06_LVL2.fits',
                    "3A":lvl2path +'FM1T00010937/MIRFM1T00010937_1_494_SE_2011-05-19T22h23m43_LVL2.fits',
                    "3B":lvl2path +'FM1T00010939/MIRFM1T00010939_1_494_SE_2011-05-20T00h04m21_LVL2.fits',
                    "3C":lvl2path +'FM1T00010843/MIRFM1T00010843_1_494_SE_2011-05-18T00h46m35_LVL2.fits',
                    "4A":lvl2path +'FM1T00010937/MIRFM1T00010937_1_494_SE_2011-05-19T22h23m43_LVL2.fits',
                    "4B":lvl2path +'FM1T00010939/MIRFM1T00010939_1_494_SE_2011-05-20T00h04m21_LVL2.fits',
                    "4C":lvl2path +'FM1T00010843/MIRFM1T00010843_1_494_SE_2011-05-18T00h46m35_LVL2.fits'}

        bkg_imgs = {"1A":lvl2path +'FM1T00010938/MIRFM1T00010938_1_495_SE_2011-05-19T23h16m52_LVL2.fits',
                    "1B":lvl2path +'FM1T00010940/MIRFM1T00010940_1_495_SE_2011-05-20T00h53m40_LVL2.fits',
                    "1C":lvl2path +'FM1T00010844/MIRFM1T00010844_1_495_SE_2011-05-18T01h53m59_LVL2.fits',
                    "2A":lvl2path +'FM1T00010938/MIRFM1T00010938_1_495_SE_2011-05-19T23h16m52_LVL2.fits',
                    "2B":lvl2path +'FM1T00010940/MIRFM1T00010940_1_495_SE_2011-05-20T00h53m40_LVL2.fits',
                    "2C":lvl2path +'FM1T00010844/MIRFM1T00010844_1_495_SE_2011-05-18T01h53m59_LVL2.fits',
                    "3A":lvl2path +'FM1T00010938/MIRFM1T00010938_1_494_SE_2011-05-19T23h16m30_LVL2.fits',
                    "3B":lvl2path +'FM1T00010940/MIRFM1T00010940_1_494_SE_2011-05-20T00h53m17_LVL2.fits',
                    "3C":lvl2path +'FM1T00010844/MIRFM1T00010844_1_494_SE_2011-05-18T01h53m27_LVL2.fits',
                    "4A":lvl2path +'FM1T00010938/MIRFM1T00010938_1_494_SE_2011-05-19T23h16m30_LVL2.fits',
                    "4B":lvl2path +'FM1T00010940/MIRFM1T00010940_1_494_SE_2011-05-20T00h53m17_LVL2.fits',
                    "4C":lvl2path +'FM1T00010844/MIRFM1T00010844_1_494_SE_2011-05-18T01h53m27_LVL2.fits'}
    elif etalon == 'ET1B':
        sci_imgs = {"1A":lvl2path +'FM1T00010918/MIRFM1T00010918_1_495_SE_2011-05-19T15h17m35_LVL2.fits',
                    "1B":lvl2path +'FM1T00010920/MIRFM1T00010920_1_495_SE_2011-05-19T16h52m14_LVL2.fits',
                    "1C":lvl2path +'FM1T00010943/MIRFM1T00010943_1_495_SE_2011-05-20T03h23m04_LVL2.fits',
                    "2A":lvl2path +'FM1T00010918/MIRFM1T00010918_1_495_SE_2011-05-19T15h17m35_LVL2.fits',
                    "2B":lvl2path +'FM1T00010920/MIRFM1T00010920_1_495_SE_2011-05-19T16h52m14_LVL2.fits',
                    "2C":lvl2path +'FM1T00010943/MIRFM1T00010943_1_495_SE_2011-05-20T03h23m04_LVL2.fits',
                    "3A":lvl2path +'FM1T00010918/MIRFM1T00010918_1_494_SE_2011-05-19T15h17m12_LVL2.fits',
                    "3B":lvl2path +'FM1T00010920/MIRFM1T00010920_1_494_SE_2011-05-19T16h51m50_LVL2.fits',
                    "3C":lvl2path +'FM1T00010943/MIRFM1T00010943_1_494_SE_2011-05-20T03h22m41_LVL2.fits',
                    "4A":lvl2path +'FM1T00010918/MIRFM1T00010918_1_494_SE_2011-05-19T15h17m12_LVL2.fits',
                    "4B":lvl2path +'FM1T00010920/MIRFM1T00010920_1_494_SE_2011-05-19T16h51m50_LVL2.fits',
                    "4C":lvl2path +'FM1T00010943/MIRFM1T00010943_1_494_SE_2011-05-20T03h22m41_LVL2.fits'}

        bkg_imgs = {"1A":lvl2path +'FM1T00010919/MIRFM1T00010919_1_495_SE_2011-05-19T16h04m35_LVL2.fits',
                    "1B":lvl2path +'FM1T00010921/MIRFM1T00010921_1_495_SE_2011-05-19T17h38m33_LVL2.fits',
                    "1C":lvl2path +'FM1T00010942/MIRFM1T00010942_1_495_SE_2011-05-20T02h30m42_LVL2.fits',
                    "2A":lvl2path +'FM1T00010919/MIRFM1T00010919_1_495_SE_2011-05-19T16h04m35_LVL2.fits',
                    "2B":lvl2path +'FM1T00010921/MIRFM1T00010921_1_495_SE_2011-05-19T17h38m33_LVL2.fits',
                    "2C":lvl2path +'FM1T00010942/MIRFM1T00010942_1_495_SE_2011-05-20T02h30m42_LVL2.fits',
                    "3A":lvl2path +'FM1T00010919/MIRFM1T00010919_1_494_SE_2011-05-19T16h04m13_LVL2.fits',
                    "3B":lvl2path +'FM1T00010921/MIRFM1T00010921_1_494_SE_2011-05-19T17h38m10_LVL2.fits',
                    "3C":lvl2path +'FM1T00010942/MIRFM1T00010942_1_494_SE_2011-05-20T02h30m20_LVL2.fits',
                    "4A":lvl2path +'FM1T00010919/MIRFM1T00010919_1_494_SE_2011-05-19T16h04m13_LVL2.fits',
                    "4B":lvl2path +'FM1T00010921/MIRFM1T00010921_1_494_SE_2011-05-19T17h38m10_LVL2.fits',
                    "4C":lvl2path +'FM1T00010942/MIRFM1T00010942_1_494_SE_2011-05-20T02h30m20_LVL2.fits'}
    elif etalon == 'ET2A':
        sci_imgs = {"1A":lvl2path +'FM1T00010912/MIRFM1T00010912_1_495_SE_2011-05-19T09h52m51_LVL2.fits',
                    "1B":lvl2path +'FM1T00010914/MIRFM1T00010914_1_495_SE_2011-05-19T12h06m04_LVL2.fits',
                    "1C":lvl2path +'FM1T00010916/MIRFM1T00010916_1_495_SE_2011-05-19T13h40m32_LVL2.fits',
                    "2A":lvl2path +'FM1T00010912/MIRFM1T00010912_1_495_SE_2011-05-19T09h52m51_LVL2.fits',
                    "2B":lvl2path +'FM1T00010914/MIRFM1T00010914_1_495_SE_2011-05-19T12h06m04_LVL2.fits',
                    "2C":lvl2path +'FM1T00010916/MIRFM1T00010916_1_495_SE_2011-05-19T13h40m32_LVL2.fits',
                    "3A":lvl2path +'FM1T00010912/MIRFM1T00010912_1_494_SE_2011-05-19T09h52m28_LVL2.fits',
                    "3B":lvl2path +'FM1T00010914/MIRFM1T00010914_1_494_SE_2011-05-19T12h05m42_LVL2.fits',
                    "3C":lvl2path +'FM1T00010916/MIRFM1T00010916_1_494_SE_2011-05-19T13h40m09_LVL2.fits',
                    "4A":lvl2path +'FM1T00010912/MIRFM1T00010912_1_494_SE_2011-05-19T09h52m28_LVL2.fits',
                    "4B":lvl2path +'FM1T00010914/MIRFM1T00010914_1_494_SE_2011-05-19T12h05m42_LVL2.fits',
                    "4C":lvl2path +'FM1T00010916/MIRFM1T00010916_1_494_SE_2011-05-19T13h40m09_LVL2.fits'}

        bkg_imgs = {"1A":lvl2path +'FM1T00010913/MIRFM1T00010913_1_495_SE_2011-05-19T11h07m18_LVL2.fits',
                    "1B":lvl2path +'FM1T00010915/MIRFM1T00010915_1_495_SE_2011-05-19T12h52m36_LVL2.fits',
                    "1C":lvl2path +'FM1T00010917/MIRFM1T00010917_1_495_SE_2011-05-19T14h28m24_LVL2.fits',
                    "2A":lvl2path +'FM1T00010913/MIRFM1T00010913_1_495_SE_2011-05-19T11h07m18_LVL2.fits',
                    "2B":lvl2path +'FM1T00010915/MIRFM1T00010915_1_495_SE_2011-05-19T12h52m36_LVL2.fits',
                    "2C":lvl2path +'FM1T00010917/MIRFM1T00010917_1_495_SE_2011-05-19T14h28m24_LVL2.fits',
                    "3A":lvl2path +'FM1T00010913/MIRFM1T00010913_1_494_SE_2011-05-19T11h06m56_LVL2.fits',
                    "3B":lvl2path +'FM1T00010915/MIRFM1T00010915_1_494_SE_2011-05-19T12h52m13_LVL2.fits',
                    "3C":lvl2path +'FM1T00010917/MIRFM1T00010917_1_494_SE_2011-05-19T14h28m01_LVL2.fits',
                    "4A":lvl2path +'FM1T00010913/MIRFM1T00010913_1_494_SE_2011-05-19T11h06m56_LVL2.fits',
                    "4B":lvl2path +'FM1T00010915/MIRFM1T00010915_1_494_SE_2011-05-19T12h52m13_LVL2.fits',
                    "4C":lvl2path +'FM1T00010917/MIRFM1T00010917_1_494_SE_2011-05-19T14h28m01_LVL2.fits'}
    elif etalon == 'ET2B':
        sci_imgs = {"1A":lvl2path +'FM1T00010963/MIRFM1T00010963_1_495_SE_2011-05-20T23h07m01_LVL2.fits',
                    "1B":lvl2path +'FM1T00010952/MIRFM1T00010952_1_495_SE_2011-05-20T11h03m18_LVL2.fits',
                    "1C":lvl2path +'FM1T00010954/MIRFM1T00010954_1_495_SE_2011-05-20T12h38m33_LVL2.fits',
                    "2A":lvl2path +'FM1T00010963/MIRFM1T00010963_1_495_SE_2011-05-20T23h07m01_LVL2.fits',
                    "2B":lvl2path +'FM1T00010952/MIRFM1T00010952_1_495_SE_2011-05-20T11h03m18_LVL2.fits',
                    "2C":lvl2path +'FM1T00010954/MIRFM1T00010954_1_495_SE_2011-05-20T12h38m33_LVL2.fits',
                    "3A":lvl2path +'FM1T00010963/MIRFM1T00010963_1_494_SE_2011-05-20T23h06m39_LVL2.fits',
                    "3B":lvl2path +'FM1T00010952/MIRFM1T00010952_1_494_SE_2011-05-20T11h02m55_LVL2.fits',
                    "3C":lvl2path +'FM1T00010954/MIRFM1T00010954_1_494_SE_2011-05-20T12h38m09_LVL2.fits',
                    "4A":lvl2path +'FM1T00010963/MIRFM1T00010963_1_494_SE_2011-05-20T23h06m39_LVL2.fits',
                    "4B":lvl2path +'FM1T00010952/MIRFM1T00010952_1_494_SE_2011-05-20T11h02m55_LVL2.fits',
                    "4C":lvl2path +'FM1T00010954/MIRFM1T00010954_1_494_SE_2011-05-20T12h38m09_LVL2.fits'}

        bkg_imgs = {"1A":lvl2path +'FM1T00010964/MIRFM1T00010964_1_495_SE_2011-05-21T00h03m53_LVL2.fits',
                    "1B":lvl2path +'FM1T00010953/MIRFM1T00010953_1_495_SE_2011-05-20T11h50m14_LVL2.fits',
                    "1C":lvl2path +'FM1T00010955/MIRFM1T00010955_1_495_SE_2011-05-20T13h25m22_LVL2.fits',
                    "2A":lvl2path +'FM1T00010964/MIRFM1T00010964_1_495_SE_2011-05-21T00h03m53_LVL2.fits',
                    "2B":lvl2path +'FM1T00010953/MIRFM1T00010953_1_495_SE_2011-05-20T11h50m14_LVL2.fits',
                    "2C":lvl2path +'FM1T00010955/MIRFM1T00010955_1_495_SE_2011-05-20T13h25m22_LVL2.fits',
                    "3A":lvl2path +'FM1T00010964/MIRFM1T00010964_1_494_SE_2011-05-21T00h03m30_LVL2.fits',
                    "3B":lvl2path +'FM1T00010953/MIRFM1T00010953_1_494_SE_2011-05-20T11h49m51_LVL2.fits',
                    "3C":lvl2path +'FM1T00010955/MIRFM1T00010955_1_494_SE_2011-05-20T13h24m59_LVL2.fits',
                    "4A":lvl2path +'FM1T00010964/MIRFM1T00010964_1_494_SE_2011-05-21T00h03m30_LVL2.fits',
                    "4B":lvl2path +'FM1T00010953/MIRFM1T00010953_1_494_SE_2011-05-20T11h49m51_LVL2.fits',
                    "4C":lvl2path +'FM1T00010955/MIRFM1T00010955_1_494_SE_2011-05-20T13h24m59_LVL2.fits'}
    if output == 'filename':
        return sci_imgs[band],bkg_imgs[band]
    elif output == 'img':
        from astropy.io import fits
        hdulist_sci,hdulist_bkg = fits.open(sci_imgs[band]), fits.open(bkg_imgs[band])
        sci_data,bkg_data = hdulist_sci[0].data[0,:,:],hdulist_bkg[0].data[0,:,:]
        hdulist_sci.close() ; hdulist_bkg.close()
        return sci_data,bkg_data

def FM_MTS_800K_BB_extended_source_through_etalon_through_pinhole(lvl2path,band,etalon=None,output='img'):
    # MRS_OPT_06 (MRS Image Quality)
    if etalon == 'ET1A':
        sci_imgs = {"1A":lvl2path +'FM1T00012175/MIRFM1T00012175_1_495_SE_2011-06-26T20h22m07_LVL2.fits',
                    "1B":lvl2path +'FM1T00012846/MIRFM1T00012846_1_495_SE_2011-07-11T00h02m43_LVL2.fits',
                    "1C":lvl2path +'FM1T00012937/MIRFM1T00012937_1_495_SE_2011-07-13T07h20m26_LVL2.fits',
                    "2A":lvl2path +'FM1T00012175/MIRFM1T00012175_1_495_SE_2011-06-26T20h22m07_LVL2.fits',
                    "2B":lvl2path +'FM1T00012846/MIRFM1T00012846_1_495_SE_2011-07-11T00h02m43_LVL2.fits',
                    "2C":lvl2path +'FM1T00012937/MIRFM1T00012937_1_495_SE_2011-07-13T07h20m26_LVL2.fits',
                    "3A":lvl2path +'FM1T00012175/MIRFM1T00012175_1_494_SE_2011-06-26T20h21m53_LVL2.fits',
                    "3B":lvl2path +'FM1T00012846/MIRFM1T00012846_1_494_SE_2011-07-11T00h02m28_LVL2.fits',
                    "3C":lvl2path +'FM1T00012937/MIRFM1T00012937_1_494_SE_2011-07-13T07h20m12_LVL2.fits',
                    "4A":lvl2path +'FM1T00012175/MIRFM1T00012175_1_494_SE_2011-06-26T20h21m53_LVL2.fits',
                    "4B":lvl2path +'FM1T00012846/MIRFM1T00012846_1_494_SE_2011-07-11T00h02m28_LVL2.fits',
                    "4C":lvl2path +'FM1T00012937/MIRFM1T00012937_1_494_SE_2011-07-13T07h20m12_LVL2.fits'}

        bkg_imgs = {"1A":lvl2path +'FM1T00012179/MIRFM1T00012179_1_495_SE_2011-06-27T06h45m05_LVL2.fits',
                    "1B":lvl2path +'FM1T00012844/MIRFM1T00012844_1_495_SE_2011-07-10T23h24m47_LVL2.fits',
                    "1C":lvl2path +'FM1T00012945/MIRFM1T00012945_1_495_SE_2011-07-13T16h03m06_LVL2.fits',
                    "2A":lvl2path +'FM1T00012179/MIRFM1T00012179_1_495_SE_2011-06-27T06h45m05_LVL2.fits',
                    "2B":lvl2path +'FM1T00012844/MIRFM1T00012844_1_495_SE_2011-07-10T23h24m47_LVL2.fits',
                    "2C":lvl2path +'FM1T00012945/MIRFM1T00012945_1_495_SE_2011-07-13T16h03m06_LVL2.fits',
                    "3A":lvl2path +'FM1T00012179/MIRFM1T00012179_1_494_SE_2011-06-27T06h44m51_LVL2.fits',
                    "3B":lvl2path +'FM1T00012844/MIRFM1T00012844_1_494_SE_2011-07-10T23h24m32_LVL2.fits',
                    "3C":lvl2path +'FM1T00012945/MIRFM1T00012945_1_494_SE_2011-07-13T16h02m52_LVL2.fits',
                    "4A":lvl2path +'FM1T00012179/MIRFM1T00012179_1_494_SE_2011-06-27T06h44m51_LVL2.fits',
                    "4B":lvl2path +'FM1T00012844/MIRFM1T00012844_1_494_SE_2011-07-10T23h24m32_LVL2.fits',
                    "4C":lvl2path +'FM1T00012945/MIRFM1T00012945_1_494_SE_2011-07-13T16h02m52_LVL2.fits'}
    elif etalon == 'ET1B':
        sci_imgs = {"1A":lvl2path +'',
                    "1B":lvl2path +'',
                    "1C":lvl2path +'',
                    "2A":lvl2path +'',
                    "2B":lvl2path +'',
                    "2C":lvl2path +'',
                    "3A":lvl2path +'',
                    "3B":lvl2path +'',
                    "3C":lvl2path +'',
                    "4A":lvl2path +'',
                    "4B":lvl2path +'',
                    "4C":lvl2path +''}

        bkg_imgs = {"1A":lvl2path +'',
                    "1B":lvl2path +'',
                    "1C":lvl2path +'',
                    "2A":lvl2path +'',
                    "2B":lvl2path +'',
                    "2C":lvl2path +'',
                    "3A":lvl2path +'',
                    "3B":lvl2path +'',
                    "3C":lvl2path +'',
                    "4A":lvl2path +'',
                    "4B":lvl2path +'',
                    "4C":lvl2path +''}
    elif etalon == 'ET2A':
        sci_imgs = {"1A":lvl2path +'',
                    "1B":lvl2path +'',
                    "1C":lvl2path +'',
                    "2A":lvl2path +'',
                    "2B":lvl2path +'',
                    "2C":lvl2path +'',
                    "3A":lvl2path +'',
                    "3B":lvl2path +'',
                    "3C":lvl2path +'',
                    "4A":lvl2path +'',
                    "4B":lvl2path +'',
                    "4C":lvl2path +''}

        bkg_imgs = {"1A":lvl2path +'',
                    "1B":lvl2path +'',
                    "1C":lvl2path +'',
                    "2A":lvl2path +'',
                    "2B":lvl2path +'',
                    "2C":lvl2path +'',
                    "3A":lvl2path +'',
                    "3B":lvl2path +'',
                    "3C":lvl2path +'',
                    "4A":lvl2path +'',
                    "4B":lvl2path +'',
                    "4C":lvl2path +''}
    elif etalon == 'ET2B':
        sci_imgs = {"1A":lvl2path +'',
                    "1B":lvl2path +'',
                    "1C":lvl2path +'',
                    "2A":lvl2path +'',
                    "2B":lvl2path +'',
                    "2C":lvl2path +'',
                    "3A":lvl2path +'',
                    "3B":lvl2path +'',
                    "3C":lvl2path +'',
                    "4A":lvl2path +'',
                    "4B":lvl2path +'',
                    "4C":lvl2path +''}

        bkg_imgs = {"1A":lvl2path +'',
                    "1B":lvl2path +'',
                    "1C":lvl2path +'',
                    "2A":lvl2path +'',
                    "2B":lvl2path +'',
                    "2C":lvl2path +'',
                    "3A":lvl2path +'',
                    "3B":lvl2path +'',
                    "3C":lvl2path +'',
                    "4A":lvl2path +'',
                    "4B":lvl2path +'',
                    "4C":lvl2path +''}
    if output == 'filename':
        return sci_imgs[band],bkg_imgs[band]
    elif output == 'img':
        from astropy.io import fits
        hdulist_sci = fits.open(sci_imgs[band])
        hdulist_bkg = fits.open(bkg_imgs[band])
        sci_data,bkg_data = hdulist_sci[0].data[0,:,:], hdulist_bkg[0].data[0,:,:]
        hdulist_sci.close() ; hdulist_bkg.close()
        return sci_data,bkg_data

def RAL_FTS_ET_observations(obsDir,etalon=None):
    import numpy as np
    if etalon == 'ET1A':
        etalon1A_file = obsDir+'MIRI_b_E1av_etalon1A_77K.txt'
        wvnrs,etalon_data = np.genfromtxt(etalon1A_file, skip_footer = 1, usecols=(0,1), delimiter = '',unpack='True')
    elif etalon == 'ET1B':
        etalon1B_file = obsDir+'MIRI_b_G3av_etalon1B_80K.txt'
        wvnrs,etalon_data = np.genfromtxt(etalon1B_file, skip_footer = 1, usecols=(0,1), delimiter = '',unpack='True')
    elif etalon == 'ET2A':
        etalon2A_file = obsDir+'MIRI_b_K1av_etalon2A_80K.txt'
        wvnrs,etalon_data = np.genfromtxt(etalon2A_file, skip_footer = 1, usecols=(0,1), delimiter = '',unpack='True')
    elif etalon == 'ET2B':
        import csv
        etalon2B_file = obsDir+'MIRI_etalon2B_80K.csv'
        wvnrs,etalon_data = [],[]
        with open(etalon2B_file, 'rb') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                wvnrs.append(np.float(row[0]) )
                etalon_data.append(np.float(row[1]) )
        wvnrs,etalon_data = np.array(wvnrs),np.array(etalon_data)
    return wvnrs,etalon_data

def FM_MTS_800K_BB_MRS_RAD_06_raster(lvl2path,band,output='img'):
    # MRS slope stability
    sci_imgs = {"1A":lvl2path +'FM1T00011453/MIRFM1T00011453_1_495_SE_2011-06-03T21h10m39_LVL2.fits',
                "1B":lvl2path +'FM1T00011455/MIRFM1T00011455_1_495_SE_2011-06-04T03h35m04_LVL2.fits',
                "1C":lvl2path +'FM1T00011457/MIRFM1T00011457_1_495_SE_2011-06-04T09h56m24_LVL2.fits',
                "2A":lvl2path +'FM1T00011453/MIRFM1T00011453_1_495_SE_2011-06-03T21h10m39_LVL2.fits',
                "2B":lvl2path +'FM1T00011455/MIRFM1T00011455_1_495_SE_2011-06-04T03h35m04_LVL2.fits',
                "2C":lvl2path +'FM1T00011457/MIRFM1T00011457_1_495_SE_2011-06-04T09h56m24_LVL2.fits',
                "3A":lvl2path +'FM1T00011453/MIRFM1T00011453_1_494_SE_2011-06-03T21h10m20_LVL2.fits',
                "3B":lvl2path +'FM1T00011455/MIRFM1T00011455_1_494_SE_2011-06-04T03h34m46_LVL2.fits',
                "3C":lvl2path +'FM1T00011457/MIRFM1T00011457_1_494_SE_2011-06-04T09h56m06_LVL2.fits',
                "4A":lvl2path +'FM1T00011453/MIRFM1T00011453_1_494_SE_2011-06-03T21h10m20_LVL2.fits',
                "4B":lvl2path +'FM1T00011455/MIRFM1T00011455_1_494_SE_2011-06-04T03h34m46_LVL2.fits',
                "4C":lvl2path +'FM1T00011457/MIRFM1T00011457_1_494_SE_2011-06-04T09h56m06_LVL2.fits'}

    bkg_imgs = {"1A":lvl2path +'FM1T00011454/MIRFM1T00011454_1_495_SE_2011-06-04T00h23m34_LVL2.fits',
                "1B":lvl2path +'FM1T00011456/MIRFM1T00011456_1_495_SE_2011-06-04T06h45m24_LVL2.fits',
                "1C":lvl2path +'FM1T00011458/MIRFM1T00011458_1_495_SE_2011-06-04T13h06m10_LVL2.fits',
                "2A":lvl2path +'FM1T00011454/MIRFM1T00011454_1_495_SE_2011-06-04T00h23m34_LVL2.fits',
                "2B":lvl2path +'FM1T00011456/MIRFM1T00011456_1_495_SE_2011-06-04T06h45m24_LVL2.fits',
                "2C":lvl2path +'FM1T00011458/MIRFM1T00011458_1_495_SE_2011-06-04T13h06m10_LVL2.fits',
                "3A":lvl2path +'FM1T00011454/MIRFM1T00011454_1_494_SE_2011-06-04T00h23m15_LVL2.fits',
                "3B":lvl2path +'FM1T00011456/MIRFM1T00011456_1_494_SE_2011-06-04T06h45m05_LVL2.fits',
                "3C":lvl2path +'FM1T00011458/MIRFM1T00011458_1_494_SE_2011-06-04T13h05m52_LVL2.fits',
                "4A":lvl2path +'FM1T00011454/MIRFM1T00011454_1_494_SE_2011-06-04T00h23m15_LVL2.fits',
                "4B":lvl2path +'FM1T00011456/MIRFM1T00011456_1_494_SE_2011-06-04T06h45m05_LVL2.fits',
                "4C":lvl2path +'FM1T00011458/MIRFM1T00011458_1_494_SE_2011-06-04T13h05m52_LVL2.fits'}
    if output == 'filename':
        return sci_imgs[band],bkg_imgs[band]
    elif output == 'img':
        from astropy.io import fits
        hdulist_sci = fits.open(sci_imgs[band])
        hdulist_bkg = fits.open(bkg_imgs[band])
        sci_data,bkg_data = hdulist_sci[0].data[0,:,:], hdulist_bkg[0].data[0,:,:]
        hdulist_sci.close(); hdulist_bkg.close()
        return sci_data,bkg_data

def FM_MTS_800K_BB_MRS_RAD_08(lvl2path,band,wp_filter=None,output='img'):
    # MRS Out of band radiation sensitivity
    if wp_filter == 'LWP':
        sci_imgs = {"1A":lvl2path +'FM1T00010838/MIRFM1T00010838_1_495_SE_2011-05-17T19h15m39_LVL2.fits',
                    "1B":lvl2path +'FM1T00010838/MIRFM1T00010838_4_495_SE_2011-05-17T19h47m12_LVL2.fits',
                    "1C":lvl2path +'FM1T00010838/MIRFM1T00010838_7_495_SE_2011-05-17T20h19m18_LVL2.fits',
                    "2A":lvl2path +'FM1T00010838/MIRFM1T00010838_1_495_SE_2011-05-17T19h15m39_LVL2.fits',
                    "2B":lvl2path +'FM1T00010838/MIRFM1T00010838_4_495_SE_2011-05-17T19h47m12_LVL2.fits',
                    "2C":lvl2path +'FM1T00010838/MIRFM1T00010838_7_495_SE_2011-05-17T20h19m18_LVL2.fits',
                    "3A":lvl2path +'FM1T00010838/MIRFM1T00010838_1_494_SE_2011-05-17T19h15m33_LVL2.fits',
                    "3B":lvl2path +'FM1T00010838/MIRFM1T00010838_4_494_SE_2011-05-17T19h47m06_LVL2.fits',
                    "3C":lvl2path +'FM1T00010838/MIRFM1T00010838_7_494_SE_2011-05-17T20h19m12_LVL2.fits',
                    "4A":lvl2path +'FM1T00010838/MIRFM1T00010838_1_494_SE_2011-05-17T19h15m33_LVL2.fits',
                    "4B":lvl2path +'FM1T00010838/MIRFM1T00010838_4_494_SE_2011-05-17T19h47m06_LVL2.fits',
                    "4C":lvl2path +'FM1T00010838/MIRFM1T00010838_7_494_SE_2011-05-17T20h19m12_LVL2.fits'}

    elif wp_filter == 'SWP':
        sci_imgs = {"1A":lvl2path +'FM1T00010838/MIRFM1T00010838_2_495_SE_2011-05-17T19h25m57_LVL2.fits',
                    "1B":lvl2path +'FM1T00010838/MIRFM1T00010838_5_495_SE_2011-05-17T19h57m37_LVL2.fits',
                    "1C":lvl2path +'FM1T00010838/MIRFM1T00010838_8_495_SE_2011-05-17T20h29m38_LVL2.fits',
                    "2A":lvl2path +'FM1T00010838/MIRFM1T00010838_2_495_SE_2011-05-17T19h25m57_LVL2.fits',
                    "2B":lvl2path +'FM1T00010838/MIRFM1T00010838_5_495_SE_2011-05-17T19h57m37_LVL2.fits',
                    "2C":lvl2path +'FM1T00010838/MIRFM1T00010838_8_495_SE_2011-05-17T20h29m38_LVL2.fits',
                    "3A":lvl2path +'FM1T00010838/MIRFM1T00010838_2_494_SE_2011-05-17T19h25m51_LVL2.fits',
                    "3B":lvl2path +'FM1T00010838/MIRFM1T00010838_5_494_SE_2011-05-17T19h57m31_LVL2.fits',
                    "3C":lvl2path +'FM1T00010838/MIRFM1T00010838_8_494_SE_2011-05-17T20h29m32_LVL2.fits',
                    "4A":lvl2path +'FM1T00010838/MIRFM1T00010838_2_494_SE_2011-05-17T19h25m51_LVL2.fits',
                    "4B":lvl2path +'FM1T00010838/MIRFM1T00010838_5_494_SE_2011-05-17T19h57m31_LVL2.fits',
                    "4C":lvl2path +'FM1T00010838/MIRFM1T00010838_8_494_SE_2011-05-17T20h29m32_LVL2.fits'}

    bkg_imgs = {"1A":lvl2path +'FM1T00010838/MIRFM1T00010838_3_495_SE_2011-05-17T19h35m52_LVL2.fits',
                "1B":lvl2path +'FM1T00010838/MIRFM1T00010838_6_495_SE_2011-05-17T20h08m03_LVL2.fits',
                "1C":lvl2path +'FM1T00010838/MIRFM1T00010838_9_495_SE_2011-05-17T20h40m04_LVL2.fits',
                "2A":lvl2path +'FM1T00010838/MIRFM1T00010838_3_495_SE_2011-05-17T19h35m52_LVL2.fits',
                "2B":lvl2path +'FM1T00010838/MIRFM1T00010838_6_495_SE_2011-05-17T20h08m03_LVL2.fits',
                "2C":lvl2path +'FM1T00010838/MIRFM1T00010838_9_495_SE_2011-05-17T20h40m04_LVL2.fits',
                "3A":lvl2path +'FM1T00010838/MIRFM1T00010838_3_494_SE_2011-05-17T19h35m46_LVL2.fits',
                "3B":lvl2path +'FM1T00010838/MIRFM1T00010838_6_494_SE_2011-05-17T20h07m57_LVL2.fits',
                "3C":lvl2path +'FM1T00010838/MIRFM1T00010838_9_494_SE_2011-05-17T20h39m58_LVL2.fits',
                "4A":lvl2path +'FM1T00010838/MIRFM1T00010838_3_494_SE_2011-05-17T19h35m46_LVL2.fits',
                "4B":lvl2path +'FM1T00010838/MIRFM1T00010838_6_494_SE_2011-05-17T20h07m57_LVL2.fits',
                "4C":lvl2path +'FM1T00010838/MIRFM1T00010838_9_494_SE_2011-05-17T20h39m58_LVL2.fits'}
    if output == 'filename':
        return sci_imgs[band],bkg_imgs[band]
    elif output == 'img':
        from astropy.io import fits
        hdulist_sci = fits.open(sci_imgs[band])
        hdulist_bkg = fits.open(bkg_imgs[band])
        sci_data,bkg_data = hdulist_sci[0].data[0,:,:], hdulist_bkg[0].data[0,:,:]
        hdulist_sci.close(); hdulist_bkg.close()
        return sci_data,bkg_data

def FM_MTS_800K_BB_MRS_OPT_01_raster(lvl2path,pointing='all',output='img'):
    # MRS FOV and Distortion Measurement
    """
    * all pointings are in DGA setup 1A/2A
    """
    sci_imgs = {'P1':lvl2path+'FM1T00011465/MIRFM1T00011465_1_495_SE_2011-06-04T17h58m38_LVL2.fits',
                'P2':lvl2path+'FM1T00011465/MIRFM1T00011465_2_495_SE_2011-06-04T18h03m33_LVL2.fits',
                'P3':lvl2path+'FM1T00011465/MIRFM1T00011465_3_495_SE_2011-06-04T18h08m33_LVL2.fits',
                'P4':lvl2path+'FM1T00011465/MIRFM1T00011465_4_495_SE_2011-06-04T18h13m28_LVL2.fits',
                'P5':lvl2path+'FM1T00011465/MIRFM1T00011465_5_495_SE_2011-06-04T18h18m28_LVL2.fits',
                'P6':lvl2path+'FM1T00011465/MIRFM1T00011465_6_495_SE_2011-06-04T18h23m29_LVL2.fits',
                'P7':lvl2path+'FM1T00011465/MIRFM1T00011465_7_495_SE_2011-06-04T18h28m29_LVL2.fits',
                'P8':lvl2path+'FM1T00011465/MIRFM1T00011465_8_495_SE_2011-06-04T18h33m24_LVL2.fits',
                'P9':lvl2path+'FM1T00011465/MIRFM1T00011465_9_495_SE_2011-06-04T18h38m24_LVL2.fits',
                'P10':lvl2path+'FM1T00011465/MIRFM1T00011465_10_495_SE_2011-06-04T18h43m19_LVL2.fits',
                'P11':lvl2path+'FM1T00011465/MIRFM1T00011465_11_495_SE_2011-06-04T18h48m19_LVL2.fits',
                'P12':lvl2path+'FM1T00011465/MIRFM1T00011465_12_495_SE_2011-06-04T18h53m14_LVL2.fits',
                'P13':lvl2path+'FM1T00011465/MIRFM1T00011465_13_495_SE_2011-06-04T18h58m14_LVL2.fits',
                'P14':lvl2path+'FM1T00011465/MIRFM1T00011465_14_495_SE_2011-06-04T19h03m15_LVL2.fits',
                'P15':lvl2path+'FM1T00011465/MIRFM1T00011465_15_495_SE_2011-06-04T19h08m10_LVL2.fits',
                'P16':lvl2path+'FM1T00011465/MIRFM1T00011465_16_495_SE_2011-06-04T19h13m10_LVL2.fits',
                'P17':lvl2path+'FM1T00011465/MIRFM1T00011465_17_495_SE_2011-06-04T19h18m05_LVL2.fits',
                'P18':lvl2path+'FM1T00011465/MIRFM1T00011465_18_495_SE_2011-06-04T19h23m05_LVL2.fits',
                'P19':lvl2path+'FM1T00011465/MIRFM1T00011465_19_495_SE_2011-06-04T19h28m05_LVL2.fits',
                'P20':lvl2path+'FM1T00011465/MIRFM1T00011465_20_495_SE_2011-06-04T19h33m05_LVL2.fits',
                'P21':lvl2path+'FM1T00011465/MIRFM1T00011465_21_495_SE_2011-06-04T19h38m01_LVL2.fits',
                'P22':lvl2path+'FM1T00011465/MIRFM1T00011465_22_495_SE_2011-06-04T19h43m01_LVL2.fits',
                'P23':lvl2path+'FM1T00011465/MIRFM1T00011465_23_495_SE_2011-06-04T19h47m56_LVL2.fits',
                'P24':lvl2path+'FM1T00011465/MIRFM1T00011465_24_495_SE_2011-06-04T19h52m56_LVL2.fits',
                'P25':lvl2path+'FM1T00011465/MIRFM1T00011465_25_495_SE_2011-06-04T19h57m51_LVL2.fits',
                'P26':lvl2path+'FM1T00011465/MIRFM1T00011465_26_495_SE_2011-06-04T20h02m51_LVL2.fits',
                'P27':lvl2path+'FM1T00011465/MIRFM1T00011465_27_495_SE_2011-06-04T20h07m46_LVL2.fits',
                'P28':lvl2path+'FM1T00011465/MIRFM1T00011465_28_495_SE_2011-06-04T20h12m47_LVL2.fits',
                'P29':lvl2path+'FM1T00011465/MIRFM1T00011465_29_495_SE_2011-06-04T20h17m42_LVL2.fits',
                'P30':lvl2path+'FM1T00011465/MIRFM1T00011465_30_495_SE_2011-06-04T20h22m42_LVL2.fits',
                'P31':lvl2path+'FM1T00011465/MIRFM1T00011465_31_495_SE_2011-06-04T20h27m42_LVL2.fits',
                'P32':lvl2path+'FM1T00011465/MIRFM1T00011465_32_495_SE_2011-06-04T20h32m42_LVL2.fits',
                'P33':lvl2path+'FM1T00011465/MIRFM1T00011465_33_495_SE_2011-06-04T20h37m37_LVL2.fits',
                'P34':lvl2path+'FM1T00011465/MIRFM1T00011465_34_495_SE_2011-06-04T20h42m37_LVL2.fits',
                'P35':lvl2path+'FM1T00011465/MIRFM1T00011465_35_495_SE_2011-06-04T20h47m32_LVL2.fits',
                'P36':lvl2path+'FM1T00011465/MIRFM1T00011465_36_495_SE_2011-06-04T20h52m33_LVL2.fits',
                'P37':lvl2path+'FM1T00011465/MIRFM1T00011465_37_495_SE_2011-06-04T20h57m33_LVL2.fits',
                'P38':lvl2path+'FM1T00011465/MIRFM1T00011465_38_495_SE_2011-06-04T21h02m28_LVL2.fits',
                'P39':lvl2path+'FM1T00011465/MIRFM1T00011465_39_495_SE_2011-06-04T21h07m28_LVL2.fits',
                'P40':lvl2path+'FM1T00011465/MIRFM1T00011465_40_495_SE_2011-06-04T21h12m23_LVL2.fits',
                'P41':lvl2path+'FM1T00011465/MIRFM1T00011465_41_495_SE_2011-06-04T21h17m23_LVL2.fits',
                'P42':lvl2path+'FM1T00011465/MIRFM1T00011465_42_495_SE_2011-06-04T21h22m23_LVL2.fits',
                'P43':lvl2path+'FM1T00011465/MIRFM1T00011465_43_495_SE_2011-06-04T21h27m19_LVL2.fits',
                'P44':lvl2path+'FM1T00011465/MIRFM1T00011465_44_495_SE_2011-06-04T21h32m19_LVL2.fits',
                'P45':lvl2path+'FM1T00011465/MIRFM1T00011465_45_495_SE_2011-06-04T21h37m14_LVL2.fits',
                'P46':lvl2path+'FM1T00011465/MIRFM1T00011465_46_495_SE_2011-06-04T21h42m14_LVL2.fits',
                'P47':lvl2path+'FM1T00011465/MIRFM1T00011465_47_495_SE_2011-06-04T21h47m14_LVL2.fits',
                'P48':lvl2path+'FM1T00011465/MIRFM1T00011465_48_495_SE_2011-06-04T21h52m09_LVL2.fits',
                'P49':lvl2path+'FM1T00011465/MIRFM1T00011465_49_495_SE_2011-06-04T21h57m09_LVL2.fits',
                'P50':lvl2path+'FM1T00011465/MIRFM1T00011465_50_495_SE_2011-06-04T22h02m05_LVL2.fits',
                'P51':lvl2path+'FM1T00011465/MIRFM1T00011465_51_495_SE_2011-06-04T22h07m05_LVL2.fits',
                'P52':lvl2path+'FM1T00011465/MIRFM1T00011465_52_495_SE_2011-06-04T22h12m00_LVL2.fits',
                'P53':lvl2path+'FM1T00011465/MIRFM1T00011465_53_495_SE_2011-06-04T22h17m00_LVL2.fits',
                'P54':lvl2path+'FM1T00011465/MIRFM1T00011465_54_495_SE_2011-06-04T22h22m00_LVL2.fits',
                'P55':lvl2path+'FM1T00011465/MIRFM1T00011465_55_495_SE_2011-06-04T22h27m00_LVL2.fits',
                'P56':lvl2path+'FM1T00011465/MIRFM1T00011465_56_495_SE_2011-06-04T22h31m55_LVL2.fits',
                'P57':lvl2path+'FM1T00011465/MIRFM1T00011465_57_495_SE_2011-06-04T22h36m55_LVL2.fits',
                'P58':lvl2path+'FM1T00011465/MIRFM1T00011465_58_495_SE_2011-06-04T22h41m51_LVL2.fits',
                'P59':lvl2path+'FM1T00011465/MIRFM1T00011465_59_495_SE_2011-06-04T22h46m51_LVL2.fits',
                'P60':lvl2path+'FM1T00011465/MIRFM1T00011465_60_495_SE_2011-06-04T22h51m51_LVL2.fits',
                'P61':lvl2path+'FM1T00011465/MIRFM1T00011465_61_495_SE_2011-06-04T22h56m46_LVL2.fits',
                'P62':lvl2path+'FM1T00011465/MIRFM1T00011465_62_495_SE_2011-06-04T23h01m46_LVL2.fits',
                'P63':lvl2path+'FM1T00011465/MIRFM1T00011465_63_495_SE_2011-06-04T23h06m42_LVL2.fits'}
    bkg_file = lvl2path+'FM1T00011464/MIRFM1T00011464_1_495_SE_2011-06-04T17h51m23_LVL2.fits'
    if pointing == 'all':
        return sci_imgs,bkg_file
    elif (pointing != 'all') & (output=='img'):
        from astropy.io import fits
        hdulist_sci,hdulist_bkg = fits.open(sci_imgs[pointing]), fits.open(bkg_file)
        sci_data,bkg_data = hdulist_sci[0].data[0,:,:],hdulist_bkg[0].data[0,:,:]
        hdulist_sci.close() ; hdulist_bkg.close()
        return sci_data,bkg_data

def FM_MTS_800K_BB_MRS_OPT_02_obs(lvl2path,pointing='all',output='img'):
    # MRS Image Quality
    """
    * all pointings are in DGA setup 1A/2A
    """
    sci_imgs = {'P1':lvl2path+'FM1T00011485/MIRFM1T00011485_1_495_SE_2011-06-07T03h12m04_LVL2.fits',
                'P2':lvl2path+'FM1T00011485/MIRFM1T00011485_2_495_SE_2011-06-07T04h11m16_LVL2.fits',
                'P3':lvl2path+'FM1T00011485/MIRFM1T00011485_3_495_SE_2011-06-07T05h10m22_LVL2.fits',
                'P4':lvl2path+'FM1T00011485/MIRFM1T00011485_4_495_SE_2011-06-07T06h10m14_LVL2.fits',
                'P5':lvl2path+'FM1T00012172/MIRFM1T00012172_1_495_SE_2011-06-26T11h52m55_LVL2.fits',
                'P6':lvl2path+'FM1T00012172/MIRFM1T00012172_2_495_SE_2011-06-26T12h52m12_LVL2.fits',
                'P7':lvl2path+'FM1T00012172/MIRFM1T00012172_3_495_SE_2011-06-26T13h51m19_LVL2.fits',
                'P8':lvl2path+'FM1T00012172/MIRFM1T00012172_4_495_SE_2011-06-26T14h50m31_LVL2.fits',
                'P9':lvl2path+'FM1T00012173/MIRFM1T00012173_1_495_SE_2011-06-26T16h20m52_LVL2.fits',
                'P10':lvl2path+'FM1T00012173/MIRFM1T00012173_2_495_SE_2011-06-26T17h19m59_LVL2.fits',
                'P11':lvl2path+'FM1T00012173/MIRFM1T00012173_3_495_SE_2011-06-26T18h19m15_LVL2.fits',
                'P12':lvl2path+'FM1T00012173/MIRFM1T00012173_4_495_SE_2011-06-26T19h18m27_LVL2.fits'}
    bkg_imgs = {}
    for p in ['P1','P2','P3','P4']:bkg_imgs[p] = lvl2path+'FM1T00011482/MIRFM1T00011482_1_495_SE_2011-06-06T21h59m56_LVL2.fits'
    for p in ['P5','P6','P7','P8','P9','P10','P11','P12']:bkg_imgs[p] = lvl2path+'FM1T00012171/MIRFM1T00012171_1_495_SE_2011-06-26T10h49m18_LVL2.fits'
    if pointing == 'all':
        return sci_imgs,bkg_imgs
    elif (pointing != 'all') & (output=='img'):
        from astropy.io import fits
        hdulist_sci,hdulist_bkg = fits.open(sci_imgs[pointing]), fits.open(bkg_imgs[pointing])
        sci_data,bkg_data = hdulist_sci[0].data[0,:,:],hdulist_bkg[0].data[0,:,:]
        hdulist_sci.close() ; hdulist_bkg.close()
        return sci_data,bkg_data

def FM_MTS_800K_BB_MRS_OPT_06_raster(lvl2path,position=None,pointing='all',output='img'):
    # MRS Across slit scan
    """
    * all pointings are in DGA setup 1A/2A
    """
    if position == 'left':
        # Left side of FOV
        sci_imgs = {'P1':lvl2path+'FM1T00012204/MIRFM1T00012204_1_495_SE_2011-06-27T22h27m07_LVL2.fits',
                    'P2':lvl2path+'FM1T00012204/MIRFM1T00012204_2_495_SE_2011-06-27T22h42m57_LVL2.fits',
                    'P3':lvl2path+'FM1T00012204/MIRFM1T00012204_3_495_SE_2011-06-27T22h58m48_LVL2.fits',
                    'P4':lvl2path+'FM1T00012204/MIRFM1T00012204_4_495_SE_2011-06-27T23h14m33_LVL2.fits',
                    'P5':lvl2path+'FM1T00012204/MIRFM1T00012204_5_495_SE_2011-06-27T23h30m18_LVL2.fits',
                    'P6':lvl2path+'FM1T00012204/MIRFM1T00012204_6_495_SE_2011-06-27T23h46m09_LVL2.fits',
                    'P7':lvl2path+'FM1T00012204/MIRFM1T00012204_7_495_SE_2011-06-28T00h01m59_LVL2.fits',
                    'P8':lvl2path+'FM1T00012204/MIRFM1T00012204_8_495_SE_2011-06-28T00h17m50_LVL2.fits',
                    'P9':lvl2path+'FM1T00012204/MIRFM1T00012204_9_495_SE_2011-06-28T00h33m35_LVL2.fits',
                    'P10':lvl2path+'FM1T00012204/MIRFM1T00012204_10_495_SE_2011-06-28T00h49m31_LVL2.fits',
                    'P11':lvl2path+'FM1T00012204/MIRFM1T00012204_11_495_SE_2011-06-28T01h05m16_LVL2.fits',
                    'P12':lvl2path+'FM1T00012204/MIRFM1T00012204_12_495_SE_2011-06-28T01h21m02_LVL2.fits',
                    'P13':lvl2path+'FM1T00012204/MIRFM1T00012204_13_495_SE_2011-06-28T01h36m52_LVL2.fits',
                    'P14':lvl2path+'FM1T00012204/MIRFM1T00012204_14_495_SE_2011-06-28T01h52m42_LVL2.fits',
                    'P15':lvl2path+'FM1T00012204/MIRFM1T00012204_15_495_SE_2011-06-28T02h08m33_LVL2.fits',
                    'P16':lvl2path+'FM1T00012204/MIRFM1T00012204_16_495_SE_2011-06-28T02h24m23_LVL2.fits',
                    'P17':lvl2path+'FM1T00012204/MIRFM1T00012204_17_495_SE_2011-06-28T02h40m09_LVL2.fits',
                    'P18':lvl2path+'FM1T00012204/MIRFM1T00012204_18_495_SE_2011-06-28T02h55m54_LVL2.fits',
                    'P19':lvl2path+'FM1T00012204/MIRFM1T00012204_19_495_SE_2011-06-28T03h11m50_LVL2.fits',
                    'P20':lvl2path+'FM1T00012204/MIRFM1T00012204_20_495_SE_2011-06-28T03h27m35_LVL2.fits',
                    'P21':lvl2path+'FM1T00012204/MIRFM1T00012204_21_495_SE_2011-06-28T03h43m20_LVL2.fits',
                    'P22':lvl2path+'FM1T00012204/MIRFM1T00012204_22_495_SE_2011-06-28T03h59m11_LVL2.fits',
                    'P23':lvl2path+'FM1T00012204/MIRFM1T00012204_23_495_SE_2011-06-28T04h14m56_LVL2.fits'}
        bkg_file = lvl2path+'FM1T00012203/MIRFM1T00012203_1_495_SE_2011-06-27T22h03m11_LVL2.fits'

    elif position == 'middle':
        # Middle of FOV
        sci_imgs = {'P1':lvl2path+'FM1T00012308/MIRFM1T00012308_1_495_SE_2011-06-30T22h43m36_LVL2.fits',
                    'P2':lvl2path+'FM1T00012308/MIRFM1T00012308_2_495_SE_2011-06-30T22h59m21_LVL2.fits',
                    'P3':lvl2path+'FM1T00012308/MIRFM1T00012308_3_495_SE_2011-06-30T23h15m12_LVL2.fits',
                    'P4':lvl2path+'FM1T00012308/MIRFM1T00012308_4_495_SE_2011-06-30T23h31m02_LVL2.fits',
                    'P5':lvl2path+'FM1T00012308/MIRFM1T00012308_5_495_SE_2011-06-30T23h46m53_LVL2.fits',
                    'P6':lvl2path+'FM1T00012308/MIRFM1T00012308_6_495_SE_2011-07-01T00h02m38_LVL2.fits',
                    'P7':lvl2path+'FM1T00012308/MIRFM1T00012308_7_495_SE_2011-07-01T00h18m29_LVL2.fits',
                    'P8':lvl2path+'FM1T00012308/MIRFM1T00012308_8_495_SE_2011-07-01T00h34m19_LVL2.fits',
                    'P9':lvl2path+'FM1T00012308/MIRFM1T00012308_9_495_SE_2011-07-01T00h50m05_LVL2.fits',
                    'P10':lvl2path+'FM1T00012308/MIRFM1T00012308_10_495_SE_2011-07-01T01h06m05_LVL2.fits',
                    'P11':lvl2path+'FM1T00012308/MIRFM1T00012308_11_495_SE_2011-07-01T01h21m50_LVL2.fits',
                    'P12':lvl2path+'FM1T00012308/MIRFM1T00012308_12_495_SE_2011-07-01T01h37m36_LVL2.fits',
                    'P13':lvl2path+'FM1T00012308/MIRFM1T00012308_13_495_SE_2011-07-01T01h53m21_LVL2.fits',
                    'P14':lvl2path+'FM1T00012308/MIRFM1T00012308_14_495_SE_2011-07-01T02h09m17_LVL2.fits',
                    'P15':lvl2path+'FM1T00012308/MIRFM1T00012308_15_495_SE_2011-07-01T02h25m02_LVL2.fits',
                    'P16':lvl2path+'FM1T00012308/MIRFM1T00012308_16_495_SE_2011-07-01T02h40m52_LVL2.fits',
                    'P17':lvl2path+'FM1T00012308/MIRFM1T00012308_17_495_SE_2011-07-01T02h56m38_LVL2.fits',
                    'P18':lvl2path+'FM1T00012308/MIRFM1T00012308_18_495_SE_2011-07-01T03h12m23_LVL2.fits',
                    'P19':lvl2path+'FM1T00012308/MIRFM1T00012308_19_495_SE_2011-07-01T03h28m19_LVL2.fits',
                    'P20':lvl2path+'FM1T00012308/MIRFM1T00012308_20_495_SE_2011-07-01T03h44m04_LVL2.fits',
                    'P21':lvl2path+'FM1T00012308/MIRFM1T00012308_21_495_SE_2011-07-01T03h59m55_LVL2.fits',
                    'P22':lvl2path+'FM1T00012308/MIRFM1T00012308_22_495_SE_2011-07-01T04h15m40_LVL2.fits',
                    'P23':lvl2path+'FM1T00012308/MIRFM1T00012308_23_495_SE_2011-07-01T04h31m30_LVL2.fits',
                    'P24':lvl2path+'FM1T00012308/MIRFM1T00012308_24_495_SE_2011-07-01T04h47m16_LVL2.fits',
                    'P25':lvl2path+'FM1T00012308/MIRFM1T00012308_25_495_SE_2011-07-01T05h03m06_LVL2.fits',
                    'P26':lvl2path+'FM1T00012308/MIRFM1T00012308_26_495_SE_2011-07-01T05h18m52_LVL2.fits',
                    'P27':lvl2path+'FM1T00012308/MIRFM1T00012308_27_495_SE_2011-07-01T05h34m43_LVL2.fits'}
        bkg_file = lvl2path+'FM1T00012203/MIRFM1T00012203_1_495_SE_2011-06-27T22h03m11_LVL2.fits'

    elif position == 'right':
        # Right side of FOV
        sci_imgs = {'P1':lvl2path+'FM1T00012206/MIRFM1T00012206_1_495_SE_2011-06-28T04h51m37_LVL2.fits',
                    'P2':lvl2path+'FM1T00012206/MIRFM1T00012206_2_495_SE_2011-06-28T05h07m28_LVL2.fits',
                    'P3':lvl2path+'FM1T00012206/MIRFM1T00012206_3_495_SE_2011-06-28T05h23m13_LVL2.fits',
                    'P4':lvl2path+'FM1T00012206/MIRFM1T00012206_4_495_SE_2011-06-28T05h39m04_LVL2.fits',
                    'P5':lvl2path+'FM1T00012206/MIRFM1T00012206_5_495_SE_2011-06-28T05h54m49_LVL2.fits',
                    'P6':lvl2path+'FM1T00012206/MIRFM1T00012206_6_495_SE_2011-06-28T06h10m34_LVL2.fits',
                    'P7':lvl2path+'FM1T00012206/MIRFM1T00012206_7_495_SE_2011-06-28T06h26m30_LVL2.fits',
                    'P8':lvl2path+'FM1T00012206/MIRFM1T00012206_8_495_SE_2011-06-28T06h42m15_LVL2.fits',
                    'P9':lvl2path+'FM1T00012206/MIRFM1T00012206_9_495_SE_2011-06-28T06h58m11_LVL2.fits',
                    'P10':lvl2path+'FM1T00012206/MIRFM1T00012206_10_495_SE_2011-06-28T07h13m56_LVL2.fits',
                    'P11':lvl2path+'FM1T00012206/MIRFM1T00012206_11_495_SE_2011-06-28T07h29m41_LVL2.fits',
                    'P12':lvl2path+'FM1T00012206/MIRFM1T00012206_12_495_SE_2011-06-28T07h45m32_LVL2.fits',
                    'P13':lvl2path+'FM1T00012206/MIRFM1T00012206_13_495_SE_2011-06-28T08h01m17_LVL2.fits',
                    'P14':lvl2path+'FM1T00012206/MIRFM1T00012206_14_495_SE_2011-06-28T08h17m03_LVL2.fits',
                    'P15':lvl2path+'FM1T00012206/MIRFM1T00012206_15_495_SE_2011-06-28T08h32m58_LVL2.fits',
                    'P16':lvl2path+'FM1T00012206/MIRFM1T00012206_16_495_SE_2011-06-28T08h48m44_LVL2.fits',
                    'P17':lvl2path+'FM1T00012206/MIRFM1T00012206_17_495_SE_2011-06-28T09h04m34_LVL2.fits',
                    'P18':lvl2path+'FM1T00012206/MIRFM1T00012206_18_495_SE_2011-06-28T09h20m24_LVL2.fits',
                    'P19':lvl2path+'FM1T00012206/MIRFM1T00012206_19_495_SE_2011-06-28T09h36m10_LVL2.fits',
                    'P20':lvl2path+'FM1T00012206/MIRFM1T00012206_20_495_SE_2011-06-28T09h52m00_LVL2.fits',
                    'P21':lvl2path+'FM1T00012206/MIRFM1T00012206_21_495_SE_2011-06-28T10h07m46_LVL2.fits',
                    'P22':lvl2path+'FM1T00012206/MIRFM1T00012206_22_495_SE_2011-06-28T10h23m36_LVL2.fits',
                    'P23':lvl2path+'FM1T00012206/MIRFM1T00012206_23_495_SE_2011-06-28T10h39m27_LVL2.fits'}
        bkg_file = lvl2path+'FM1T00012203/MIRFM1T00012203_1_495_SE_2011-06-27T22h03m11_LVL2.fits'
    if pointing == 'all':
        return sci_imgs,bkg_file
    elif (pointing != 'all') & (output=='img'):
        from astropy.io import fits
        hdulist_sci,hdulist_bkg = fits.open(sci_imgs[pointing]), fits.open(bkg_file)
        sci_data,bkg_data = hdulist_sci[0].data[0,:,:],hdulist_bkg[0].data[0,:,:]
        hdulist_sci.close() ; hdulist_bkg.close()
        return sci_data,bkg_data
    elif (pointing != 'all') & (output=='filename'):
        return sci_imgs[pointing],bkg_file

def FM_MTS_800K_BB_MRS_OPT_08(lvl2path,wp_filter=None,output='img'):
    # MRS Wavelength Characterization
    # Wave-Pass (wp) filter can be Long-Wave-Pass (LWP) filter or Short-Wave-Pass (SWP) filter.
    sci_imgs = {'LWP':lvl2path+'FM1T00010841/MIRFM1T00010841_1_495_SE_2011-05-17T22h54m09_LVL2.fits',
                'LWP_HOLE':lvl2path+'FM1T00010842/MIRFM1T00010842_1_495_SE_2011-05-17T23h35m45_LVL2.fits',
                'SWP':lvl2path+'FM1T00010950/MIRFM1T00010950_1_495_SE_2011-05-20T09h25m54_LVL2.fits',
                'SWP_HOLE':lvl2path+'FM1T00010951/MIRFM1T00010951_1_495_SE_2011-05-20T10h16m00_LVL2.fits'}
    bkg_imgs = {'LWP':lvl2path+'FM1T00012203/MIRFM1T00012203_1_495_SE_2011-06-27T22h03m11_LVL2.fits',
                'SWP':lvl2path+'FM1T00010953/MIRFM1T00010953_1_495_SE_2011-05-20T11h50m14_LVL2.fits'}
    if output == 'img':
        from astropy.io import fits
        hdulist_filter,hdulist_hole,hdulist_bkg = fits.open(sci_imgs[wp_filter]),fits.open(sci_imgs[wp_filter+'_HOLE']),fits.open(bkg_imgs[wp_filter])
        filter_data,hole_data,bkg_data = hdulist_filter[0].data[0,:,:],hdulist_hole[0].data[0,:,:],hdulist_bkg[0].data[0,:,:]
        return filter_data,hole_data,bkg_data
    elif output == 'filename':
            return sci_imgs[wp_filter],sci_imgs[wp_filter+'_HOLE'],bkg_imgs[wp_filter]



#-- CV1RR & CV2 & CV3 data
def CV_800K_BB_MRS_OPT_02_obs(dataDir,band,campaign=None,pointing='all',output='img'):
    """ load MIRIM PSFs band 1A/2A"""
    import os
    import glob

    files = [os.path.basename(i) for i in glob.glob(dataDir+'*')]
    subchannels = ['SHORT','MED','LONG']
    MIRIMPSF_dictionary = {}
    if campaign == 'CV2':
        # 16 pointings in total
        pointings = ['P'+str(i) for i in range(17)]
        for point in pointings:
            mylist = []
            for subchannel in subchannels:
                sub = 'MIRM0363-{}-{}'.format(point,subchannel)
                mylist.extend([s for s in files if sub in s])
            MIRIMPSF_dictionary[campaign+'_'+point] = list(mylist)
        if pointing == 'all':
            return MIRIMPSF_dictionary
        elif (pointing != 'all') & (output=='img'):
            from astropy.io import fits
            if band in ['1A','2A']: sci_idx = 1
            elif band in ['1B','2B']:sci_idx = 5
            elif band in ['1C','2C']:sci_idx = 9
            elif band in ['3A','4A']:sci_idx = 0
            elif band in ['3B','4B']:sci_idx = 4
            elif band in ['3C','4C']:sci_idx = 8
            hdulist_sci,hdulist_bkg = fits.open(dataDir+MIRIMPSF_dictionary[campaign+'_'+pointing][sci_idx]), fits.open(dataDir+MIRIMPSF_dictionary[campaign+'_'+pointing][sci_idx+2])
            sci_data,bkg_data = hdulist_sci[0].data[0,:,:],hdulist_bkg[0].data[0,:,:]
            hdulist_sci.close() ; hdulist_bkg.close()
            return sci_data,bkg_data
    elif campaign == 'CV3':
        # 16 pointings in total
        pointings = ['Q'+str(i) for i in range(17)]
        for point in pointings:
            sub = 'MIRM103-{}-SHORT'.format(point)
            MIRIMPSF_dictionary[campaign+'_'+point] = [s for s in files if sub in s]
        if pointing == 'all':
            return MIRIMPSF_dictionary
        elif (pointing != 'all') & (output=='img'):
            from astropy.io import fits
            if band in ['1A','2A']: sci_idx = 1
            elif band in ['3A','4A']:sci_idx = 0
            hdulist_sci,hdulist_bkg = fits.open(dataDir+MIRIMPSF_dictionary[campaign+'_'+pointing][sci_idx]), fits.open(dataDir+MIRIMPSF_dictionary[campaign+'_'+pointing][sci_idx+2])
            sci_data,bkg_data = hdulist_sci[0].data[0,:,:],hdulist_bkg[0].data[0,:,:]
            hdulist_sci.close() ; hdulist_bkg.close()
            return sci_data,bkg_data
        """
        Dictionary keys are equivalent to PSF measurements in CV2 and CV3 tests (with different pointings).

        Dictionary indeces within keys, for CV2 obs. are equivalent to:
        [0,1,2,3] : SHORT_494,SHORT_495,SHORTB_494,SHORTB_495
        [4,5,6,7] : MED_494,MED_495,MEDB_494,MEDB_495
        [8,9,10,11] : LONG_494,LONG_495,LONGB_494,LONGB_495

        Dictionary indeces within keys, for CV3 obs. are equivalent to:
        [0,1,2,3] : SHORT_494,SHORT_495,SHORTB_494,SHORTB_495
        There are only SHORT CV3 PSF measurements
        """
    elif campaign == 'CV1RR':
        sci_imgs = {'P0':dataDir+'MIRM03639-P0-3296092823_1_495_SE_2013-10-23T09h57m38_LVL2.fits',
                    'P1':dataDir+'MIRM03639-P1-3296094802_1_495_SE_2013-10-23T10h28m43_LVL2.fits',
                    'P2':dataDir+'MIRM03639-P2-3296100758_1_495_SE_2013-10-23T11h00m44_LVL2.fits',
                    'P3':dataDir+'MIRM03639-P3-3296103650_1_495_SE_2013-10-23T11h21m49_LVL2.fits',
                    'P4':dataDir+'MIRM03639-P4-3296110642_1_495_SE_2013-10-23T11h44m59_LVL2.fits',
                    'P5':dataDir+'MIRM03639-P5-3296112357_1_495_SE_2013-10-23T12h07m05_LVL2.fits',
                    'P6':dataDir+'MIRM03639-P6-3296114131_1_495_SE_2013-10-23T12h07m21_LVL2.fits',
                    'P7':dataDir+'MIRM03639-P7-3296115814_1_495_SE_2013-10-23T12h28m20_LVL2.fits',
                    'P8':dataDir+'MIRM03639-P8-3296121530_1_495_SE_2013-10-23T12h49m36_LVL2.fits',
                    'P9':dataDir+'MIRM03639-P9-3296123231_1_495_SE_2013-10-23T13h12m55_LVL2.fits',
                    'P10':dataDir+'MIRM03639-P10-3296124946_1_495_SE_2013-10-23T13h13m07_LVL2.fits',
                    'P11':dataDir+'MIRM03639-P11-3296130652_1_495_SE_2013-10-23T13h33m51_LVL2.fits',
                    'P12':dataDir+'MIRM03639-P12-3296132334_1_495_SE_2013-10-23T13h54m37_LVL2.fits',
                    'P13':dataDir+'MIRM03639-P13-3296134016_1_495_SE_2013-10-23T14h16m47_LVL2.fits',
                    'P14':dataDir+'MIRM03639-P14-3296135712_1_495_SE_2013-10-23T14h39m58_LVL2.fits',
                    'P15':dataDir+'MIRM03639-P15-3296141418_1_495_SE_2013-10-23T14h40m08_LVL2.fits',
                    'P16':dataDir+'MIRM03639-P16-3296143123_1_495_SE_2013-10-23T15h17m44_LVL2.fits'}
        if pointing == 'all':
            return sci_imgs
        elif (pointing != 'all') & (output=='img'):
            from astropy.io import fits
            hdulist_sci = fits.open(sci_imgs[pointing])
            sci_data = hdulist_sci[0].data[0,:,:]
            hdulist_sci.close()
            return sci_data

#-- OTIS campaign
def OTIS_ASPA_semiextended_source(lvl2path,band,pointing=None,output='img'):
    if pointing == 'v03':
        # source centroid is at a higher beta value
        sci_imgs = {"1A":lvl2path +'MIRM32313-SS-V03-7249024654_1_495_SE_2017-09-06T02h54m21_LVL2.fits',
                    "1B":lvl2path +'MIRM32313-MM-V03-7249025513_1_495_SE_2017-09-06T03h03m01_LVL2.fits',
                    "1C":lvl2path +'MIRM32313-LL-V03-7249030323_1_495_SE_2017-09-06T03h08m31_LVL2.fits',
                    "2A":lvl2path +'MIRM32313-SS-V03-7249024654_1_495_SE_2017-09-06T02h54m21_LVL2.fits',
                    "2B":lvl2path +'MIRM32313-MM-V03-7249025513_1_495_SE_2017-09-06T03h03m01_LVL2.fits',
                    "2C":lvl2path +'MIRM32313-LL-V03-7249030323_1_495_SE_2017-09-06T03h08m31_LVL2.fits',
                    "3A":lvl2path +'MIRM32313-SS-V03-7249024654_1_494_SE_2017-09-06T02h54m21_LVL2.fits',
                    "3B":lvl2path +'MIRM32313-MM-V03-7249025513_1_494_SE_2017-09-06T03h03m01_LVL2.fits',
                    "3C":lvl2path +'MIRM32313-LL-V03-7249030323_1_494_SE_2017-09-06T03h08m31_LVL2.fits',
                    "4A":lvl2path +'MIRM32313-SS-V03-7249024654_1_494_SE_2017-09-06T02h54m21_LVL2.fits',
                    "4B":lvl2path +'MIRM32313-MM-V03-7249025513_1_494_SE_2017-09-06T03h03m01_LVL2.fits',
                    "4C":lvl2path +'MIRM32313-LL-V03-7249030323_1_494_SE_2017-09-06T03h08m31_LVL2.fits'}

        bkg_imgs = {"1A":lvl2path +'MIRM32313-SS-V03B-7249024337_1_495_SE_2017-09-06T02h49m41_LVL2.fits',
                    "1B":lvl2path +'MIRM32313-MM-V03B-7249025018_1_495_SE_2017-09-06T02h57m31_LVL2.fits',
                    "1C":lvl2path +'MIRM32313-LL-V03B-7249025837_1_495_SE_2017-09-06T03h05m21_LVL2.fits',
                    "2A":lvl2path +'MIRM32313-SS-V03B-7249024337_1_495_SE_2017-09-06T02h49m41_LVL2.fits',
                    "2B":lvl2path +'MIRM32313-MM-V03B-7249025018_1_495_SE_2017-09-06T02h57m31_LVL2.fits',
                    "2C":lvl2path +'MIRM32313-LL-V03B-7249025837_1_495_SE_2017-09-06T03h05m21_LVL2.fits',
                    "3A":lvl2path +'MIRM32313-SS-V03B-7249024337_1_494_SE_2017-09-06T02h49m41_LVL2.fits',
                    "3B":lvl2path +'MIRM32313-MM-V03B-7249025018_1_494_SE_2017-09-06T02h57m31_LVL2.fits',
                    "3C":lvl2path +'MIRM32313-LL-V03B-7249025837_1_494_SE_2017-09-06T03h05m21_LVL2.fits',
                    "4A":lvl2path +'MIRM32313-SS-V03B-7249024337_1_494_SE_2017-09-06T02h49m41_LVL2.fits',
                    "4B":lvl2path +'MIRM32313-MM-V03B-7249025018_1_494_SE_2017-09-06T02h57m31_LVL2.fits',
                    "4C":lvl2path +'MIRM32313-LL-V03B-7249025837_1_494_SE_2017-09-06T03h05m21_LVL2.fits'}
    elif pointing == 'v05':
        # source centroid is at a lower beta value
        sci_imgs = {"1A":lvl2path +'MIRM32313-SS-V05-7249031538_1_495_SE_2017-09-06T03h23m41_LVL2.fits',
                    "1B":lvl2path +'MIRM32313-MM-V05-7249032353_1_495_SE_2017-09-06T03h31m21_LVL2.fits',
                    "1C":lvl2path +'MIRM32313-LL-V05-7249033205_1_495_SE_2017-09-06T03h37m41_LVL2.fits',
                    "2A":lvl2path +'MIRM32313-SS-V05-7249031538_1_495_SE_2017-09-06T03h23m41_LVL2.fits',
                    "2B":lvl2path +'MIRM32313-MM-V05-7249032353_1_495_SE_2017-09-06T03h31m21_LVL2.fits',
                    "2C":lvl2path +'MIRM32313-LL-V05-7249033205_1_495_SE_2017-09-06T03h37m41_LVL2.fits',
                    "3A":lvl2path +'MIRM32313-SS-V05-7249031538_1_494_SE_2017-09-06T03h23m41_LVL2.fits',
                    "3B":lvl2path +'MIRM32313-MM-V05-7249032353_1_494_SE_2017-09-06T03h31m21_LVL2.fits',
                    "3C":lvl2path +'MIRM32313-LL-V05-7249033205_1_494_SE_2017-09-06T03h37m41_LVL2.fits',
                    "4A":lvl2path +'MIRM32313-SS-V05-7249031538_1_494_SE_2017-09-06T03h23m41_LVL2.fits',
                    "4B":lvl2path +'MIRM32313-MM-V05-7249032353_1_494_SE_2017-09-06T03h31m21_LVL2.fits',
                    "4C":lvl2path +'MIRM32313-LL-V05-7249033205_1_494_SE_2017-09-06T03h37m41_LVL2.fits'}

        bkg_imgs = {"1A":lvl2path +'MIRM32313-SS-V05B-7249031039_1_495_SE_2017-09-06T03h18m01_LVL2.fits',
                    "1B":lvl2path +'MIRM32313-MM-V05B-7249031906_1_495_SE_2017-09-06T03h26m01_LVL2.fits',
                    "1C":lvl2path +'MIRM32313-LL-V05B-7249032716_1_495_SE_2017-09-06T03h34m51_LVL2.fits',
                    "2A":lvl2path +'MIRM32313-SS-V05B-7249031039_1_495_SE_2017-09-06T03h18m01_LVL2.fits',
                    "2B":lvl2path +'MIRM32313-MM-V05B-7249031906_1_495_SE_2017-09-06T03h26m01_LVL2.fits',
                    "2C":lvl2path +'MIRM32313-LL-V05B-7249032716_1_495_SE_2017-09-06T03h34m51_LVL2.fits',
                    "3A":lvl2path +'MIRM32313-SS-V05B-7249031039_1_494_SE_2017-09-06T03h18m01_LVL2.fits',
                    "3B":lvl2path +'MIRM32313-MM-V05B-7249031906_1_494_SE_2017-09-06T03h26m01_LVL2.fits',
                    "3C":lvl2path +'MIRM32313-LL-V05B-7249032716_1_494_SE_2017-09-06T03h34m51_LVL2.fits',
                    "4A":lvl2path +'MIRM32313-SS-V05B-7249031039_1_494_SE_2017-09-06T03h18m01_LVL2.fits',
                    "4B":lvl2path +'MIRM32313-MM-V05B-7249031906_1_494_SE_2017-09-06T03h26m01_LVL2.fits',
                    "4C":lvl2path +'MIRM32313-LL-V05B-7249032716_1_494_SE_2017-09-06T03h34m51_LVL2.fits'}
    if output == 'filename':
        return sci_imgs[band],bkg_imgs[band]
    elif output == 'img':
        from astropy.io import fits
        hdulist_sci,hdulist_bkg = fits.open(sci_imgs[band]), fits.open(bkg_imgs[band])
        sci_data,bkg_data = hdulist_sci[0].data[0,:,:],hdulist_bkg[0].data[0,:,:]
        hdulist_sci.close() ; hdulist_bkg.close()
        return sci_data,bkg_data
