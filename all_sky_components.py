from scipy.misc import factorial as fac
from sys import argv
import time
from mpl_toolkits.mplot3d import Axes3D  
import numpy as np
import matplotlib.pyplot as plt
from pyrap.tables import table 
import astropy.io.fits as fits
from astropy import wcs
from pyuvdata import UVBeam
import os
import sys
import glob
import argparse
import shutil
import copy
import healpy
import scipy.stats as stats
import pytz
import datetime
import ephem
from scipy import interpolate
from astropy.time import Time
import pylab as plt
from astropy.table import Table
from scipy import interpolate
import healpy as hp
import astropy 
from astropy import units as u
from astropy.coordinates import SkyCoord


## diffuse emission
haslam_map=healpy.fitsfunc.read_map('/home/ntsikelelo/my_files/GLEAM_models/GLEAM_beam_corrected_models/haslam408_dsds_Remazeilles2014.fits')
NSIDE=512
Npix=healpy.nside2npix(NSIDE)
temperature=haslam_map.T

f_408_GHz=0.4 #GHz
maj0=56.0*60 #seconds
min0=56.0*60
maj0_rad=np.deg2rad(56*(0.5/30))
min0_rad=np.deg2rad(56*(0.5/30))
constant=(1.222*10**3)/(f_408_GHz**2*maj0*min0)
Omega=healpy.nside2resol(NSIDE)**2
B=temperature*(1/constant)


f_150=150.0 #MHz
f_408=408.0 #MHz
Flux_408_MHz=B*Omega
Flux_150_MHz=Flux_408_MHz*(f_150/f_408)**(-0.7)
pix=np.arange(Npix)
theta,phi=healpy.pix2ang(NSIDE,pix,lonlat=True)
from astropy import units as u
theta_units = theta * u.deg
phi_units = phi * u.deg
c = SkyCoord(frame="galactic", l=theta_units, b=phi_units)
dec=c.cirs.dec.deg
ra=c.cirs.ra.deg
np.save('/vault-ike/ntsikelelo/diffuse_emmission_flux_haslam.npy',Flux_150_MHz)
np.save('/vault-ike/ntsikelelo/diffuse_emmission_RA_haslam.npy',ra)
np.save('/vault-ike/ntsikelelo/diffuse_emmission_DEC_haslam.npy',dec)
print("diffuse model done")


## Fornax
ms = table('/home/ntsikelelo/zen.2458115.31193.xx.HH.uvR.ms.split')
fld = table('/home/ntsikelelo/zen.2458115.31193.xx.HH.uvR.ms.split'+'::FIELD')
radec0 = fld.getcol('PHASE_DIR').squeeze().reshape(1,2)
radec0 = np.tile(radec0, (60,1))
fld.close() 
point_ra=np.rad2deg(radec0[0,0])
point_dec=np.rad2deg(radec0[0,1])  
fornax_field="/home/ntsikelelo/fornaxA.cl.fits"
hdu=fits.open(fornax_field)
head = hdu[0].header
npix1 = head["NAXIS1"]
npix2 = head["NAXIS2"]

 # get WCS
w = wcs.WCS(fornax_field) 


if head['CTYPE3'] == 'FREQ':
            freq_ax = 3
            stok_ax = 4
else:
            freq_ax = 4
            stok_ax = 3
            
npix1 = head["NAXIS1"]
npix2 = head["NAXIS2"]
nstok = head["NAXIS{}".format(stok_ax)]
nfreq = head["NAXIS{}".format(freq_ax)]


lon_arr, lat_arr = np.meshgrid(np.arange(npix1), np.arange(npix2))
lon, lat, s, f = w.all_pix2world(lon_arr.ravel(), lat_arr.ravel(), 0, 0, 0)
lon = lon.reshape(npix2, npix1)
lat = lat.reshape(npix2, npix1)
RA=lon
DEC=lat
data = hdu[0].data
np.save('/vault-ike/ntsikelelo/Fornax_RA.npy',RA.ravel())
np.save('/vault-ike/ntsikelelo/Fornax_DEC.npy',DEC.ravel())
data_f=np.zeros(shape=(1024,RA.ravel().shape[0]))
for freq in range (1024):
    data_f[freq,:]=data[freq,0,:,:].ravel()
np.save('/vault-ike/ntsikelelo/Fornax_Flux.npy',data_f)
print("Fornax model done")



#1 Jy = 10−26 W m−2 Hz−1
# 21 cm
from astropy.coordinates import SkyCoord  
import astropy.units as u


flux=np.zeros(shape=(103,786432))
NSIDE=256
Omega=healpy.nside2resol(256)**2
Npix=healpy.nside2npix(NSIDE)
pix=np.arange(Npix)
theta,phi=healpy.pix2ang(NSIDE,pix,lonlat=True)
from astropy import units as u
theta_units = theta * u.deg
phi_units = phi * u.deg
c = SkyCoord(frame="galactic", l=theta_units, b=phi_units)
dec=c.cirs.dec.deg
ra=c.cirs.ra.deg
np.save('/vault-ike/ntsikelelo/21_cm_images_all_sky_RA.npy',ra)
np.save('/vault-ike/ntsikelelo/21_cm_images_all_sky_DEC.npy',dec)
Cv_Jy=10
#1 Jy = 10−26 W m−2 Hz−1
for freq in range (103):
    hdulist = fits.open('/home/ntsikelelo/maps_256_Nkc50_seed1001/map_nside256_seed1001_f'+str(freq)+'.fits')
    T=hdulist[0].data 
    f=(110+freq*(100/1024))*1e6
    c=3e8
    lambda_=c/f
    k_B=1.38064852e-23 
    constant=(2*k_B*Omega)/lambda_**2
    flux[freq,:]=(T*constant)*10**26 ## conversion to Jy
np.save('/vault-ike/ntsikelelo/21_cm_images_all_sky.npy',flux)
print("21 cm model done")