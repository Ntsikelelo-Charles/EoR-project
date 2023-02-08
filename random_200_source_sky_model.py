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
from scipy.interpolate import interp1d

def beam_interp_func(theta, phi, data):
            # convert to radians
            theta = copy.copy(theta) * np.pi / 180.0
            phi = copy.copy(phi) * np.pi / 180.0
            shape = theta.shape
            beam_interp = [healpy.get_interp_val(m, theta.ravel(), phi.ravel(), lonlat=False).reshape(shape) for m in data]
            return np.array(beam_interp)
        
def lookupNearest(theta_s,phi_s,theta,phi):
  
  index = np.where((np.abs(theta-theta_s)==np.min(np.abs(theta-theta_s))) & 
                   (np.abs(phi-phi_s)==np.min(np.abs(phi-phi_s))))

  return index

# generate random sky models
RA_inner=[]
DE_inner=[]
while len(RA_inner)<100:
   RA_cen=np.random.randint(-55,55,1)
   DE_cen=np.random.randint(-35,35,1)
   r=np.sqrt(RA_cen**2+DE_cen**2)
   if r<30:
        RA_inner.append(RA_cen)
        DE_inner.append(DE_cen)
        
RA_outer=[]
DE_outer=[]
while len(RA_outer)<100:
   RA_cen=np.random.randint(-55,55,1)
   DE_cen=np.random.randint(-35,35,1)
   r=np.sqrt(RA_cen**2+DE_cen**2)
   if r>40:
        RA_outer.append(RA_cen)
        DE_outer.append(DE_cen)
        
plt.plot(RA_inner,DE_inner,'ob')        
plt.plot(RA_outer,DE_outer,'or')
RA_inner=np.array(RA_inner)[:,0]
DE_inner=np.array(DE_inner)[:,0]

RA_outer=np.array(RA_outer)[:,0]
DE_outer=np.array(DE_outer)[:,0]

path_to_rand='/vault-ike/ntsikelelo/simulated_ms_files/Artificial_sky_models/20_random_distributed_sources/'
np.save(path_to_rand+'RA_cen.npy',RA_inner)
np.save(path_to_rand+'DE_cen.npy',DE_inner)
np.save(path_to_rand+'RA_cen_outer.npy',RA_outer)
np.save(path_to_rand+'DE_cen_outer.npy',DE_outer)


# pointing direction
ms = table('/home/ntsikelelo/zen.2458115.31193.xx.HH.uvR.ms.split')
fld = table('/home/ntsikelelo/zen.2458115.31193.xx.HH.uvR.ms.split'+'::FIELD')
radec0 = fld.getcol('PHASE_DIR').squeeze().reshape(1,2)
radec0 = np.tile(radec0, (60,1))
fld.close() 
point_ra=np.rad2deg(radec0[0,0])
point_dec=np.rad2deg(radec0[0,1])
path_to_rand='/vault-ike/ntsikelelo/simulated_ms_files/Artificial_sky_models/20_random_distributed_sources/'

RA_cen=np.zeros(shape=(200))
DE_cen=np.zeros(RA_cen.shape)
RA_cen[0:100]=np.load(path_to_rand+'RA_cen.npy')
DE_cen[0:100]=np.load(path_to_rand+'DE_cen.npy')
RA_cen[100:200]=np.load(path_to_rand+'RA_cen_outer.npy')
DE_cen[100:200]=np.load(path_to_rand+'DE_cen_outer.npy')
DE=DE_cen+point_dec
RA=RA_cen+point_ra
flux=np.ones(DE.shape)
flux[0:100]=flux[100:200]*1e-2
flux[100:200]=flux[100:200]*1e2
spectral_index=np.ones(DE.shape)*(-0.7)


for o in range (flux.shape[0]):
    sky_model= open(path_to_rand+"model_sky_gains_30_rand_original_full.txt","w+")
    sky_model.write("#format:name ra_d dec_d i q u v emaj_d emin_d pa_d spi freq0"+"\n")
    sources=[]
    for k in range (flux.shape[0]):
        sources.append("SRC"+str(k)+" "+str(RA[k])+" "+str(DE[k])+" "+str(flux[k])+" "+"0"+" "+"0"+" "+"0"+" "+"0"+" "+"0"+" "+"0"+" "+str(-0.7)+" "+"150e6")
    for i in range(len(sources)):
        sky_model.write(sources[i]+"\n")
    sky_model.close()  

#  #HERA beam
# print("Now simulating with HERA_beam")
# uvb = UVBeam()
# uvb.read_beamfits('/home/ntsikelelo/NF_HERA_power_beam_healpix128.fits')
# pol_ind = np.where(uvb.polarization_array ==-6)[0][0]
# beam_maps = np.abs(uvb.data_array[0, 0, pol_ind, :, :])
# beam_freqs = uvb.freq_array.squeeze() / 1e6
# Nbeam_freqs = len(beam_freqs)
# beam_nside = healpy.npix2nside(beam_maps.shape[1])

# theta = np.sqrt( (RA - point_ra)**2 + (DE - point_dec)**2 ) # center origin at the array dec and RA
# phi = np.arctan2((DE-point_dec), (RA-point_ra)) + np.pi

# pb = beam_interp_func(theta, phi,beam_maps)

# data_freqs=np.linspace(100,200,1024)
# Ndata_freqs = len(data_freqs)

# beam_values=np.zeros(shape=(1024, DE_cen.shape[0]))
# for source in range (pb.shape[1]):
#     pb_interp = interpolate.interp1d(beam_freqs,pb[:,source], kind='cubic')
#     beam_values[:,source]=pb_interp(data_freqs)

# gains=np.zeros((1,8,1024,DE_cen.shape[0],1), dtype=complex)
# gains[:,:,:,:,0]=np.sqrt(beam_values)
# np.save(path_to_rand+'Gains_HERA_Fagnoni_beam_full.npy', gains)

n_chan=1024
#         mutual coupling beams
beam_name=np.array(['06','22','26'])

for k in range (len(beam_name)):
    print("Now simulating beam "+beam_name[k])
    uvb = UVBeam()
    uvb.read_beamfits('/home/ntsikelelo/HERA-Beams/NF_CrossCouplingBeams/NF_HERA-Dipole_ccbeam_port'+beam_name[k]+'_E-field_phasecorrected.fits')
    uvb.peak_normalize()
    print("beam normalised")

    beam_maps = uvb.data_array[0, 0, 0, :, :, :]
    beam_freqs = uvb.freq_array.squeeze() / 1e6
    Nbeam_freqs = len(beam_freqs)
    phi_grid,theta_grid=np.meshgrid(uvb.axis1_array,uvb.axis2_array) ## phi,theta
    
    theta = np.sqrt( (RA_cen)**2 + (DE_cen)**2 ) # source origin at the array dec and RA
    phi = np.arctan2((DE_cen), (RA_cen)) + np.pi

  
    pb=np.zeros((beam_freqs.shape[0],DE_cen.shape[0]),dtype=complex)
    for source in range (len(phi)):
        index=lookupNearest(np.deg2rad(theta[source]),np.deg2rad(phi[source]),theta_grid,phi_grid)
        index=np.array(index)[:,0]
        pb[:,source]=beam_maps[:,index[0],index[1]] 
        
        ##phase at zenith    
    theta_0 = np.sqrt( (0)**2 + (0)**2 ) # center origin at the array dec and RA
    phi_0 = np.arctan2((0), (0)) + np.pi
    index_0=lookupNearest(np.deg2rad(theta_0),np.deg2rad(phi_0),theta_grid,phi_grid)
    index_0=np.array(index_0)[:,0]
    pb_0=beam_maps[:,index_0[0],index_0[1]]

    for source in range(DE_cen.shape[0]):
        pb[:,source]=pb[:,source]*(np.conjugate(pb_0)/np.abs(pb_0))

    data_freqs=np.linspace(100,200,n_chan)
    Ndata_freqs = len(data_freqs)
    beam_values=np.zeros(shape=(n_chan, DE_cen.shape[0]),dtype=complex)

    print("interpolating over freq")
    for source in range (pb.shape[1]):
        pb_interp_abs = interpolate.interp1d(beam_freqs,np.abs(pb[:,source]), kind='cubic')
        pb_interp_phase = interpolate.interp1d(beam_freqs,pb[:,source], kind='cubic')
        phase=np.angle(pb_interp_phase(data_freqs))
        amp=pb_interp_abs(data_freqs)
        beam_values[:,source]=amp*np.exp(1j*phase)

    gains=np.zeros((1,1,n_chan,DE_cen.shape[0],1), dtype=complex)
    gains[:,:,:,:,0]=beam_values
    np.save(path_to_rand+'sky_model_mutual_'+beam_name[k]+'_full.npy', gains)


HERA_beam_mutual_06=np.load(path_to_rand+'sky_model_mutual_06_full.npy')
HERA_beam_mutual_22=np.load(path_to_rand+'sky_model_mutual_22_full.npy')
HERA_beam_mutual_26=np.load(path_to_rand+'sky_model_mutual_26_full.npy')

## mutual_coupling_gain
mutual_gains=np.zeros(shape=(1,8,1024,HERA_beam_mutual_06.shape[3],1),dtype=complex)
mutual_gains[:,0,:,:,:]=HERA_beam_mutual_06
mutual_gains[:,1,:,:,:]=HERA_beam_mutual_26
mutual_gains[:,2,:,:,:]=HERA_beam_mutual_26
mutual_gains[:,3,:,:,:]=HERA_beam_mutual_22
mutual_gains[:,4,:,:,:]=HERA_beam_mutual_22
mutual_gains[:,5,:,:,:]=HERA_beam_mutual_22
mutual_gains[:,6,:,:,:]=HERA_beam_mutual_22
mutual_gains[:,7,:,:,:]=HERA_beam_mutual_22

np.save(path_to_rand+'Gains_mutual_coupling_EEC_ECC_CCC_full.npy',mutual_gains)