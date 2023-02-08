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
from scipy.interpolate import interp1d

field_off_set=np.array([0,40,105])
field_name=np.array(['0h','05h','1h'])
path_to_GLEAM='/vault-ike/ntsikelelo/simulated_ms_files/GLEAM_simulations_29_DD_gain_corruption/Gains/'
n_chan=1024

def lookupNearest(theta_s,phi_s,theta,phi):
  
  index = np.where((np.abs(theta-theta_s)==np.min(np.abs(theta-theta_s))) & 
                   (np.abs(phi-phi_s)==np.min(np.abs(phi-phi_s))))

  return index
def beam_interp_func(theta, phi, data):
            # convert to radians
            theta = copy.copy(theta) * np.pi / 180.0
            phi = copy.copy(phi) * np.pi / 180.0
            shape = theta.shape
            beam_interp = [healpy.get_interp_val(m, theta.ravel(), phi.ravel(), lonlat=False).reshape(shape) for m in data]
            return np.array(beam_interp)
        
        
        
models=np.array(["diffuse_emission","GLEAM","Fornax"])
for k in range (len(models)):
    sky_model_type=models[k]
    for fi in range(len(field_name)):


        # pointing direction
        ms = table('/home/ntsikelelo/zen.2458115.31193.xx.HH.uvR.ms.split')
        fld = table('/home/ntsikelelo/zen.2458115.31193.xx.HH.uvR.ms.split'+'::FIELD')
        radec0 = fld.getcol('PHASE_DIR').squeeze().reshape(1,2)
        radec0 = np.tile(radec0, (60,1)) 
        fld.close() 
        point_ra=np.rad2deg(radec0[0,0])+field_off_set[fi]
        point_dec=np.rad2deg(radec0[0,1])   


        RA_diff_all=np.load('/vault-ike/ntsikelelo/diffuse_emmission_RA_haslam.npy')
        DE_diff_all=np.load('/vault-ike/ntsikelelo/diffuse_emmission_DEC_haslam.npy')
        index_diff=np.where((RA_diff_all<point_ra+50) &(RA_diff_all>point_ra-50) & (DE_diff_all<point_dec+35) & (DE_diff_all>point_dec-35)) 
        RA_diff=RA_diff_all[index_diff]
        DE_diff=DE_diff_all[index_diff]


         ## Fornax
        RA_fornax=np.load('/vault-ike/ntsikelelo/Fornax_RA.npy')
        DE_fornax=np.load('/vault-ike/ntsikelelo/Fornax_DEC.npy')


       ## GLEAM sources
        gleam_catalogue = Table.read('/home/ntsikelelo/my_files/GLEAM_models/GLEAM_models_var/GLEAM_EGC_v2.fits') 
        original_flux=gleam_catalogue['int_flux_151']
        original_flux[np.where(np.isnan(original_flux)==True)]=0

        indices1=np.where((gleam_catalogue['RAJ2000']<point_ra+50) &(gleam_catalogue['RAJ2000']>point_ra-50)& (gleam_catalogue['DEJ2000']<point_dec+35) & (gleam_catalogue['DEJ2000']>point_dec-35) & (original_flux>0.2))

        DE1=gleam_catalogue['DEJ2000'][indices1]
        RA1=gleam_catalogue['RAJ2000'][indices1]

        RA=[]
        DE=[]

        if sky_model_type =="Fornax" :      
            for k in range(len(RA_fornax)):
                RA.append(RA_fornax[k]) 
            for k in range(len(DE_fornax)):
                   DE.append(DE_fornax[k])


        if sky_model_type =="GLEAM" :    

            for k in range(len(RA1)):
                RA.append(RA1[k])   
            for k in range(len(DE1)):
                 DE.append(DE1[k])   

        if sky_model_type =="diffuse_emission" :   
            for k in range(len(RA_diff)):
                RA.append(RA_diff[k])
            for k in range(len(DE_diff)):
                   DE.append(DE_diff[k])


        RA=np.array(RA) 
        DE=np.array(DE)
        
        print ('All sky-components loaded, number of sources = ' +str(len(RA)))
        RA_cen=RA-point_ra
        DE_cen=DE-point_dec
        
                #HERA beam
        print("Now simulating field "+field_name[fi]+" and "+sky_model_type+" with HERA_beam")
        uvb = UVBeam()
        uvb.read_beamfits('/home/ntsikelelo/NF_HERA_power_beam_healpix128.fits')
        pol_ind = np.where(uvb.polarization_array ==-6)[0][0]
        beam_maps = np.abs(uvb.data_array[0, 0, pol_ind, :, :])
        beam_freqs = uvb.freq_array.squeeze() / 1e6
        Nbeam_freqs = len(beam_freqs)
        beam_nside = healpy.npix2nside(beam_maps.shape[1])

        theta = np.sqrt( (RA - point_ra)**2 + (DE - point_dec)**2 ) # center origin at the array dec and RA
        phi = np.arctan2((DE-point_dec), (RA-point_ra)) + np.pi

        pb = beam_interp_func(theta, phi,beam_maps)
      
        data_freqs=np.linspace(100,200,n_chan)
        Ndata_freqs = len(data_freqs)

        beam_values=np.zeros(shape=(n_chan, DE_cen.shape[0]))
        for source in range (pb.shape[1]):
            pb_interp = interpolate.interp1d(beam_freqs,pb[:,source], kind='cubic')
            beam_values[:,source]=pb_interp(data_freqs)

        gains=np.zeros(shape=(1,8,n_chan,DE_cen.shape[0],1), dtype=complex)
        gains[:,:,:,:,0]=np.sqrt(beam_values)
        np.save(path_to_GLEAM+'Gains_HERA_Fagnoni_beam'+field_name[fi]+'_'+sky_model_type+'.npy', gains)

     
#         mutual coupling beams
        beam_name=np.array(['06','22','26'])

        for k in range (len(beam_name)):
            print("Now simulating beam "+beam_name[k]+sky_model_type)
            uvb = UVBeam()
            uvb.read_beamfits('/home/ntsikelelo/HERA-Beams/NF_CrossCouplingBeams/NF_HERA-Dipole_ccbeam_port'+beam_name[k]+'_E-field_phasecorrected.fits')
            uvb.peak_normalize()
            print("beam normalised")

            beam_maps = uvb.data_array[0, 0, 0, :, :, :]
            beam_freqs = uvb.freq_array.squeeze() / 1e6
            Nbeam_freqs = len(beam_freqs)
            phi_grid,theta_grid=np.meshgrid(uvb.axis1_array,uvb.axis2_array) ## phi,theta grid of beam

            theta = np.sqrt( (RA_cen)**2 + (DE_cen)**2 ) # source local az za 
            phi = np.arctan2((DE_cen), (RA_cen)) + np.pi
            
            ##phase at zenith    
            theta_0 = np.sqrt( (0)**2 + (0)**2 ) # center origin at the array dec and RA
            phi_0 = np.arctan2((0), (0)) + np.pi
            index_0=lookupNearest(np.deg2rad(theta_0),np.deg2rad(phi_0),theta_grid,phi_grid)
            index_0=np.array(index_0)[:,0]
            pb_0=beam_maps[:,index_0[0],index_0[1]]
            
            pb=np.zeros((beam_freqs.shape[0],DE_cen.shape[0]),dtype=complex)
            for source in range (len(phi)):
                index=lookupNearest(np.deg2rad(theta[source]),np.deg2rad(phi[source]),theta_grid,phi_grid)
                index=np.array(index)[:,0]
                pb[:,source]=beam_maps[:,index[0],index[1]]*(np.conjugate(pb_0)/np.abs(pb_0)) 


     

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
            np.save(path_to_GLEAM+sky_model_type+field_name[fi]+'_mutual_'+beam_name[k]+'_diffuse.npy', gains)




        # type edge triad ECC
    for fi in range (len(field_name)):
        HERA_beam_mutual_06=np.load(path_to_GLEAM+sky_model_type+field_name[fi]+'_mutual_06_diffuse.npy')
        HERA_beam_mutual_22=np.load(path_to_GLEAM+sky_model_type+field_name[fi]+'_mutual_22_diffuse.npy')
        HERA_beam_mutual_26=np.load(path_to_GLEAM+sky_model_type+field_name[fi]+'_mutual_26_diffuse.npy')

        ## mutual_coupling_gain
        mutual_gains=np.load(path_to_GLEAM+'Gains_HERA_Fagnoni_beam'+field_name[fi]+'_'+sky_model_type+'.npy')
        mutual_gains[:,0,:,:,:]=HERA_beam_mutual_06
        mutual_gains[:,1,:,:,:]=HERA_beam_mutual_26
        mutual_gains[:,2,:,:,:]=HERA_beam_mutual_26
        mutual_gains[:,3,:,:,:]=HERA_beam_mutual_22
        mutual_gains[:,4,:,:,:]=HERA_beam_mutual_22
        mutual_gains[:,5,:,:,:]=HERA_beam_mutual_22
        mutual_gains[:,6,:,:,:]=HERA_beam_mutual_22
        mutual_gains[:,7,:,:,:]=HERA_beam_mutual_22

        np.save(path_to_GLEAM+'Gains_mutual_coupling_EEC_ECC_CCC'+field_name[fi]+'_'+sky_model_type+'.npy',mutual_gains)



