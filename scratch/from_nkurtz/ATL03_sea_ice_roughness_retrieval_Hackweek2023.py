# -*- coding: utf-8 -*-
"""


@author: Nathan Kurtz

Runs a surface height and roughness retrieval from ATL03 photon cloud, uses ATL07 as initial guess and comparison

INPUTS:
		ATL03_file
		ATL07_file

        

"""



from datetime import datetime
#import glob
import numpy as np
import is2_model as is2m
#import netCDF4
#from netCDF4 import Dataset
import h5py
import pdb  #python debugger
import matplotlib
import matplotlib.pyplot as plt
#import os
import pandas as pd
#import pickle
import scipy
from scipy import interpolate
from scipy.signal import find_peaks, resample, resample_poly
from scipy.optimize import curve_fit
#from tqdm import tqdm
#import time



#  Set options for curve fitting
fit_opts = {'ftol': 0.01, 'max_nfev': 500, 'xtol': 0.01}






# INPUT SECTION ---------------------------------------------------------------


# ATL03

#ATL03_file = 'C:/Users/nkurtz/Desktop/ICESat2/Data/badfit/ATL03_20190315103022_11760201_005_01.h5'
ATL03_file = 'C:/Users/nkurtz/Desktop/ICESat2/Data/badfit/ATL03_20190315104554_11760203_005_01.h5'
ATL07_file = 'C:/Users/nkurtz/Desktop/ICESat2/Data/badfit/ATL07-01_20190315103022_11760201_005_01.h5'
beam = 'gt2r'


# END OF INPUT SECTION --------------------------------------------------------



out_dF = pd.DataFrame({'Lat': [],
					   'Lon': [],
					   'Height diff': [],
					   'FPB': [],
					   'N photons': [],
					   })




	
	
# Print welocome message with date and time to screen
now = datetime.now()
dt_string = now.strftime('%d/%m/%Y %H:%M:%S')
print(dt_string)
print('')
print('=====================================')





###################Read ATL07 file###################################

now = datetime.now()
dt_string = now.strftime('%d/%m/%Y %H:%M:%S')
print(dt_string)
print('')
print('Reading ATL07 file...')
print('=====================================')


hf = h5py.File(ATL07_file, 'r')	#Open the file


gtx = hf.get(beam)
ATL07_lat = gtx.get('sea_ice_segments/latitude').value
ATL07_lon = gtx.get('sea_ice_segments/longitude').value
heightx = hf.get(beam+'/sea_ice_segments/heights')
ATL07_elev = heightx.get('height_segment_height').value	#Retreived surface elevation
ATL07_ocean_tide = hf[beam]['sea_ice_segments']['geophysical']['height_segment_ocean'][:]
ATL07_ocean_load_tide = hf[beam]['sea_ice_segments']['geophysical']['height_segment_load'][:]
ATL07_pole_tide = hf[beam]['sea_ice_segments']['geophysical']['height_segment_pole'][:]

#Exmax (dual Gaussian) parameters
ATL07_h1 = gtx.get('sea_ice_segments/stats/exmax_mean_1').value
ATL07_h2 = gtx.get('sea_ice_segments/stats/exmax_mean_2').value
ATL07_w1 = gtx.get('sea_ice_segments/stats/exmax_stdev_1').value
ATL07_w2 = gtx.get('sea_ice_segments/stats/exmax_stdev_2').value
ATL07_ratio = gtx.get('sea_ice_segments/stats/exmax_mix').value


ATL07_dac = hf.get(beam+'/sea_ice_segments/geophysical/height_segment_dac').value 	#AVISO dynamic atmospheric correction
ATL07_ib = hf.get(beam+'/sea_ice_segments/geophysical/height_segment_ib').value 	#Inverse barometer correction
ATL07_earth = hf.get(beam+'/sea_ice_segments/geophysical/height_segment_earth').value 	#solid earth tide
ATL07_geoid = hf.get(beam+'/sea_ice_segments/geophysical/height_segment_geoid').value 	#geoid
ATL07_mss = hf.get(beam+'/sea_ice_segments/geophysical/height_segment_mss').value 	#mean sea surface
ATL07_delta_time = hf.get(beam+'/sea_ice_segments/delta_time').value	#Number of seconds since the GPS epoch on midnight Jan. 6, 1980 

ATL07_fpb_corr = hf.get(beam+'/sea_ice_segments/stats/fpb_corr').value #first photon bias correction
#ATL07_window_size = hf.get(beam+'/ancillary_data/fine_surface_finding/bin_f').value #Fine surface bin size


height = hf.get(beam+'/sea_ice_segments/heights')
ATL07_quality_flag = height.get('height_segment_fit_quality_flag').value #flag from 1-5 with 1 being the best
ATL07_quality = height.get('height_segment_quality').value 	#1 for good fit, 0 for bad
ATL07_elev_rms = height.get('height_segment_rms').value 	#RMS difference between modeled and observed photon height distribution
ATL07_seg_length = height.get('height_segment_length_seg').value 	#along track length of segment
ATL07_height_confidence = height.get('height_segment_confidence')	.value #Height segment confidence flag
ATL07_reflectance = height.get('height_segment_asr_calc').value 	#apparent surface reflectance
ATL07_ssh_flag = height.get('height_segment_ssh_flag').value 	#Flag for potential leads, 0=sea ice, 1 = sea surface
ATL07_seg_type = height.get('height_segment_type').value 	#0=cloud covered
ATL07_gauss_width = height.get('height_segment_w_gaussian').value 	#Width of Gaussian fit



hf.close()

ATL07_corrections = ATL07_ocean_tide + ATL07_ocean_load_tide + ATL07_mss + ATL07_ib


####End read ATL07 file










#################Read ATL03 file###################################

now = datetime.now()
dt_string = now.strftime('%d/%m/%Y %H:%M:%S')
print(dt_string)
print('')
print('Reading ATL03 file...')
print('=====================================')



hf = h5py.File(ATL03_file, 'r')	#Open the file

# Read ATL03 parameters of interest
ph_delta_time = hf[beam + '/heights/delta_time'][:]
ph_track_dist = hf[beam + '/heights/dist_ph_along'][:]
ph_lat = hf[beam + '/heights/lat_ph'][:]
ph_lon = hf[beam + '/heights/lon_ph'][:]
ph_height = hf[beam + '/heights/h_ph'][:]
geoid = hf[beam + '/geophys_corr/geoid'][:]
ph_confidence = hf[beam + '/heights/signal_conf_ph'][:, 0] #0: land, 1: ocean, 2: sea ice, 3: land ice, 4: inland water
#ph_pulse_id = ATL03file[beam + '/heights/ph_id_pulse'][:]


hf.close()


#########End read ATL03 file







goodfits =  np.where( (ATL07_quality_flag <= 4) )  #Using only where fit quality flag was good
npts = np.size(goodfits)

for ngood in range(10,npts-1):
	#	#Get min and max delta times, could also use ph_track_dist
	min_dtime = ATL07_delta_time[ngood] -  1*ATL07_seg_length[ngood]/2.0 / 6900.0     #Can change this to accumulate more photons, assumes orbital speed of 6.9 km/s
	max_dtime = ATL07_delta_time[ngood] +  1*ATL07_seg_length[ngood]/2.0 / 6900.0
	
	print("ATL07 quality flag, delta_times, elev, width: ",ATL07_quality_flag[ngood],min_dtime,max_dtime,ATL07_elev[ngood],ATL07_gauss_width[ngood])
	


	photons_loc = np.where( (ph_delta_time > min_dtime) & (ph_delta_time < max_dtime) & ( (ph_height-ATL07_corrections[ngood]) > (ATL07_elev[ngood]-3.0) ) & ( (ph_height-ATL07_corrections[ngood]) < (ATL07_elev[ngood]+5.0)) )

#	plt.plot(ph_delta_time[photons_loc],ph_height[photons_loc]-ATL07_corrections[ngood],'+k')	
#	plt.show()
	
	nrb = 70
	rb_res = 0.1
	track_point = ATL07_elev[ngood]
	range_bins_m = np.arange(1, nrb) * rb_res - (nrb-1)/2*rb_res - ATL07_elev[ngood] #(nrb-1)/3*rb_res  # Set range bins with 3 cm resolution
	wf = np.histogram(ph_height[photons_loc]-ATL07_corrections[ngood],bins=range_bins_m)
	WF_norm_this = wf[0][:]/np.max(wf[0][:])
	wf_rb = wf[1][:]
#	p1=plt.plot(wf_rb[0:np.size(wf[0][:])],wf[0][:])
#	p1 = plt.plot([track_point,0],[track_point,np.max(wf[0][:])],'k') #Plot ATL07 point which is the mean height location
#	plt.show()
#	
#	print("N photons: ",np.size(photons_loc))
	
	
	
	# Calculate lognormal fit
	#range_bins_m = np.arange(1, nrb) * rb_res - (nrb-1)/3*rb_res  # Set range bins with 3 cm resolution
	range_bins_m_hr = np.arange(1, nrb*5) * rb_res/5 - (5*nrb-1)/2*rb_res/5 - ATL07_elev[ngood]  # Set range bins with 5x rb_res cm resolution for use when generating pulse shape and distributions
	range_bins_unit = np.arange(1, nrb)
	
	tracker = lambda xdata, *p: is2m.is2modelfit(xdata, p,range_bins_m_hr)

	rb_use = range_bins_m[0:np.size(WF_norm_this)]
	x0 = np.asarray([1.0, track_point*0, ATL07_gauss_width[ngood]])  # y scale factor, x shift (in meters), roughness
	lb = x0 - [0.05, 0.5, 0.3]
	ub = x0 + [0.05, 0.5, 0.3]
	if lb[2] < 0:
		lb[2] = 0.005
	GSFC_optimal, GSFC_cov = curve_fit(tracker, rb_use, WF_norm_this, x0, bounds=(lb, ub), method='trf',  **fit_opts)
	#GSFC_optimal[2]=0.9


	GSFC_fit = is2m.is2modelfit(rb_use, GSFC_optimal,range_bins_m_hr)
	# Calculate sum of squared residuals
	GSFC_resnorm = np.sum((WF_norm_this - GSFC_fit)**2)
	
	

	#Get Dual-Gaussian for plotting
	m1 = ATL07_h1[ngood]
	m2 = ATL07_h2[ngood]
	w1 = ATL07_w1[ngood]
	w2 = ATL07_w2[ngood]
	weight1 = ATL07_ratio[ngood]
	weight2 = (1.0-weight1)
	xin = wf_rb[0:np.size(wf[0][:])]
	dual_Gaussian_fit = weight1*1/(np.sqrt(2*np.pi)*w1) * np.exp(-0.5*(xin-m1)**2/(w1)**2) + weight2*1/(np.sqrt(2*np.pi)*w2) * np.exp(-0.5*(xin-m2)**2/(w2)**2)
	dual_Gaussian_fit = dual_Gaussian_fit/np.max(dual_Gaussian_fit)
	surf_pdf = dual_Gaussian_fit
	
	
	    ####Generate IS-2 transmit pulse shape and convolve with Gaussian
	mu = 0.0
	sigma = 0.116767  #sigma of Gaussian in meters  
	tau = 9.92750  #exponential relaxation time in meters
	taufit_hr = xin
	taufit = xin
    ##Convert range bin times to x values in meters with mean of x at tau=0
	mean_xmg = mu + 1/tau
	xmgtau = taufit_hr + mean_xmg
	yy = tau/2*np.exp(tau/2*(2*mu + tau*sigma**2 - 2*xmgtau))*scipy.special.erfc( (mu + tau*sigma**2 - xmgtau) / (np.sqrt(2)*sigma) )

#    # Take a running mean of the power and resample to the original resolution
#	rb_scale_factor = 5
#	avg_factor=np.fix(rb_scale_factor/2 - 0.5)
#	yy2 = np.convolve(yy, np.repeat(1.0, avg_factor)/avg_factor, mode='same')
#	yy2 = yy2[np.arange(rb_scale_factor-1, np.size(range_bins_m_hr), rb_scale_factor)]
#	yy2 = yy2/np.max(yy2)
#	yy2[yy2 < 0] = 0
#	if np.size(yy2) > np.size(taufit):
#		yy2=yy2[0:np.size(yy2)-1]
#	yy=yy2

    ##  Convolve surfPDF and YY to get model
	power = np.convolve(surf_pdf,yy)
    ## Change axis size after convolution
	Taxis=np.arange(0,len(power))*(taufit[1]-taufit[0]) + taufit[0]*2

    ##  Get convolved power matrix to same size as input
	startloc = np.argmin(np.abs(Taxis - taufit[0])) # finds location of taufit start on Taxis
	power_final = power[startloc:startloc+len(taufit)]

    ##  normalize, remove negative values
	power_final = power_final/np.max(power_final)
	power_final[power_final < 0] = 0
	dual_Gaussian_fit = power_final / np.max(power_final)
    #####################################
	
	

	
	p2=plt.plot(wf_rb[0:np.size(wf[0][:])],WF_norm_this)
	p1w,=plt.plot(wf_rb[0:np.size(wf[0][:])],dual_Gaussian_fit,label='Dual Gaussian fit')
	p2 = plt.plot([track_point,0],[track_point,1],'k')
	p1a,=plt.plot(rb_use+GSFC_optimal[1]-0*range_bins_m_hr[0],GSFC_fit/np.max(GSFC_fit),label='Log-normal fit')
	#p2=plt.plot(rb_use+GSFC_optimal[1],WF_norm_this,'k')
	p2 = plt.plot([0,0],[0,1],'k') #Plot 0 point which is the mean height location
	#p1 = plt.plot([1.0*GSFC_optimal[1],1.0*GSFC_optimal[1]],[0,1])  #Plot Ron's ATL07 tracking point
	plt.legend(handles=[p1w, p1a])
	plt.axis([-2, 2, 0, 1]) 
	plt.xlabel('Relative Height [m]')
	plt.ylabel('PDF')
	plt.show()
	
	print("N photons,ATL07 elev, retrack elev, ATL07 rough, retrack roughness: ",np.size(photons_loc),track_point,GSFC_optimal[1],ATL07_gauss_width[ngood],GSFC_optimal[2])
	
	pdb.set_trace()

















