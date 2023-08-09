#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import scipy
#import pdb
#import matplotlib
#import matplotlib.pyplot as plt


def is2modelfit(xdata,p,range_bins_m_hr):


	'''
	3 input parameters
	p[0] = amplitude scale factor
	p[1] = x_shift to align x-axes of model and obs (surface location)
	p[2] = standard deviation of sfc height (surface roughness)


	xdata = elapsed ns [1.5625, 3.1250, 4.68...]
	'''
	##  add the tau shift to align model and obs
	taufit = (xdata + p[1]).ravel()
	taufit_hr = (range_bins_m_hr + p[1]).ravel()

    #p0 = 1
	c = 299792458; pi = 3.141529654


    ####Generate IS-2 transmit pulse shape
	mu = 0.0
	sigma = 0.116767  #sigma of Gaussian in meters  
	tau = 9.92750  #exponential relaxation time in meters
    ##Convert range bin times to x values in meters with mean of x at tau=0
	mean_xmg = mu + 1/tau
	xmgtau = taufit_hr + mean_xmg
	yy = tau/2*np.exp(tau/2*(2*mu + tau*sigma**2 - 2*xmgtau))*scipy.special.erfc( (mu + tau*sigma**2 - xmgtau) / (np.sqrt(2)*sigma) )

    # Take a running mean of the power and resample to the original resolution
	rb_scale_factor = 5
	avg_factor=np.fix(rb_scale_factor/2 - 0.5)
	yy2 = np.convolve(yy, np.repeat(1.0, avg_factor)/avg_factor, mode='same')
	yy2 = yy2[np.arange(rb_scale_factor-1, np.size(range_bins_m_hr), rb_scale_factor)]
	yy2 = yy2/np.max(yy2)
	yy2[yy2 < 0] = 0
	if np.size(yy2) > np.size(taufit):
		yy2=yy2[0:np.size(yy2)-1]
	yy=yy2

	#yy_rb = wf[1][:]
	#yy = yy/np.max(yy) #normalize to 1
    #####################################



    ##  Log-normal distribution surface pdf with mu = 0
	mu = 0
    ##std_ln = sqrt( (exp(x(3)^2) - 1) * exp(2*mu + x(3)^2));  ##standard deviation of log normal distribution
	sig_surf = p[2]  #surface roughness in meters
	mean_surf = np.exp(mu + sig_surf**2/2)
    ##Convert range bin times to x values in meters with mean of x at tau=0
	xgtau = taufit_hr + mean_surf ##
	#pdb.set_trace()
	surf_pdf = 1/(xgtau*sig_surf*np.sqrt(2*pi)) * np.exp( -1.0*(np.log(xgtau) - mu)**2 / (2*sig_surf**2))  ##log-normal function distribution
	surf_pdf[xgtau < 0] = 0   ##to get rid of negative values which blow up the distribution
	#surf_pdf = surf_pdf/np.max(surf_pdf) #normalize to 1 to make for better numeric calculations
	
	# Take a running mean of the power and resample to the original resolution
	rb_scale_factor = 5
	avg_factor=np.fix(rb_scale_factor/2 - 0.5)
	surf_pdf2 = np.convolve(surf_pdf, np.repeat(1.0, avg_factor)/avg_factor, mode='same')
	surf_pdf2 = surf_pdf2[np.arange(rb_scale_factor-1, np.size(range_bins_m_hr), rb_scale_factor)]
	surf_pdf2 = surf_pdf2/np.max(surf_pdf2)
	surf_pdf2[surf_pdf2 < 0] = 0
	if np.size(surf_pdf2) > np.size(taufit):
		surf_pdf2=surf_pdf2[0:np.size(surf_pdf2)-1]
	surf_pdf=surf_pdf2

	
	

	
    

	
	###Gaussian
    #sig_surf = p[2]
    #sig_surf = 2*sig_surf/c    #converting to time units
    #surf_pdf = 1/(np.sqrt(2*np.pi)*sig_surf) * np.exp(-0.5*(taufit*1e-9/sig_surf)**2)
    #surf_pdf = surf_pdf/max(surf_pdf) #normalize to 1 to make for better numeric calculations


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

    ##  Scale amplitude by scale factor
	power_final = p[0] * power_final

	return power_final






def is2modelfit_rough(xdata,p,surf_pdf,range_bins_m_hr):


	'''
	2 input parameters
	p[0] = amplitude scale factor
	p[1] = x_shift to align x-axes of model and obs (surface location)
	
	surf_pdf is normalized surface pdf taken from photon cloud
	'''
	##  add the tau shift to align model and obs
	taufit = (xdata + p[1]).ravel()
	taufit_hr = (range_bins_m_hr + p[1]).ravel()

    #p0 = 1
	c = 299792458; pi = 3.141529654



	   ####Generate IS-2 transmit pulse shape
	mu = 0.0
	sigma = 0.116767  #sigma of Gaussian in meters  
	tau = 9.92750  #exponential relaxation time in meters
    ##Convert range bin times to x values in meters with mean of x at tau=0
	mean_xmg = mu + 1/tau
	xmgtau = taufit_hr + mean_xmg
	yy = tau/2*np.exp(tau/2*(2*mu + tau*sigma**2 - 2*xmgtau))*scipy.special.erfc( (mu + tau*sigma**2 - xmgtau) / (np.sqrt(2)*sigma) )

    # Take a running mean of the power and resample to the original resolution
	rb_scale_factor = 5
	avg_factor=np.fix(rb_scale_factor/2 - 0.5)
	yy2 = np.convolve(yy, np.repeat(1.0, avg_factor)/avg_factor, mode='same')
	yy2 = yy2[np.arange(rb_scale_factor-1, np.size(range_bins_m_hr), rb_scale_factor)]
	yy2 = yy2/np.max(yy2)
	yy2[yy2 < 0] = 0
	if np.size(yy2) > np.size(taufit):
		yy2=yy2[0:np.size(yy2)-1]
	yy=yy2

	#yy_rb = wf[1][:]
	#yy = yy/np.max(yy) #normalize to 1
    #####################################



#    ##  Log-normal distribution surface pdf with mu = 0
#	mu = 0
#    ##std_ln = sqrt( (exp(x(3)^2) - 1) * exp(2*mu + x(3)^2));  ##standard deviation of log normal distribution
#	sig_surf = p[2]  #surface roughness in meters
#	mean_surf = np.exp(mu + sig_surf**2/2)
#    ##Convert range bin times to x values in meters with mean of x at tau=0
#	xgtau = taufit + mean_surf ##
#	surf_pdf = 1/(xgtau*sig_surf*np.sqrt(2*pi)) * np.exp( -1.0*(np.log(xgtau) - mu)**2 / (2*sig_surf**2))  ##log-normal function distribution
#	surf_pdf[xgtau < 0] = 0   ##to get rid of negative values which blow up the distribution
#	#pdb.set_trace()
#	surf_pdf = surf_pdf/np.max(surf_pdf) #normalize to 1 to make for better numeric calculations
#    
    
	
	###Gaussian
    #sig_surf = p[2]
    #sig_surf = 2*sig_surf/c    #converting to time units
    #surf_pdf = 1/(np.sqrt(2*np.pi)*sig_surf) * np.exp(-0.5*(taufit*1e-9/sig_surf)**2)
    #surf_pdf = surf_pdf/max(surf_pdf) #normalize to 1 to make for better numeric calculations


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

    ##  Scale amplitude by scale factor
	power_final = p[0] * power_final

	return power_final






def is2modelfit_gaussian(xdata,p,range_bins_m_hr):


	'''
	3 input parameters
	p[0] = amplitude scale factor
	p[1] = x_shift to align x-axes of model and obs (surface location)
	p[2] = standard deviation of sfc height (surface roughness)


	xdata = elapsed ns [1.5625, 3.1250, 4.68...]
	'''
	##  add the tau shift to align model and obs
	taufit = (xdata + p[1]).ravel()
	taufit_hr = (range_bins_m_hr + p[1]).ravel()

    #p0 = 1
	c = 299792458; pi = 3.141529654


    ####Generate IS-2 transmit pulse shape
	mu = 0.0
	sigma = 0.116767  #sigma of Gaussian in meters  
	tau = 9.92750  #exponential relaxation time in meters
    ##Convert range bin times to x values in meters with mean of x at tau=0
	mean_xmg = mu + 1/tau
	xmgtau = taufit_hr + mean_xmg
	yy = tau/2*np.exp(tau/2*(2*mu + tau*sigma**2 - 2*xmgtau))*scipy.special.erfc( (mu + tau*sigma**2 - xmgtau) / (np.sqrt(2)*sigma) )

    # Take a running mean of the power and resample to the original resolution
	rb_scale_factor = 5
	avg_factor=np.fix(rb_scale_factor/2 - 0.5)
	yy2 = np.convolve(yy, np.repeat(1.0, avg_factor)/avg_factor, mode='same')
	yy2 = yy2[np.arange(rb_scale_factor-1, np.size(range_bins_m_hr), rb_scale_factor)]
	yy2 = yy2/np.max(yy2)
	yy2[yy2 < 0] = 0
	if np.size(yy2) > np.size(taufit):
		yy2=yy2[0:np.size(yy2)-1]
	yy=yy2

	#yy_rb = wf[1][:]
	#yy = yy/np.max(yy) #normalize to 1
    #####################################



    ##  Log-normal distribution surface pdf with mu = 0
#	mu = 0
#    ##std_ln = sqrt( (exp(x(3)^2) - 1) * exp(2*mu + x(3)^2));  ##standard deviation of log normal distribution
#	sig_surf = p[2]  #surface roughness in meters
#	mean_surf = np.exp(mu + sig_surf**2/2)
#    ##Convert range bin times to x values in meters with mean of x at tau=0
#	xgtau = taufit + mean_surf ##
#	surf_pdf = 1/(xgtau*sig_surf*np.sqrt(2*pi)) * np.exp( -1.0*(np.log(xgtau) - mu)**2 / (2*sig_surf**2))  ##log-normal function distribution
#	surf_pdf[xgtau < 0] = 0   ##to get rid of negative values which blow up the distribution
#	#pdb.set_trace()
#	surf_pdf = surf_pdf/np.max(surf_pdf) #normalize to 1 to make for better numeric calculations
#    
    
	
	###Gaussian
	sig_surf = p[2]
	surf_pdf = 1/(np.sqrt(2*np.pi)*sig_surf) * np.exp(-0.5*(taufit_hr/sig_surf)**2)
	
		# Take a running mean of the power and resample to the original resolution
	rb_scale_factor = 5
	avg_factor=np.fix(rb_scale_factor/2 - 0.5)
	surf_pdf2 = np.convolve(surf_pdf, np.repeat(1.0, avg_factor)/avg_factor, mode='same')
	surf_pdf2 = surf_pdf2[np.arange(rb_scale_factor-1, np.size(range_bins_m_hr), rb_scale_factor)]
	surf_pdf2 = surf_pdf2/np.max(surf_pdf2)
	surf_pdf2[surf_pdf2 < 0] = 0
	if np.size(surf_pdf2) > np.size(taufit):
		surf_pdf2=surf_pdf2[0:np.size(surf_pdf2)-1]
	surf_pdf=surf_pdf2
	#surf_pdf = surf_pdf/max(surf_pdf) #normalize to 1 to make for better numeric calculations


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

    ##  Scale amplitude by scale factor
	power_final = p[0] * power_final

	return power_final



def is2modelfit_pond(xdata,p,range_bins_m_hr):


	'''
	3 input parameters
	p[0] = Pond bottom amplitude scale factor
	p[1] = x_shift to align x-axes of model and obs (surface location)
	p[2] = standard deviation of sfc height (surface roughness)


	xdata = elapsed ns [1.5625, 3.1250, 4.68...]
	'''
	##  add the tau shift to align model and obs
	taufit = (xdata + p[1]).ravel()
	taufit_hr = (range_bins_m_hr + p[1]).ravel()

    #p0 = 1
	c = 299792458; pi = 3.141529654
	alpha = 0.05 #absorption coefficient m^-1 for bubble free ice/water from Perovich, 2003
	vol_backscatter = 0.01  #Volume backscatter 
	


    ####Generate IS-2 transmit pulse shape
	mu = 0.0
	sigma = 0.116767  #sigma of Gaussian in meters  
	tau = 9.92750  #exponential relaxation time in meters
    ##Convert range bin times to x values in meters with mean of x at tau=0
	mean_xmg = mu + 1/tau
	xmgtau = taufit_hr + mean_xmg 
	yy = tau/2*np.exp(tau/2*(2*mu + tau*sigma**2 - 2*xmgtau))*scipy.special.erfc( (mu + tau*sigma**2 - xmgtau) / (np.sqrt(2)*sigma) )

    # Take a running mean of the power and resample to the original resolution
	rb_scale_factor = 5
	avg_factor=np.fix(rb_scale_factor/2 - 0.5)
	yy2 = np.convolve(yy, np.repeat(1.0, avg_factor)/avg_factor, mode='same')
	yy2 = yy2[np.arange(rb_scale_factor-1, np.size(range_bins_m_hr), rb_scale_factor)]
	yy2 = yy2/np.max(yy2)
	yy2[yy2 < 0] = 0
	if np.size(yy2) > np.size(taufit):
		yy2=yy2[0:np.size(yy2)-1]
	yy=yy2

	#yy_rb = wf[1][:]
	#yy = yy/np.max(yy) #normalize to 1
    #####################################



    ##  Log-normal distribution surface pdf with mu = 0
	mu = 0
    ##std_ln = sqrt( (exp(x(3)^2) - 1) * exp(2*mu + x(3)^2));  ##standard deviation of log normal distribution
	sig_surf = p[2]  #surface roughness in meters
	mean_surf = np.exp(mu + sig_surf**2/2)
    ##Convert range bin times to x values in meters with mean of x at tau=0
	xgtau = taufit_hr + mean_surf #+ p[1]##
	surf_pdf = 1/(xgtau*sig_surf*np.sqrt(2*pi)) * np.exp( -1.0*(np.log(xgtau) - mu)**2 / (2*sig_surf**2))  ##log-normal function distribution
	surf_pdf[xgtau < 0] = 0   ##to get rid of negative values which blow up the distribution
	#surf_pdf = surf_pdf/np.max(surf_pdf) #normalize to 1 to make for better numeric calculations
	
	# Take a running mean of the power and resample to the original resolution
	rb_scale_factor = 5
	avg_factor=np.fix(rb_scale_factor/2 - 0.5)
	surf_pdf2 = np.convolve(surf_pdf, np.repeat(1.0, avg_factor)/avg_factor, mode='same')
	surf_pdf2 = surf_pdf2[np.arange(rb_scale_factor-1, np.size(range_bins_m_hr), rb_scale_factor)]
	surf_pdf2 = surf_pdf2/np.max(surf_pdf2)
	surf_pdf2[surf_pdf2 < 0] = 0
	if np.size(surf_pdf2) > np.size(taufit):
		surf_pdf2=surf_pdf2[0:np.size(surf_pdf2)-1]
	surf_pdf=surf_pdf2

	
	

	
    

	
	###Gaussian
    #sig_surf = p[2]
    #sig_surf = 2*sig_surf/c    #converting to time units
    #surf_pdf = 1/(np.sqrt(2*np.pi)*sig_surf) * np.exp(-0.5*(taufit*1e-9/sig_surf)**2)
    #surf_pdf = surf_pdf/max(surf_pdf) #normalize to 1 to make for better numeric calculations


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
	
	
	
	## Convolve with volume backscatter profile
	
    # Construct surface and volume scatter profile of water and ice layers as function of time
#taufit_water = (xdata + p0[1] + water_loc_time).ravel()
	taufit_ice = (xdata + p0[1]).ravel()
    
#    # Do this section to make sure one of the points corresponds exactly to an interface point
#    snowtdiff = taufit_snow - snow_loc_time
#    Tdiffs_minloc = np.nanargmin(np.abs(taufit_snow - 1*snow_loc_time))
#    min_Tdiff_snow = np.round(Tdiffs_minloc)
#    snow_offset = snowtdiff[min_Tdiff_snow]
#    
#    icetdiff = taufit_ice
#    Tdiffs_minloc = np.nanargmin(np.abs(taufit_ice))
#    min_Tdiff_ice = np.round(Tdiffs_minloc)
#    ice_offset = snowtdiff[min_Tdiff_ice]
#    
#    pow_pdf_snow_interp_func = interpolate.interp1d(tau_pdf_snow2, pow_pdf_snow, kind='linear', fill_value='extrapolate') # Interpolation function
#    pow_pdf_snow = pow_pdf_snow_interp_func(taufit_snow - snow_offset)
#    
#    pow_pdf_ice_interp_func = interpolate.interp1d(tau_pdf_ice2, pow_pdf_ice, kind='linear', fill_value='extrapolate') # Interpolation function
#    pow_pdf_ice = pow_pdf_ice_interp_func(taufit_ice - ice_offset)
#    
#    taufit_snow = taufit_snow - snow_offset
#    taufit_ice = taufit_ice - ice_offset
#    
#    Tdiffs_minloc = np.nanargmin(np.abs(taufit_snow - 1*snow_loc_time))
#    min_Tdiff_snow = np.round(Tdiffs_minloc)
#
#    Tdiffs_minloc = np.nanargmin(np.abs(taufit_ice))
#    min_Tdiff_ice = np.round(Tdiffs_minloc)
#    
#    # Snow layer backscatter
#    snow_vol_scat = 0 * pow_pdf_snow
#    sig_v_snow = sig_v_uv_snow * k_t_snow**2 / (1 * k_e_snow)
#    alpha_v_snow = k_e_snow
#    
#    snow_vol_scat = (sig_v_snow * alpha_v_snow * np.exp(-1.0 * c_snow * k_e_snow * taufit_snow * 1e-9))
#    novolscat = [idx for idx in range(len(taufit_snow)) if (taufit_snow[idx] <  snow_loc_time)]
#    snow_vol_scat[novolscat] = 0
#    novolscat2 = [idx for idx in range(len(taufit_snow)) if (taufit_snow[idx] >=  snow_loc_time)]
#    snow_vol_scat[novolscat2] = 0
#    snow_vol_scat[min_Tdiff_snow] = snow_backscatter + 0 * sig_v_uv_snow * k_e_snow
#    
#    # Ice layer backscatter
#    
#    sig_v_ice = sig_v_uv_ice * k_t_ice**2 * k_t_snow**2 / (1*k_e_ice)
#    alpha_v_ice = k_e_ice
#    
#    ice_vol_scat = (sig_v_ice * alpha_v_ice * np.exp(-1.0 * k_e_snow * snow_depth/2) * np.exp(-1.0 * alpha_v_ice * c_ice * taufit_ice*1e-9))
#    novolscat = [idx for idx in range(len(taufit_ice)) if (taufit_ice[idx] < 0)]
#    ice_vol_scat[novolscat] = 0
#    # Backscatter at snow surface
#    ice_vol_scat[min_Tdiff_ice] = (ice_backscatter * k_t_snow**2 * np.exp(-1.0 * k_e_snow * snow_depth/2) + (0)/(1) * sig_v_ice * np.exp(-1.0 * k_e_snow * snow_depth/2) * alpha_v_ice * np.exp(-1.0 * alpha_v_ice * c_ice * taufit_ice[min_Tdiff_ice]*1e-9))
#    
#    # Now construct ice waveform and snow waveforms, then sum using superposition
#    tempbad = [idx for idx in range(len(pow_pdf_snow)) if (np.isfinite(pow_pdf_snow)[idx] == False)]
#    pow_pdf_snow[tempbad] = 0
#    tempbad = [idx for idx in range(len(pow_pdf_ice)) if (np.isfinite(pow_pdf_ice)[idx] == False)]
#    pow_pdf_ice[tempbad] = 0
#    
#    power_final_snow = np.convolve(pow_pdf_snow, snow_vol_scat)
#    tau_final_snow = np.arange(0, len(power_final_snow)) * (taufit_snow[1] - taufit_snow[0]) + taufit_snow[0]*2
#    
#    power_final_ice = np.convolve(pow_pdf_ice, ice_vol_scat)
#    tau_final_ice = np.arange(0, len(power_final_ice)) * (taufit_ice[1] - taufit_ice[0]) + taufit_ice[0]*2
#
#    atemp2 = np.nanargmin(np.abs(tau_final_snow - taufit_snow[0]))
#    power_final_snow = power_final_snow[atemp2:atemp2+len(taufit_snow)]
#    tau_final_snow = tau_final_snow[atemp2:atemp2+len(taufit_snow)]
#    
#    atemp2 = np.nanargmin(np.abs(tau_final_ice - taufit_ice[0]))
#    power_final_ice = power_final_ice[atemp2:atemp2+len(taufit_ice)]
#    tau_final_ice = tau_final_ice[atemp2:atemp2+len(taufit_ice)]
#    
#    power_final_snow2_intep_func = interpolate.interp1d(tau_final_snow, power_final_snow, kind='linear', fill_value='extrapolate') # Interpolation function
#    power_final_snow2 = power_final_snow2_intep_func(tau_final_ice)
#    
#    power_final = power_final_snow2 + power_final_ice + mnoise
#    power_final = power_final/np.max(power_final)
#    
#    power_final[np.isnan(power_final)] = 0
#
#    # Take a running mean of the power and resample to the original resolution
#    avg_factor = np.fix(rb_scale_factor/2 - 0.5)
#    #power_final2 = np.convolve(power_final, np.repeat(1.0, avg_factor*2)/avg_factor*2, mode='same')
#    power_final2 = gf.moving_average(power_final, int(avg_factor))
#    
#    power_final2 = power_final2[np.arange(rb_scale_factor-1, 128*rb_scale_factor, rb_scale_factor)]
#
#    ##  normalize, remove negative values
#    power_final2 = power_final2/np.max(power_final2)
#    power_final2[power_final2 < 0] = 0
#power_final = power_final2
	
	
	
	

    ##  Scale amplitude by scale factor
	power_final = p[0] * power_final

	return power_final
