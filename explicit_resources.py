#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 19:58:18 2024

@author: anne-cathrinedott
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:29:16 2022

@author: anne-cathrinedott
"""
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
import spiceypy as spice
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d
from datetime import datetime
import os
import math
#%% Spice Kernels

kernelsPath = os.path.join(os.path.dirname(os.getcwd()),"kernels")

#Path for kernels
leapSecondsKernelPath = os.path.join(kernelsPath,"naif0012.tls")
IokernelPath=os.path.join(kernelsPath,"jup365.bsp")
PCKKernelPath=os.path.join(kernelsPath,"pck00010.tpc")
JUPKernelPath=os.path.join(kernelsPath,"de440s.bsp")

#Load Kernels
spice.furnsh(leapSecondsKernelPath)
spice.furnsh(IokernelPath)
spice.furnsh(PCKKernelPath)
spice.furnsh(JUPKernelPath)
 #%%One single time step u=current temperature distribution

def euler_step(T_curr,dz,dt,alpha,production,loss,i_bc,t_0,Z_0,i_lon,therm_rad_Jup,sun_ref_Jup):
    '''
    

    Parameters
    ----------
    T_curr : 1D vector
        Current Temperature Distribution.
    dz : Integer
        Stepsize in z direction.
    dt : Integer
        Stepsize in t direction.
    alpha : float
        Thermal diffusivity value.
    production : float
        Production rate.
    loss : float
        loss rate.
    i_bc : float
        Inner Boundary Condition.
    t_0 : float
        to normalize t dimension.
    Z_0 : float
        To normalize z dimension.

    Returns
    -------
    T_ret : 1D array
        New temperature distribution with depth.

    '''
    L=len(T_curr) #note: last element of u=u[L-1]
    T_ret=np.zeros(L)
    
    #PDE discretized
    for i in range(1,L-1): #note: exclusive len(u)-1
        T_ret[i]=((alpha*t_0/Z_0**2)*dt/(dz**2))*(T_curr[i+1]-2*T_curr[i]+T_curr[i-1])+T_curr[i]

    if i_lon<90 or i_lon>270:
        T_ret[0]=sun_ref_Jup*dz+therm_rad_Jup*dz+production*dz-loss*dz*T_curr[0]**4+T_ret[1] #outer bc with add. prod terms at sub jov side
    else:
       T_ret[0]=production*dz-loss*dz*T_curr[0]**4+T_ret[1] #outer bc
    
    
    
    T_ret[L-1]=i_bc*dz+T_ret[L-2] #inner bc
    
    
    return T_ret


#%%Iteration over all time steps, columnwise in array
def euler_time_evolution(temp_0,z_vals,t_vals,times,alpha,loss,i_bc,i_lat,i_lon,Z_0,T_0,t_0,rho,cp,A,const,const_sunlight): 
    '''
    

    Parameters
    ----------
    temp_0 : 1D float array
        Temperature distribution with depth at initial time step.
    z_vals : 1D float array
        Depths where (sub)surface temperature is calculated.
    t_vals : 1D float array
        Time points where Temperature is calculated.
    times : et_times (=t_vals for spice commands)
        (=t_vals for spice commands).
    alpha : float
        thermal diffusivity.
    loss : float
        loss rate.
    i_bc : float
        inner boundary condition.
    i_lat : float
        current latitude in Io body fixed reference frame.
    i_lon : float
        current longitude in Io body fixed reference frame.
    Z_0 : float
        To normalize z dimension.
    T_0 : float
        to normalize T dimension.
    t_0 : float
        to normalize t dimension.
    rho : float
        subsurface density.
    cp : float
        subsurface specific heat capacity.
    A : float
        Geometric albedo.

    Returns
    -------
    time_evolution : array of float
        complete time evolution with dimension (len(z)xlen(t)) in K/T_0.
    ang_zen : array of float
        Zenith angle as a function of time in deg.
    ang_gamma : array of float
        angle gamma as a function of time in deg.
    ang_crit : array of float
        critical angle as a function of time in deg. That angle determines the times that Io is in eclipse by Jupiter
    Fs : array of float
        Solar radiation perpendicular to the surface as a Function of time in W/m^2.
    prod : array of float
        production rate as a function of time.

    '''
    
    #% Spice Kernels
    kernelsPath = os.path.join(os.path.dirname(os.getcwd()),"kernels")

    #Path for kernels
    leapSecondsKernelPath = os.path.join(kernelsPath,"naif0012.tls")
    IokernelPath=os.path.join(kernelsPath,"jup365.bsp")
    PCKKernelPath=os.path.join(kernelsPath,"pck00010.tpc")
    JUPKernelPath=os.path.join(kernelsPath,"de440s.bsp")

    #Load Kernels
    spice.furnsh(leapSecondsKernelPath)
    spice.furnsh(IokernelPath)
    spice.furnsh(PCKKernelPath)
    spice.furnsh(JUPKernelPath)
    time_evolution=np.zeros((len(z_vals),len(t_vals)))
    
    #Initial condition = first column
    time_evolution[:,0]=temp_0
    
    #Number of time/space steps
    Nt=len(t_vals)
    Nz=len(z_vals)
    
    #Step lengths (equidistant)
    dz=(z_vals[len(z_vals)-1]-z_vals[0])/(Nz-1)
    dt=(t_vals[len(t_vals)-1]-t_vals[0])/(Nt-1)
    
    const_dist=5.2 #AU
    ang_zen=[]
    sun_dist=[]
    ang_gamma=[]
    ang_crit=[]
    Fs=[]
    prod=[]
    ang_beta=[]
    for i in range(1,Nt):
        #Insolation and zenithangle (individually for each point (theta, phi) on surface)
        #angles
        P=  np.array(spice.sphrec(1821.5, np.deg2rad(90-i_lat), np.deg2rad(360-i_lon)))       #position vector on Io's surface (R_Io from Jovian Satellize Fact sheet), colat and lat (according to IAU) in radians
        J, lighttimesJ= spice.spkpos('JUPITER', times[i], 'IAU_IO', 'NONE', 'IO')         # vector from Io's to Jupiters Barycenter
        S, lighttimesS= spice.spkpos('SUN', times[i], 'IAU_IO', 'NONE', 'IO')        #vector from Io's to sun's barycenter 
        S_J, lighttimesS_J= spice.spkpos('SUN', times[i], 'IAU_IO', 'NONE', 'JUPITER')        #vector from Jupiter's to Sun's barycenter  
        
        J=np.array(J.T)
        S=np.array(S.T)
        S_J=np.array(S_J.T)
        
        #needed values for heat equation 
        ang_zenith=np.rad2deg(np.arccos(np.dot(S,P)/(norm(S)*norm(P)))) #xi
        xi_2=np.rad2deg(np.arccos(np.dot(J,P)/(norm(J)*norm(P))))
        gamma = np.rad2deg(np.arccos((np.dot(S,J))/(norm(S)*norm(J)))) #angle J,S
        gamma_crit= np.rad2deg(np.arcsin(71492/norm(J)))
        beta=np.rad2deg(np.arccos((np.dot(S_J,J))/(norm(S_J)*norm(J))))

        
        ang_zen.append(ang_zenith)
        ang_gamma.append(gamma)
        ang_crit.append(gamma_crit)
        sun_dist.append(norm(S))
        ang_beta.append(beta)
        
        if  (gamma<=gamma_crit) or (ang_zenith>90 and ang_zenith<180): #eclipse & nightside condition
            F=0
        else:
            #F=1361/((const_dist)**2) #jovian orbit eccentricity=0
            F=1361/((norm(S)/1.496e+8)**2)
        
        temp_curr=time_evolution[:,i-1]
        production=(Z_0/((rho*cp*alpha)*T_0))*F*(1-A)*abs(np.cos(np.deg2rad(ang_zenith)))  
        therm_rad_Jup=(Z_0/T_0)*const*(71492/norm(J))**2*abs(np.cos(np.deg2rad(xi_2)))
        sun_ref_Jup=(Z_0/T_0)*const_sunlight*(71492/norm(J))**2*abs(np.cos(np.deg2rad(xi_2)))*(beta/180)

        #omega=2*np.pi/(42.5*60*60)
        #lon_temp=omega*t_vals[i]*t_0
        #production=(Z_0/((rho*cp*alpha)*T_0))*F*(1-A)*abs(np.cos(np.deg2rad(i_lat)))  #inclination wrt orbital plane=0
        time_evolution[:,i]= euler_step(temp_curr, dz, dt, alpha, production,loss, i_bc,t_0,Z_0,i_lon,therm_rad_Jup,sun_ref_Jup) #time step in column
        
        Fs.append(F)
        prod.append(production*(T_0/Z_0)*rho*cp*alpha)
    
    return time_evolution,np.array(ang_zen),np.array(ang_gamma),np.array(ang_crit),np.array(Fs),np.array(prod),np.array(sun_dist),np.array(ang_beta)




#clean up kernels
spice.kclear()

#%%Iteration for calculating map of surface temps, columnwise in array
def euler_time_evolution_full(temp_0,z_vals,t_vals,times,alpha,loss,i_bc,i_lat,i_lon,Z_0,T_0,t_0,rho,cp,A,eq_count,step_day,temp1,utc_times,const,const_sunlight): 
    '''
    

    Parameters
    ----------
    temp_0 : 1D float array
        Temperature distribution with depth at initial time step.
    z_vals : 1D float array
        Depths where (sub)surface temperature is calculated.
    t_vals : 1D float array
        Time points where Temperature is calculated.
    times : et_times (=t_vals for spice commands)
        (=t_vals for spice commands).
    alpha : float
        thermal diffusivity.
    loss : float
        loss rate.
    i_bc : float
        inner boundary condition.
    i_lat : float
        current latitude in Io body fixed reference frame.
    i_lon : float
        current longitude in Io body fixed reference frame.
    Z_0 : float
        To normalize z dimension.
    T_0 : float
        to normalize T dimension.
    t_0 : float
        to normalize t dimension.
    rho : float
        subsurface density.
    cp : float
        subsurface specific heat capacity.
    A : float
        Geometric albedo.

    Returns
    -------
    time_evolution : array of float
        complete time evolution with dimension (len(z)xlen(t)) in K/T_0.
    ang_zen : array of float
        Zenith angle as a function of time in deg.
    ang_gamma : array of float
        angle gamma as a function of time in deg.
    ang_crit : array of float
        critical angle as a function of time in deg. That angle determines the times that Io is in eclipse by Jupiter
    Fs : array of float
        Solar radiation perpendicular to the surface as a Function of time in W/m^2.
    prod : array of float
        production rate as a function of time.

    '''
    print('Lat=',i_lat)
    print('Lon=',i_lon)
    #% Spice Kernels
    kernelsPath = os.path.join(os.path.dirname(os.getcwd()),"kernels")

    #Path for kernels
    leapSecondsKernelPath = os.path.join(kernelsPath,"naif0012.tls")
    IokernelPath=os.path.join(kernelsPath,"jup365.bsp")
    PCKKernelPath=os.path.join(kernelsPath,"pck00010.tpc")
    JUPKernelPath=os.path.join(kernelsPath,"de440s.bsp")

    #Load Kernels
    spice.furnsh(leapSecondsKernelPath)
    spice.furnsh(IokernelPath)
    spice.furnsh(PCKKernelPath)
    spice.furnsh(JUPKernelPath)
    time_evolution=np.zeros((len(z_vals),len(t_vals)))
    
    #Initial condition = first column
    time_evolution[:,0]=temp_0
    
    #Number of time/space steps
    Nt=len(t_vals)
    Nz=len(z_vals)
    
    #Step lengths (equidistant)
    dz=(z_vals[len(z_vals)-1]-z_vals[0])/(Nz-1)
    dt=(t_vals[len(t_vals)-1]-t_vals[0])/(Nt-1)
    
  
    
    for i in range(1,Nt):
        #Insolation and zenithangle (individually for each point (theta, phi) on surface)
        #angles
        P=  np.array(spice.sphrec(1821.5, np.deg2rad(90-i_lat), np.deg2rad(360-i_lon)))       #position vector on Io's surface (R_Io from Jovian Satellize Fact sheet), colat and lat (according to IAU) in radians
        J, lighttimesJ= spice.spkpos('JUPITER', times[i], 'IAU_IO', 'NONE', 'IO')         # vector from Io's to Jupiters Barycenter
        S, lighttimesS= spice.spkpos('SUN', times[i], 'IAU_IO', 'NONE', 'IO')        #vector from Io's to sun's barycenter 
        S_J, lighttimesS_J= spice.spkpos('SUN', times[i], 'IAU_IO', 'NONE', 'JUPITER')        #vector from Jupiter's to Sun's barycenter 
   
        J=np.array(J.T)
        S=np.array(S.T)
        S_J=np.array(S_J.T)
        
        #needed values for heat equation 
        ang_zenith=np.rad2deg(np.arccos(np.dot(S,P)/(norm(S)*norm(P))))       # xi
        xi_2=np.rad2deg(np.arccos(np.dot(J,P)/(norm(J)*norm(P))))
        gamma =np.rad2deg(np.arccos((np.dot(S,J))/(norm(S)*norm(J)))) #angle J,S
        gamma_crit= np.rad2deg(np.arcsin(71492/norm(J)))
        beta=np.rad2deg(np.arccos((np.dot(S_J,J))/(norm(S_J)*norm(J))))
        
        
        if  (gamma<=gamma_crit) or (ang_zenith>90 and ang_zenith<180): #eclipse condition
            F=0
        else:
            F=1361/((norm(S)/1.496e+8)**2)
            
        temp_curr=time_evolution[:,i-1]
        production=(Z_0/((rho*cp*alpha)*T_0))*F*(1-A)*abs(np.cos(np.deg2rad(ang_zenith)))  
        therm_rad_Jup=(Z_0/T_0)*const*(71492/norm(J))**2*abs(np.cos(np.deg2rad(xi_2)))
        sun_ref_Jup=(Z_0/T_0)*const_sunlight*(71492/norm(J))**2*abs(np.cos(np.deg2rad(xi_2)))*(beta/180)
        
        time_evolution[:,i]= euler_step(temp_curr, dz, dt, alpha, production,loss, i_bc,t_0,Z_0,i_lon,therm_rad_Jup,sun_ref_Jup) #time step in column
        

    
    return time_evolution[0,int(eq_count*step_day+temp1):int(eq_count*step_day+temp1+step_day)],utc_times[int(eq_count*step_day+temp1):int(eq_count*step_day+temp1+step_day)] #One day in thermal eq- only surface temp




#clean up kernels
spice.kclear()

#%% Ang_ series for all Longitudes
def Ang_series_map(z_vals, t_vals,i_lon, times, eq_count, step_day,LT_start,utc_times): 
    '''
    

    Parameters
    ----------
    z_vals : TYPE
        DESCRIPTION.
    t_vals : TYPE
        DESCRIPTION.
    i_lon : TYPE
        DESCRIPTION.
    times : TYPE
        DESCRIPTION.
    eq_count : TYPE
        DESCRIPTION.
    step_day : TYPE
        DESCRIPTION.
    LT_start : TYPE
        DESCRIPTION.
    utc_times : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    print('Lon=',i_lon)
    #% Spice Kernels
    kernelsPath = os.path.join(os.path.dirname(os.getcwd()),"kernels")

    #Path for kernels
    leapSecondsKernelPath = os.path.join(kernelsPath,"naif0012.tls")
    IokernelPath=os.path.join(kernelsPath,"jup365.bsp")
    PCKKernelPath=os.path.join(kernelsPath,"pck00010.tpc")
    JUPKernelPath=os.path.join(kernelsPath,"de440s.bsp")

    #Load Kernels
    spice.furnsh(leapSecondsKernelPath)
    spice.furnsh(IokernelPath)
    spice.furnsh(PCKKernelPath)
    spice.furnsh(JUPKernelPath)
    
    #Number of time/space steps
    Nt=len(t_vals)
    
    ang_zen=[]
    
    for i in range(1,Nt):
        #Insolation and zenithangle (individually for each point (theta, phi) on surface)
        #angles
        P=  np.array(spice.sphrec(1821.5, np.deg2rad(90-0), np.deg2rad(360-i_lon)))       #position vector on Io's surface (R_Io from Jovian Satellize Fact sheet), colat and lat (according to IAU) in radians
        J, lighttimesJ= spice.spkpos('JUPITER', times[i], 'IAU_IO', 'NONE', 'IO')         # vector from Io's to Jupiters Barycenter
        S, lighttimesS= spice.spkpos('SUN', times[i], 'IAU_IO', 'NONE', 'IO')        #vector from Io's to sun's barycenter 
   
        J=np.array(J.T)
        S=np.array(S.T)
        
        #needed values for heat equation 
        ang_zenith=np.rad2deg(np.arccos(np.dot(S,P)/(norm(S)*norm(P))))       # xi
        #gamma =np.rad2deg(np.arccos((np.dot(S,J))/(norm(S)*norm(J)))) #angle J,S
        #gamma_crit= np.rad2deg(np.arcsin(71492/norm(J)))
        
        ang_zen.append(ang_zenith)
    
    
    Ang_series=one_day_timeseries(eq_count, step_day,LT_start, np.array(ang_zen),utc_times)[0]

    return np.array(Ang_series)


#clean up kernels
spice.kclear()
#%%When is surface Temperature in thermal equilibrium 

def equilibrium_counter(array_day,step_day,time_series_inpt,T_0):
    '''
    

    Parameters
    ----------
    array_day : array of float
        Elements are amount of time steps for 1,2,... Io days.
    step_day : float
        Number of time steps per Io! Day.
    time_series_inpt : array of float
        complete time series of surface temperature.
    T_0 : float
        to normalize T dimension.

    Returns
    -------
    eq_count : int
        Number of Io days after surface is in thermal equilibrium (=eq_count*(42.5/24) Earth days).

    '''
    eq_count=0 #equilibrium counter 
    for i in range(0,len(array_day)-1):
        temp_1=i*step_day
        temp_2=step_day*(i+1)
        diff=abs(time_series_inpt[int(temp_1)]-time_series_inpt[int(temp_2)])
        if abs(diff*T_0)<=0.005:
            break
        else:
            eq_count+=1
            
    return eq_count

#%% Find minimum for plotting start, returns new time series (one day) startting at LT 00:00 Uhr 

def find_LT(ang_zenith,eq_count,step_day,time_series):
    '''
    Functions finds Local Time index. 

    Parameters
    ----------
    ang_zenith : array of float
        zenith angles as function of time.
    eq_count : int
         Number of Io days after surface is in thermal equilibrium (=eq_count*(42.5/24) Earth days)..
    step_day : int 
        amount of time steps per Io day.
    time_series : array of float
        time series for which local time should be determined.

    Returns
    -------
    new_time_series : array of float
        new time series starting at local time.

    '''
    
    temp1=np.where(ang_zenith[int(eq_count*step_day):int((eq_count*step_day)+(step_day*2)+1)]==np.array(ang_zenith[int(eq_count*step_day):int((eq_count*step_day)+(step_day*2)+1)]).max())
    temp1=temp1[0][0]
    
    return temp1

#%%
    
def one_day_timeseries(eq_count,step_day,temp1,time_series,times):
    '''
    

    Parameters
    ----------
    time_series : TYPE
        DESCRIPTION.
    eq_count : TYPE
        DESCRIPTION.
    step_day : TYPE
        DESCRIPTION.
    temp1 : TYPE
        DESCRIPTION.

    Returns
    -------
    new_time_series : TYPE
        DESCRIPTION.

    '''
    
    
    if (len(time_series.shape) > 1):
        new_time_series=time_series[:,int(eq_count*step_day+temp1):int(eq_count*step_day+temp1+step_day)]
    else:
        new_time_series=time_series[int(eq_count*step_day+temp1):int(eq_count*step_day+temp1+step_day)]
        
    one_day_times=times[int(eq_count*step_day+temp1):int(eq_count*step_day+temp1+step_day)]
    
    return new_time_series,one_day_times 


#%%

def ts_jovian_year(time_series,eq_count,step_day,ang_zenith,t):
    '''
    One Jovian Year time series (Local Time)

    Parameters
    ----------
    time_series : TYPE
        DESCRIPTION.
    eq_count : TYPE
        DESCRIPTION.
    step_day : TYPE
        DESCRIPTION.
    ang_zenith : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    time_series_year : TYPE
        DESCRIPTION.
    total_days : TYPE
        DESCRIPTION.

    '''
    if (len(time_series.shape) > 1): #determine time series in thermal equilibrium
        time_series_year=time_series[0,int(eq_count*step_day):] #eq=500
    else:
        time_series_year=time_series[int(eq_count*step_day):] #eq=500
        
    t_year=t[int(eq_count*step_day):] #and the corresponding utc times
    LT_start_year=find_LT(ang_zenith,eq_count,step_day,time_series_year) #To start at Local Time 00 o'clock
    total_days=math.floor(len(time_series_year[LT_start_year:])/step_day) #Simulate full days from then til 

    time_series_year=time_series_year[LT_start_year:int(total_days*step_day)+LT_start_year]
    t_year=t_year[LT_start_year:int(total_days*step_day)+LT_start_year]
    
    return time_series_year,total_days,t_year

#%% Defines Spring and Autumm geometrically (Spring/Autumm Equinoxes)

def equinoxes(times,utc_times):
    '''

    Parameters
    ----------
    times : TYPE
        DESCRIPTION.
    utc_times : TYPE
        DESCRIPTION.

    Returns
    -------
    season_times : TYPE
        DESCRIPTION.

    '''
    
    #% Spice Kernels
    kernelsPath = os.path.join(os.path.dirname(os.getcwd()),"kernels")

    #Path for kernels
    leapSecondsKernelPath = os.path.join(kernelsPath,"naif0012.tls")
    IokernelPath=os.path.join(kernelsPath,"jup365.bsp")
    PCKKernelPath=os.path.join(kernelsPath,"pck00010.tpc")
    JUPKernelPath=os.path.join(kernelsPath,"de440s.bsp")

    #Load Kernels
    spice.furnsh(leapSecondsKernelPath)
    spice.furnsh(IokernelPath)
    spice.furnsh(PCKKernelPath)
    spice.furnsh(JUPKernelPath)
    
    season_times=[]
    for i in range(0,len(times)):
        SJ, lighttimesSJ= spice.spkpos('SUN', times[i], 'IAU_JUPITER', 'NONE', 'JUPITER')        #vector from Jupiter's to sun's barycenter 
        
        SJ=np.array(SJ.T)/norm(SJ)
        
        #Test if z perp to SJ
        perp=np.dot([0,0,1],SJ)
        
            
        if  abs(perp)<=1e-4: #Spring/Autumn condition
            season_times.append(utc_times[i])
    return np.array(season_times)

#clean up kernels
spice.kclear()

#%%#%% Defines Summer and Winter geometrically  (Summer/Winter Solstice)
def solstices(times,utc_times):
    '''
    

    Parameters
    ----------
    times : TYPE
        DESCRIPTION.
    utc_times : TYPE
        DESCRIPTION.

    Returns
    -------
    summer_solstice_idx : TYPE
        DESCRIPTION.
    winter_solstice_idx : TYPE
        DESCRIPTION.

    '''
    #% Spice Kernels
    kernelsPath = os.path.join(os.path.dirname(os.getcwd()),"kernels")

    #Path for kernels
    leapSecondsKernelPath = os.path.join(kernelsPath,"naif0012.tls")
    IokernelPath=os.path.join(kernelsPath,"jup365.bsp")
    PCKKernelPath=os.path.join(kernelsPath,"pck00010.tpc")
    JUPKernelPath=os.path.join(kernelsPath,"de440s.bsp")

    #Load Kernels
    spice.furnsh(leapSecondsKernelPath)
    spice.furnsh(IokernelPath)
    spice.furnsh(PCKKernelPath)
    spice.furnsh(JUPKernelPath)
    
    beta_array=[] #angle between SJ and rot axis [0,0,1]
    for i in range(0,len(times)):
        SJ, lighttimesSJ= spice.spkpos('SUN', times[i], 'IAU_JUPITER', 'NONE', 'JUPITER')        #vector from Jupiter's to sun's barycenter 
        
        SJ=np.array(SJ.T)/norm(SJ)
        
        #Test if z perp to SJ
        rot_axis=[0,0,1]
        beta=np.rad2deg(np.arccos((np.dot(SJ,rot_axis))/(norm(SJ)*norm(rot_axis))))
        beta_array.append(beta)
            
    summer_solstice_idx=np.where(beta_array==np.min(beta_array))[0][0]   
    winter_solstice_idx=np.where(beta_array==np.max(beta_array))[0][0] 
    summer_solstice_idx2=np.where(beta_array==np.min(beta_array[winter_solstice_idx:]))[0][0]
    
    return summer_solstice_idx,winter_solstice_idx,summer_solstice_idx2

#clean up kernels
spice.kclear()



