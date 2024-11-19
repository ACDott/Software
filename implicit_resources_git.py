#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 09:43:07 2022

@author: anne-cathrinedott
"""
import numpy as np
from numpy.linalg import norm
from scipy.linalg import solve
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
import spiceypy as spice
from datetime import datetime
from datetime import timedelta, date
import os
import math
#%% Define Matrix

def define_matrix(L,alpha,t_0,R_0,dt,dr):
    '''
    

    Parameters
    ----------
    L : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    t_0 : TYPE
        DESCRIPTION.
    R_0 : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    dr : TYPE
        DESCRIPTION.

    Returns
    -------
    M : TYPE
        DESCRIPTION.

    '''

    #Create Matrix  
    D=(alpha*t_0/R_0**2)*dt/(dr**2)
    main=np.ones(L)
    main[1:L-1]=1+2*D
    
    sub_top=np.zeros(L)
    sub_top[2:L]=-D
    #sub_top[1]=-1 
    #CP/Ana
    #sub_top[1]=0

    
    sub_bot=np.zeros(L)
    sub_bot[0:L-2]=-D
    #sub_bot[L-2]=-1 
    #CP/Ana
    #sub_bot[L-2]=0
    
    entries = np.array([main, sub_bot, sub_top]) 
    diags = np.array([0, -1, 1])
    M = spdiags(entries, diags, L, L).toarray()
    
    return M 

#%% Create full time series

def time_evolution(temp_0,z_vals,t_vals,times,M,dr,dt,loss,inner_bc,i_lat,i_lon,R_0,T_0,rho,cp,alpha,Albedo,const,const_sunlight):
    '''
    

    Parameters
    ----------
    temp_0 : TYPE
        DESCRIPTION.
    z_vals : TYPE
        DESCRIPTION.
    t_vals : TYPE
        DESCRIPTION.
    times : TYPE
        DESCRIPTION.
    M : TYPE
        DESCRIPTION.
    dr : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.
    loss : TYPE
        DESCRIPTION.
    inner_bc : TYPE
        DESCRIPTION.
    i_lat : TYPE
        DESCRIPTION.
    i_lon : TYPE
        DESCRIPTION.
    R_0 : TYPE
        DESCRIPTION.
    T_0 : TYPE
        DESCRIPTION.
    rho : TYPE
        DESCRIPTION.
    cp : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    Albedo : TYPE
        DESCRIPTION.

    Returns
    -------
    time_ev : TYPE
        DESCRIPTION.

    '''
    L=len(temp_0)
    #Initial condition = first column
    Nt=len(t_vals)
    #time_ev[:,0]=u_0
    
    
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
    time_ev=np.zeros((len(z_vals),len(t_vals)))
    
    #Initial condition = first column
    time_ev[:,0]=temp_0
    
    
    const_dist=5.2 #AU
    ang_zen=[]
    sun_dist=[]
    ang_gamma=[]
    ang_crit=[]
    Fs=[]
    prod=[]
    
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
        gamma =np.rad2deg(np.arccos((np.dot(S,J))/(norm(S)*norm(J)))) #angle J,S
        gamma_crit= np.rad2deg(np.arcsin(71492/norm(J)))
        xi_2=np.rad2deg(np.arccos(np.dot(J,P)/(norm(J)*norm(P))))
        beta=np.rad2deg(np.arccos((np.dot(S_J,J))/(norm(S_J)*norm(J))))
        
        ang_zen.append(ang_zenith)
        ang_gamma.append(gamma)
        ang_crit.append(gamma_crit)
        sun_dist.append(norm(S))
        if  (gamma<=gamma_crit) or (ang_zenith>90 and ang_zenith<180): #eclipse & nightside condition
            F=0
        else:
            #F=1361/((const_dist)**2) #jovian orbit eccentricity=0
            F=1361/((norm(S)/1.496e+8)**2)
        
        
        time_ev[:,i]= solve(M,temp_0) #time step in column
        
        production=(R_0/((rho*cp*alpha)*T_0))*F*(1-Albedo)*abs(np.cos(np.deg2rad(ang_zenith)))  
        therm_rad_Jup=(R_0/T_0)*const*(71492/norm(J))**2*abs(np.cos(np.deg2rad(xi_2)))
        sun_ref_Jup=(R_0/T_0)*const_sunlight*(71492/norm(J))**2*abs(np.cos(np.deg2rad(xi_2)))*(beta/180)
      
        #Neumann BC explicit
        if i_lon<90 or i_lon>270:
            time_ev[0,i]=sun_ref_Jup*dr+therm_rad_Jup*dr+production*dr-loss*dr*temp_0[0]**4+time_ev[1,i] #outer bc with add. prod terms at sub jov side
        else:
            time_ev[0,i]=production*dr-loss*dr*temp_0[0]**4+time_ev[1,i]
        
        time_ev[L-1,i]=inner_bc*dr+time_ev[L-2,i]
        

        temp_0=time_ev[:,i].copy()
        
        Fs.append(F)
        prod.append(production*(T_0/R_0)*rho*cp*alpha)
        
        
    return time_ev,np.array(ang_zen),np.array(ang_gamma),np.array(ang_crit),np.array(Fs),np.array(prod),np.array(sun_dist)
#%%Solar heat flux
def solarheat_flux(t_vals,times,i_lat,i_lon,Albedo):
    #Initial condition = first column
    Nt=len(t_vals)
    #time_ev[:,0]=u_0
    
    
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
    
    Fs=[]
    ang_zen=[]

    
    for i in range(1,Nt):
        #Insolation and zenithangle (individually for each point (theta, phi) on surface)
        #angles
        P=  np.array(spice.sphrec(1821.5, np.deg2rad(90-i_lat), np.deg2rad(360-i_lon)))       #position vector on Io's surface (R_Io from Jovian Satellize Fact sheet), colat and lat (according to IAU) in radians
        J, lighttimesJ= spice.spkpos('JUPITER', times[i], 'IAU_IO', 'NONE', 'IO')         # vector from Io's to Jupiters Barycenter
        S, lighttimesS= spice.spkpos('SUN', times[i], 'IAU_IO', 'NONE', 'IO')        #vector from Io's to sun's barycenter 
       
        
        J=np.array(J.T)
        S=np.array(S.T)
    
        
        #needed values for heat equation 
        ang_zenith=np.rad2deg(np.arccos(np.dot(S,P)/(norm(S)*norm(P))))       # xi
        gamma =np.rad2deg(np.arccos((np.dot(S,J))/(norm(S)*norm(J)))) #angle J,S
        gamma_crit= np.rad2deg(np.arcsin(71492/norm(J)))
    
        
        if  (gamma<=gamma_crit) or (ang_zenith>90 and ang_zenith<180): #eclipse & nightside condition
            F=0
        else:
            #F=1361/((const_dist)**2) #jovian orbit eccentricity=0
            F=1361/((norm(S)/1.496e+8)**2)
        
        prod=F*(1-Albedo)*abs(np.cos(np.deg2rad(ang_zenith)))
        
        Fs.append(prod)
        ang_zen.append(ang_zenith)
    
    return Fs,ang_zen
#%% Create full time series

def time_evolution_full(temp_0,z_vals,t_vals,times,M,dr,dt,loss,inner_bc,i_lat,i_lon,R_0,T_0,rho,cp,alpha,Albedo,eq_count,step_day,temp1,utc_times):
    print('Lat=',i_lat)
    print('Lon=',i_lon)
    L=len(temp_0)
    #Initial condition = first column
    Nt=len(t_vals)
    #time_ev[:,0]=u_0
    
    
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
    time_ev=np.zeros((len(z_vals),len(t_vals)))
    
    #Initial condition = first column
    time_ev[:,0]=temp_0
    
    
    const_dist=5.2 #AU
    ang_zen=[]
    sun_dist=[]
    ang_gamma=[]
    ang_crit=[]
    Fs=[]
    prod=[]
    
    for i in range(1,Nt):
        #Insolation and zenithangle (individually for each point (theta, phi) on surface)
        #angles
        P=np.array(spice.sphrec(1821.5, np.deg2rad(90-i_lat), np.deg2rad(360-i_lon)))       #position vector on Io's surface (R_Io from Jovian Satellize Fact sheet), colat and lat (according to IAU) in radians
        J, lighttimesJ= spice.spkpos('JUPITER', times[i], 'IAU_IO', 'NONE', 'IO')         # vector from Io's to Jupiters Barycenter
        S, lighttimesS= spice.spkpos('SUN', times[i], 'IAU_IO', 'NONE', 'IO')        #vector from Io's to sun's barycenter 
   
        J=np.array(J.T)
        S=np.array(S.T)
        
        #needed values for heat equation 
        ang_zenith=np.rad2deg(np.arccos(np.dot(S,P)/(norm(S)*norm(P))))       # xi
        gamma =np.rad2deg(np.arccos((np.dot(S,J))/(norm(S)*norm(J)))) #angle J,S
        gamma_crit= np.rad2deg(np.arcsin(71492/norm(J)))
        
        ang_zen.append(ang_zenith)
        ang_gamma.append(gamma)
        ang_crit.append(gamma_crit)
        sun_dist.append(norm(S))
        if  (gamma<=gamma_crit) or (ang_zenith>90 and ang_zenith<180): #eclipse & nightside condition
            F=0
        else:
            #F=1361/((const_dist)**2) #jovian orbit eccentricity=0
            F=1361/((norm(S)/1.496e+8)**2)
        
        
        time_ev[:,i]= solve(M,temp_0) #time step in column
        
        production=(R_0/((rho*cp*alpha)*T_0))*F*(1-Albedo)*abs(np.cos(np.deg2rad(ang_zenith)))  
        
      
        #Neumann BC explicit
        time_ev[0,i]=production*dr-loss*dr*temp_0[0]**4+time_ev[1,i]
        time_ev[L-1,i]=inner_bc*dr+time_ev[L-2,i]
        

        temp_0=time_ev[:,i].copy()
        
        Fs.append(F)
        prod.append(production*(T_0/R_0)*rho*cp*alpha)
        
        
    return time_ev[0,int(eq_count*step_day+temp1):int(eq_count*step_day+temp1+step_day)],utc_times[int(eq_count*step_day+temp1):int(eq_count*step_day+temp1+step_day)] #One day in thermal eq- only surface temp

#%%
def find_nearest_time_idx(date, date_list, position=''): 
    '''
    

    Parameters
    ----------
    date : TYPE
        DESCRIPTION.
    date_list : TYPE
        DESCRIPTION.
    position : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    idx : TYPE
        DESCRIPTION.

    '''
    
    if date in date_list:
        idx=np.where(date_list==date)[0][0]
    else:
        if position=='start':
            temp=datetime.strptime(date,'%Y-%m-%d')
            temp=temp-timedelta(days=1)
            date=datetime.strftime(temp,"%Y-%m-%d")
            idx=np.where(date_list==date)[0][0]
        elif position=='end':
            temp=datetime.strptime(date,'%Y-%m-%d')
            temp=temp+timedelta(days=1)
            date=datetime.strftime(temp,"%Y-%m-%d")
            idx=np.where(date_list==date)[0][0]
    return idx