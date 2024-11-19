 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 09:43:06 2022

@author: anne-cathrinedott
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as colors
#from scipy.interpolate import make_interp_spline
#from scipy.interpolate import interp1d
import os 
import spiceypy as spice
from datetime import datetime
from datetime import timedelta, date
import matplotlib.dates as mdates
#import matplotlib.animation as manimation
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D



#%% import functions
import sys
sys.path.append('/Users/anne-cathrinedott/Desktop/Uni/Promotion/IoSpice')
import implicit_resources, explicit_resources
#%%
T_reference = np.load('T_map.npy') #Surface Temp Map (Instantaneous from old model) as start
T_0=T_reference[90,90] #K 

#%% Parameters/ Constants

s = 5.67e-8 # (Wm^-2K^-4)  Stefan Boltzman constant 
A = 0.62 # spherical Albedo Io #0.62
Io_day = 42.5 #h (Day length Io)
Earth_day=24

K=1.46  # values from Leone et al. K = mean of  10km und 0 km
rho = 2096 #kgm^-3 (Leone et al.)
cp = 290 # Jkg^-1K^-1 (Leone et al.)
eps=0.9     #(Matson et al) Emissivity 

fac=1/10 #Factor for parameter Study
alpha = (K/(rho*cp))*fac  #m^2s^-1 (thermal diffusivity)

T_initial=100#K equilibrium Temperature (without insolation) 90.1 (J=3), 60.3 (J=3*0.2) 57 (J=2.4*0.2) 59.5
Jint_fac=1
J_int = 2.4*0.2*Jint_fac #W/m^2 #Internal heating rate when assuming that 80% is lost through volc. eruptions

eps_Jup=0.9
T_Jup=165
A_IR=0.5
A_Jup=0.54
F_Jup=50

depth = 2 #m (maximum modeling depth)
simulation_days=300 #amount of simulated Io days -> one Jovian Year:2980(eq=500) 300 for parameter study maps
                                #4075 for jov year plot


#%% Kernels 
kernelsPath = os.path.join(os.path.dirname(os.getcwd()),"kernels")
leapSecondsKernelPath = os.path.join(kernelsPath,"naif0012.tls")
spice.furnsh(leapSecondsKernelPath)
#%%
start_date= '03/21/98'#northern summer (start: 03/21/98 --> around 99/06/07) PERIHELION
                                        #northern winter (start:05/21/04 --> around 05/07/08) APHELION
                                        #'01/01/94' for jov-year simulation
start_date_temp=datetime.strptime(start_date,"%m/%d/%y")

end_time=simulation_days*(Io_day/Earth_day)  #Io days (-> convert into earth days)  #amount of Io days that are simulated 

end_date = start_date_temp + timedelta(days=end_time) #earth end_date

end_date = end_date.strftime("%m/%d/%Y, %H:%M:%S")
start_date=start_date_temp.strftime("%m/%d/%Y, %H:%M:%S")
del start_date_temp

#Only 3 Days to check things
#start_date='04/01/2022'
#end_date='04/04/2022'
#end_time=3


#Transfer into number of TDB seconds past the J2000
et_start = spice.str2et(start_date)
et_end = spice.str2et(end_date)

#%% Set up Domain (all in Io days)
end_time_io=end_time/(Io_day/Earth_day)
layers=65

#t_0=((Io_day/24)*60*60) #Earth day in s --> Plot is than local time
t_0=Io_day*60*60
Z_0=(np.sqrt(alpha*t_0))

z_step=(depth/Z_0)/layers #/1000 für alpha=0 0.1

t_step=0.006 #resolution 15 minutes 0.006 (year phi=180), 9 minutes 0.004 (else),

t_end=end_time_io+t_step# (Earth days)
z_end=depth/Z_0+z_step

t=np.arange(0,t_end,t_step)  
z=np.arange(0,z_end,z_step)

Nt=len(t)
L=len(z)
#%%Input for SPICE


times = [x*(et_end-et_start)/Nt + et_start for x in range(Nt)]
utc_times=spice.et2utc(times, "ISOC", 0)
#%%
loss=eps*s*Z_0*T_0**3/(rho*cp*alpha) 
#loss=eps*s*T_0**3/(rho*cp) #for alpha=0
inner_bc=(Z_0/T_0)*(1/(rho*cp*alpha))*J_int

Prod_Jup_const=((s*eps_Jup*T_Jup**4)/(rho*cp*alpha))*(1-A_IR) #constant part of production term due to thermal radiation from Jupiter
sun_Jup_const=((F_Jup*A_Jup)/(rho*cp*alpha))*(1-A) #constant part of production term due to sunlight reflected from Jupiter

#%% Temporal resolution
resolution=t_step*Io_day*60 #temporal resolution (in minutes)

#%%Matrix 
M=implicit_resources.define_matrix(L,alpha,t_0,Z_0,t_step,z_step)

#%% Grid
#%% Grid 

####map complete -> 5° Grid
#latitude_deg=np.arange(-90,-44,5) #-90 - -45 --> s_two
latitude_deg=np.arange(-40,1,5) #-40 - 0  --> s_one
#latitude_deg=np.arange(5,46,5) #5 - 45  --> n_two
#latitude_deg=np.arange(50,91,5) #50- 90  --> n_one

longitude_deg=np.arange(0,361,5)

#####one day
#latitude_deg=np.arange(-90,1) #SHS latitude in degrees 
#latitude_deg=np.arange(1,91)  #NHS

########complete
#latitude_deg=np.arange(-90,91)
#longitude_deg=np.arange(0,361) #longitude in degrees


#%% Some things to work with only one Io day
step_day=round(1/t_step,1)#+1 #time steps per Io day With 0:00 and 24:00 is same point
array_day=np.arange(0,Nt,int(step_day)) #len(array_day)=amount of modeled days 
t_day=np.linspace(0,Earth_day,int(step_day)) #times of a day in Earth hours (for plotting)

#%%Solar heat Flux as a function of Latitude
#theta=np.arange(0,91)
#phi=0
#F_sol=[]

#for i in range(0,len(theta)):
  #  print(theta[i])
    #Fs,zenith_angls=implicit_resources.solarheat_flux(t,times,theta[i],phi,A)
    #Fs_daymax=np.amax(Fs)
    #F_sol.append(Fs_daymax)
    
#%%First: for theta=0, phi=0
theta=0 #latitude
phi=0  #longitude

time_series=np.zeros((len(z),len(t)))
time_series[:,0]=T_initial/T_0

time_series,ang_zenith,ang_gamma,gamma_crit,Fs,prod,sun_dist=implicit_resources.time_evolution(time_series[:,0], z, t,times, M,z_step,t_step, loss, inner_bc, theta, phi,Z_0,T_0,rho,cp,alpha,A,Prod_Jup_const,sun_Jup_const)
time_series=time_series*T_0

#plt.plot(t,time_series[0,:])


#%% Find Universal time with reference time series at
eq_count=250 #(Io_days) 250 for parameter study, 500
#plot(x)
#%% 
ref_time_series=np.zeros((len(z),len(t)))
ref_ang_zenith=np.zeros(len(t))
ref_time_series[:,0]=T_initial/T_0
ref_ang_zenith=implicit_resources.time_evolution(ref_time_series[:,0], z, t,times, M,z_step,t_step, loss, inner_bc, 0, 0,Z_0,T_0,rho,cp,alpha,A,Prod_Jup_const,sun_Jup_const)[1][:]
print('Ref')
LT_start=explicit_resources.find_LT(ref_ang_zenith,eq_count,step_day,ref_time_series)
del ref_time_series,ref_ang_zenith


#%% Thermal equilibrium arrays
#eq_count=explicit_resources.equilibrium_counter(array_day,step_day,time_series[0,:],T_0)

Temp_series,utc_times_1d=explicit_resources.one_day_timeseries(eq_count, step_day,LT_start, time_series,utc_times)
Ang_series=explicit_resources.one_day_timeseries(eq_count, step_day,LT_start, ang_zenith,utc_times)[0]
Crit_series=explicit_resources.one_day_timeseries(eq_count, step_day,LT_start, gamma_crit,utc_times)[0]
Fs_series=explicit_resources.one_day_timeseries(eq_count, step_day,LT_start, Fs,utc_times)[0]
Gamma_series=explicit_resources.one_day_timeseries(eq_count, step_day,LT_start, ang_gamma,utc_times)[0]
prod_series=explicit_resources.one_day_timeseries(eq_count, step_day, LT_start,prod,utc_times)[0]

point_LT=explicit_resources.find_LT(Ang_series,0,step_day,Temp_series) #= 0:00 h point for 1 Day time series for Io local time check
#%%date from str into datetime
utc_1d=[datetime.strptime(d,"%Y-%m-%dT%H:%M:%S") for d in utc_times_1d]

#%% Whole Io -->all lats and lons  

time_series_full=np.zeros((len(latitude_deg),len(longitude_deg),len(t_day)))
time_series_temp=np.zeros((len(z),len(t)))
time_series_temp[:,0]=T_initial/T_0
J_int_arr=[]
case='mixed_new'

if case=='mixed_1':
    for i in range(len(longitude_deg)):
        for j in range(len(latitude_deg)):
            if latitude_deg[j]<=-45 and latitude_deg[j]>-70: #heat flux at poles higher
                print('Add heat flux at:' ,latitude_deg[j])
                J_int = (2.4)*0.4 #W/m^2 #Internal heating rate when assuming that 80% is lost through volc. eruptions
                inner_bc=(Z_0/T_0)*(1/(rho*cp*alpha))*J_int
                time_series_full[int(len(latitude_deg)-1-j),i,:]=implicit_resources.time_evolution_full(time_series_temp[:,0], z, t,times, M, z_step, t_step,loss,inner_bc, latitude_deg[j], longitude_deg[i],Z_0,T_0,rho,cp,alpha,A,eq_count,step_day,LT_start,utc_times)[0]
            if latitude_deg[j]<=-70: #heat flux at poles higher
                print('Add heat flux at:' ,latitude_deg[j])
                J_int = (2.4)*0.6 #W/m^2 #Internal heating rate when assuming that 80% is lost through volc. eruptions
                inner_bc=(Z_0/T_0)*(1/(rho*cp*alpha))*J_int
                time_series_full[int(len(latitude_deg)-1-j),i,:]=implicit_resources.time_evolution_full(time_series_temp[:,0], z, t,times, M, z_step, t_step,loss,inner_bc, latitude_deg[j], longitude_deg[i],Z_0,T_0,rho,cp,alpha,A,eq_count,step_day,LT_start,utc_times)[0]    
            else:
                J_int = (2.4+1)*0.2 #initital inner bc
                inner_bc=(Z_0/T_0)*(1/(rho*cp*alpha))*J_int
                time_series_full[int(len(latitude_deg)-1-j),i,:]=implicit_resources.time_evolution_full(time_series_temp[:,0], z, t,times, M, z_step, t_step,loss,inner_bc, latitude_deg[j], longitude_deg[i],Z_0,T_0,rho,cp,alpha,A,eq_count,step_day,LT_start,utc_times)[0]

if case=='mixed_new':
    for i in range(len(longitude_deg)):
        for j in range(len(latitude_deg)):
            if latitude_deg[j]<=-60 or latitude_deg[j]>=60: #heat flux at poles higher
                print('Add heat flux at:' ,latitude_deg[j])
                J_int = 1.44-(0.96*math.cos(math.radians(latitude_deg[j])))
                J_int_arr.append(J_int)
                inner_bc=(Z_0/T_0)*(1/(rho*cp*alpha))*J_int
                time_series_full[int(len(latitude_deg)-1-j),i,:]=implicit_resources.time_evolution_full(time_series_temp[:,0], z, t,times, M, z_step, t_step,loss,inner_bc, latitude_deg[j], longitude_deg[i],Z_0,T_0,rho,cp,alpha,A,eq_count,step_day,LT_start,utc_times)[0]
            else:
                J_int = (2.4)*0.2 #initital inner bc
                J_int_arr.append(J_int)
                inner_bc=(Z_0/T_0)*(1/(rho*cp*alpha))*J_int
                time_series_full[int(len(latitude_deg)-1-j),i,:]=implicit_resources.time_evolution_full(time_series_temp[:,0], z, t,times, M, z_step, t_step,loss,inner_bc, latitude_deg[j], longitude_deg[i],Z_0,T_0,rho,cp,alpha,A,eq_count,step_day,LT_start,utc_times)[0]

if case=='normal':
    for i in range(len(longitude_deg)):
        for j in range(len(latitude_deg)):
             time_series_full[int(len(latitude_deg)-1-j),i,:]=implicit_resources.time_evolution_full(time_series_temp[:,0], z, t,times, M, z_step, t_step,loss,inner_bc, latitude_deg[j], longitude_deg[i],Z_0,T_0,rho,cp,alpha,A,eq_count,step_day,LT_start,utc_times)[0]
             
         #complere- not parallel
         #time_series_full[longitude_deg[i],90+latitude_deg[j],:]=explicit_resources.euler_time_evolution_full(time_series_temp[:,0], z, t,times, alpha, loss, inner_bc, latitude_deg[j], longitude_deg[i],Z_0,T_0,t_0,rho,cp,A,eq_count,step_day,LT_start,utc_times)[0]
    
#np.save('Jint_array.npy',J_int_arr)   
#corr_times_1=explicit_resources.euler_time_evolution_full(time_series_temp[:,0], z, t,times, alpha, loss, inner_bc, latitude_deg[0], longitude_deg[0],Z_0,T_0,t_0,rho,cp,A,eq_count,step_day,LT_start,utc_times,Prod_Jup_const,sun_Jup_const)[1]
#corr_times=[datetime.strptime(d,"%Y-%m-%dT%H:%M:%S") for d in corr_times_1]
#del corr_times_1
       
time_series_full=time_series_full*T_0
np.save('T_Map_Perihelion_mixedenhanced_s_one.npy',time_series_full)
#np.save('T_Map_Perihelion_corrtimes_mixedenhanced.npy',corr_times)    

#%% Ang series for all Longitudes -> Zenith angles to determine the dayside 
#longitude_deg=np.arange(0,361,5)
#ang_series_full=np.zeros((len(t_day),len(longitude_deg)))

#for i in range(len(longitude_deg)):
  #  ang_series_full[:,i]=explicit_resources.Ang_series_map(z, t,longitude_deg[i], times, eq_count, step_day,LT_start,utc_times)
 
#np.save('T_Map_Perihelion_Angzenith_mixedenhanced.npy',ang_series_full)  

plot(x)

#%% All Latitudes but one Day (time series full one day) - Parallel

time_series_full=np.zeros((len(latitude_deg),len(t_day)))
time_series_temp=np.zeros((len(z),len(t)))
time_series_temp[:,0]=T_initial/T_0

for i in range(len(latitude_deg)):
   #complete:
   #time_series_full[:,90+latitude_deg[i]]=explicit_resources.euler_time_evolution_full(time_series_temp[:,0], z, t,times, alpha, loss, inner_bc, latitude_deg[i], phi,Z_0,T_0,t_0,rho,cp,A,eq_count,step_day,LT_start,utc_times,Prod_Jup_const,sun_Jup_const)[0]
   
   #parellel northern- southern
   time_series_full[int(len(latitude_deg)-1-i),:]=explicit_resources.euler_time_evolution_full(time_series_temp[:,0], z, t,times, alpha, loss, inner_bc, latitude_deg[i], phi,Z_0,T_0,t_0,rho,cp,A,eq_count,step_day,LT_start,utc_times,Prod_Jup_const,sun_Jup_const)[0]
   
   
#corr_times_1=explicit_resources.euler_time_evolution_full(time_series_temp[:,0], z, t,times, alpha, loss, inner_bc, latitude_deg[i], phi,Z_0,T_0,t_0,rho,cp,A,eq_count,step_day,LT_start,utc_times,Prod_Jup_const,sun_Jup_const)[1]
#corr_times=[datetime.strptime(d,"%Y-%m-%dT%H:%M:%S") for d in corr_times_1]
#del corr_times_1
       
time_series_full=time_series_full*T_0
time_series_full=time_series_full.T
np.save('ts_full_anti-jov_shs_phi=180_pericenter.npy',time_series_full)
#np.save('ts_full_anti-jov_phi=180_pericenter_corrtimes.npy',corr_times)



#%% Parameter study heat flux at poles. 
heat_fluxes=[0.4,0.55,0.7,0.85,1,1.15,1.3,1.45,1.6,1.75,1.9,2.05,2.2,2.35,2.5,2.65,2.8,3]
heat_fluxes=np.asarray(heat_fluxes)
south_pole_temp=[]
north_pole_temp=[]

theta=-90
for i in range(len(heat_fluxes)):
    time_series=np.zeros((len(z),len(t)))
    time_series[:,0]=T_initial/T_0
    print(theta)
    print(heat_fluxes[i])
    inner_bc=(Z_0/T_0)*(1/(rho*cp*alpha))*heat_fluxes[i]
    time_series,ang_zenith,ang_gamma,gamma_crit,Fs,prod,sun_dist=implicit_resources.time_evolution(time_series[:,0], z, t,times, M,z_step,t_step, loss, inner_bc, theta, phi,Z_0,T_0,rho,cp,alpha,A,Prod_Jup_const,sun_Jup_const)
    time_series=time_series*T_0
    eq_count=250 #(Io_days) 250 for parameter study, 50^0 
    ref_time_series=np.zeros((len(z),len(t)))
    ref_ang_zenith=np.zeros(len(t))
    ref_time_series[:,0]=T_initial/T_0
    ref_ang_zenith=implicit_resources.time_evolution(ref_time_series[:,0], z, t,times, M,z_step,t_step, loss, inner_bc, 0, 0,Z_0,T_0,rho,cp,alpha,A,Prod_Jup_const,sun_Jup_const)[1][:]
    LT_start=explicit_resources.find_LT(ref_ang_zenith,eq_count,step_day,ref_time_series)
    del ref_time_series,ref_ang_zenith
    Temp_series,utc_times_1d=explicit_resources.one_day_timeseries(eq_count, step_day,LT_start, time_series,utc_times)
    south_pole_temp.append(Temp_series[0,0])
    
theta=90
for i in range(len(heat_fluxes)):
    time_series=np.zeros((len(z),len(t)))
    time_series[:,0]=T_initial/T_0
    print(theta)
    print(heat_fluxes[i])
    inner_bc=(Z_0/T_0)*(1/(rho*cp*alpha))*heat_fluxes[i]
    time_series,ang_zenith,ang_gamma,gamma_crit,Fs,prod,sun_dist=implicit_resources.time_evolution(time_series[:,0], z, t,times, M,z_step,t_step, loss, inner_bc, theta, phi,Z_0,T_0,rho,cp,alpha,A,Prod_Jup_const,sun_Jup_const)
    time_series=time_series*T_0
    eq_count=250 #(Io_days) 250 for parameter study, 50^0 
    ref_time_series=np.zeros((len(z),len(t)))
    ref_ang_zenith=np.zeros(len(t))
    ref_time_series[:,0]=T_initial/T_0
    ref_ang_zenith=implicit_resources.time_evolution(ref_time_series[:,0], z, t,times, M,z_step,t_step, loss, inner_bc, 0, 0,Z_0,T_0,rho,cp,alpha,A,Prod_Jup_const,sun_Jup_const)[1][:]
    LT_start=explicit_resources.find_LT(ref_ang_zenith,eq_count,step_day,ref_time_series)
    del ref_time_series,ref_ang_zenith
    Temp_series,utc_times_1d=explicit_resources.one_day_timeseries(eq_count, step_day,LT_start, time_series,utc_times)
    north_pole_temp.append(Temp_series[0,0])




#%% Year's daily mean (0,0)
time_series_year,total_days,utc_year=explicit_resources.ts_jovian_year(time_series, eq_count, step_day, ang_zenith, utc_times)
Fs_year=explicit_resources.ts_jovian_year(Fs, eq_count, step_day, ang_zenith, utc_times)[0]
sun_dist_year=explicit_resources.ts_jovian_year(sun_dist, eq_count, step_day, ang_zenith, utc_times)[0]
prod_year=explicit_resources.ts_jovian_year(prod, eq_count, step_day, ang_zenith, utc_times)[0]
utc_year=[datetime.strptime(d,"%Y-%m-%dT%H:%M:%S") for d in utc_year]
#%%
days_mean=[] #daily averaged surface temperature
days_max=[] #daily max surface temperature
days_min=[] #daily min surface temperature
Fs_mean=[] #daily averaged solar irradiance 
sun_dist_mean=[] #daily averaged distance to sun
Fs_max=[] #maximum solar irradiance
utc_means=[] #array for means
prod_max=[] #daily max heat production rate

days_max_00=[]
test_max_00=[]
utc_means_00=[]

count=0
for i in range(0,total_days):
    count=count+1
    
    mean_temp=np.mean(time_series_year[int(i*step_day):int((i+1)*step_day)])
    mean_Fs=np.mean(Fs_year[int(i*step_day):int((i+1)*step_day)])
    mean_sun_dist=np.mean(sun_dist_year[int(i*step_day):int((i+1)*step_day)])/1.496e+8
    max_Fs=np.amax(Fs_year[int(i*step_day):int((i+1)*step_day)])
    max_prod=np.amax(prod_year[int(i*step_day):int((i+1)*step_day)])
    day_curr=utc_year[int(step_day*i)]
    max_days=np.max(time_series_year[int(i*step_day):int((i+1)*step_day)])
    min_days=np.min(time_series_year[int(i*step_day):int((i+1)*step_day)])
    
   
    
    days_mean.append(mean_temp)
    days_max.append(max_days)
    days_min.append(min_days)
    Fs_mean.append(mean_Fs)
    prod_max.append(max_prod)
    sun_dist_mean.append(mean_sun_dist)
    Fs_max.append(max_Fs)
    utc_means.append(day_curr)
    
    if (count%70==0): #speichert nur jeden x. punkt
        test_max_00.append(max_days)
        utc_means_00.append(day_curr)
    
    del mean_temp,mean_Fs,max_Fs,mean_sun_dist,day_curr,max_prod
    
if phi==0:
    for i in range(0,len(days_max)):
        max_days_00=np.mean(days_max[i:i+10])
      #  
        days_max_00.append(max_days_00)
        #
        del max_days_00
    
    


day_means_proxy=np.add(days_max,days_min)/2
N_mean_proxy=explicit_resources.so2_density_cm2(np.array(day_means_proxy))
    

N_mean=explicit_resources.so2_density_cm2(np.array(days_mean))
N_max=explicit_resources.so2_density_cm2(np.array(days_max))
N_min=explicit_resources.so2_density_cm2(np.array(days_min))
#%%Determine Equinoxes and Solstices
kernelsPath = os.path.join(os.path.dirname(os.getcwd()),"kernels")
leapSecondsKernelPath = os.path.join(kernelsPath,"naif0012.tls")
spice.furnsh(leapSecondsKernelPath)

temp_date=np.array([datetime.strftime(d,"%Y-%m-%d") for d in utc_means])
temp=[datetime.strftime(d,"%Y-%m-%d %H:%M:%S") for d in utc_means]
utc_means_et=[spice.utc2et(temp[x]) for x in range(len(utc_means))] #convert utc_means from datetime to string and to et

season_times=explicit_resources.equinoxes(utc_means_et,utc_means)

solstices_idx=explicit_resources.solstices(utc_means_et,utc_means)
#del temp,utc_means_et  








