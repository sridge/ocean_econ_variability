import xarray as xr
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

from scipy import stats

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import scipy.io as scio
import cartopy
import cmocean

def fix_cesm_time(ds):

    time_new = ds.time-datetime.timedelta(seconds=1)
    ds['time'] = time_new
    
    return ds

def mmolm3cms_to_molm2m(output_var):

    output_var = output_var*1e-2 # mmol/m3 cm/s to mmol/m2/s
#    output_var = output_var*ds.TAREA*1e-4*1e-3 # mmol/m2/s to mol/s
    
    nyears = len(output_var.groupby('time.year').groups)
    
    spm = np.array([31,28,31,30,31,30,31,31,30,31,30,31])*24*60*60
    spm = np.tile(spm,nyears)
    spm = xr.DataArray(spm,dims='time')
    output_var=output_var*spm

    return output_var

def corr2d(x,y,tdim):
    
    # Compute data length, mean and standard deviation along time axis: 
    n = len(x[tdim])
    xmean = x.mean(dim=tdim)
    ymean = y.mean(dim=tdim)
    xstd  = x.std(dim=tdim)
    ystd  = y.std(dim=tdim)

    # Compute covariance along time axis
    cov   =  ((x - xmean)*(y - ymean)).sum(dim=tdim)/(n)

    # Compute correlation along time axis
    cor   = cov/(xstd*ystd)
    
    return cor

def prep4plot(tlat,tlon,X):
    tlon = np.where(np.greater_equal(tlon,min(tlon[:,0])),tlon-360,tlon)

    tlon = np.concatenate((tlon,tlon+360),1)
    tlat = np.concatenate((tlat,tlat),1)
    X = np.concatenate((X,X),1)

    tlon = tlon-360.

    return tlat,tlon,X

def  plot_global(lon,lat,grid,vmin,vmax,cmap,cesm=True):
    
    if cesm:
        
        lat,lon,grid = prep4plot(lat,lon,grid)
        lon,lat,grid = lon[:,0:450],lat[:,0:450],grid[:,0:450]
    

    fig_dpi = 300
    coast_color = '0.5'
    bg_color = '0.2'

    fig = plt.figure(dpi=fig_dpi)

    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()

    cax = plt.pcolormesh(lon,lat,grid,
                         cmap=cmap,
                         transform=ccrs.PlateCarree(),
                         vmin = vmin, vmax = vmax)

    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m',
                                                    edgecolor = coast_color,
                                                    linewidth = 0.5,
                                                    facecolor = bg_color))
    
    return ax,cax

def time_series(ax=plt.gca()):
    
    plt.grid(axis='y',alpha=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tick_params(left=False)
    ax.xaxis.set_ticks_position('bottom')


def twinx(x,y1,y2,xlabel,y1label,y2label,color1='tab:orange',color2='tab:blue',dpi=None):
    
    # get max,min,and standard deviation
    std_y1 = stats.tstd(y1) 
    mu_y1 = y1.mean()
    
    std_y2 = stats.tstd(y2) 
    mu_y2 = y2.mean()
    
    # normalize using z-score
    y1 = stats.zscore(y1)
    y2 = stats.zscore(y2)
    
    # max and min z-scores, will used for ylim
    glo_max = np.array([y2.max(),y1.max()])
    glo_max = glo_max.max()
    
    glo_min = np.array([y2.min(),y1.min()])
    glo_min = glo_min.min()
    
    fig, ax1 = plt.subplots()
    
    if dpi:
        fig.dpi = dpi

    # first plot --------
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label, color='k')
    ax1.plot(x, y1, color=color1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    
    ax1.set_ylim(glo_min-0.5,glo_max) # ylim set to z-score max and min
    # convert z-score ylabels back to raw values
    locs, labels = plt.yticks()
    labels = locs*std_y1 + mu_y1
    labels = np.round(labels,2)
    plt.yticks(locs,labels)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.tick_params(axis='y', labelcolor=color1, left=False)
    ax1.grid(axis='y',alpha=0.5)
    
    # second plot --------
    ax2.set_ylabel(y2label, color='k')  # we already handled the x-label with ax1
    ax2.plot(x, y2, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2, right=False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.set_ticks_position('bottom')
    
    ax2.set_ylim(glo_min-0.5,glo_max) # ylim set to z-score max and min
    # convert z-score ylabels back to raw values
    labels = locs*std_y2 + mu_y2
    labels = np.round(labels,2)
    plt.yticks(locs,labels)

    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    return ax1,ax2
    
def plot_rcp(scen):
    
    atm_co2_hist = xr.open_dataset(f'../4.5 vs 8.5/ATM_CO2_HIST.nc')
    atm_co2_proj = xr.open_dataset(f'../4.5 vs 8.5/ATM_CO2_{scen}.nc')

    atm_co2 = xr.concat((atm_co2_hist['ATM_CO2'],atm_co2_proj['ATM_CO2']),dim='year')

    plt.figure(dpi=300)
    plt.plot(atm_co2.year.sel(year=slice(1850,2005)),atm_co2.sel(year=slice(1850,2005)),label='observed')
    plt.plot(atm_co2.year.sel(year=slice(2006,2100)),atm_co2.sel(year=slice(2006,2100)),label='predicted')
    plt.ylabel('Atmospheric CO$_2$ (ppm)')
    # plt.axvline(2006,linestyle='--',color='0.4')
    plt.xlabel('Year')
    time_series(ax=plt.gca())
    # plt.xlim(1850,2100)
    plt.legend()
    
