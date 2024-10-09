import numpy as np
import xarray as xr
import glob, os, sys
import matplotlib.pyplot as plt
import calendar
from datetime import date, datetime, timedelta
import time
from wavesnspikes import wns
import cartopy.crs as ccrs
import cartopy.feature as cfeature


key = "DATA_PATH"
value = os.getenv(key)
data_path = value   #"~/blue-cloud-dataspace/MEI/CMCC/MHW/input/test_dataset"

def intensity(SST,pc90,clim):
    intensity=SST-pc90
    intensity=np.ma.masked_where(intensity<0,intensity)
    intensity[np.isnan(intensity)]=0
    anomaly=SST-clim
    anomaly[np.isnan(anomaly)]=0

    MHW=np.copy(intensity)
    MHW[MHW>0]=1
    MHW[MHW<0]=0

    return anomaly, MHW

year=2024
mon=7
day=31 # why this is 14 instead of 13? as it is below

if year==2024:
    op=True
else:
    op=False
    
# target_date=date(2022,5,13)
target_date=date(year,mon,day)

# SATELLITE DATA # -- LOCAL SETUP

# CMS OBS Satellite SST #
# path_obs="/data/inputs/METOCEAN/historical/obs/ocean/satellite/CMS/MedSea/SST/L4/day/"

path_obs=data_path
if op:
    # file_obs=path_obs+"{0}/{1}/{0}{1}{2}000000-GOS-L4_GHRSST-SSTfnd-OISST_HR_NRT-MED-v02.0-fv03.0.nc".format(year,str(mon).zfill(2),str(day).zfill(2))
    # clim_obs=xr.open_dataset("/data/cmcc/rm33823/CMS_SST/CMS_SST_Climatology_19872021_rect00625.nc")
    # area_obs=xr.open_dataset("/data/cmcc/rm33823/CMS_SST/cell_areas_CMS_NRT.nc")    
    #file_obs=path_obs+"/{0}{1}{2}000000-GOS-L4_GHRSST-SSTfnd-OISST_HR_NRT-MED-v02.0-fv03.0.nc".format(year,str(mon).zfill(2),str(day).zfill(2))
    file_obs = path_obs+"/NRT.nc"
    #clim_obs=xr.open_dataset(f'{data_path}/CMS_SST_Climatology_19872021_rect00625.nc')
    clim_obs=xr.open_dataset(f'{data_path}/CLIM.nc')
    #area_obs=xr.open_dataset(f'{data_path}/cell_areas_CMS_NRT.nc') 
    area_obs=xr.open_dataset(f'{data_path}/CELL.nc')
else:
    file_obs=path_obs+"{0}/{1}/{0}{1}{2}000000-GOS-L4_GHRSST-SSTfnd-OISST_HR_REP-MED-v02.0-fv03.0.nc".format(year,str(mon).zfill(2),str(day).zfill(2))
    clim_obs=xr.open_dataset("/data/cmcc/rm33823/CMS_SST/CMS_SST_Climatology_19872021.nc")
    area_obs=xr.open_dataset("/data/cmcc/rm33823/CMS_SST/cell_areas_CMS.nc")                                  

data_obs=xr.open_dataset(file_obs)
obs_lons=clim_obs.lon.values
obs_lats=clim_obs.lat.values

grid_area=area_obs.cell_area.values

obs=data_obs.analysed_sst.values[0]
targetday_index=target_date.toordinal()-date(year,1,1).toordinal()

obs_pc90=clim_obs.pc90.values[targetday_index]
obs_clim=clim_obs.clim.values[targetday_index]

anomaly_obs,MHW_obs=intensity(obs,obs_pc90,obs_clim)


fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(8,4.5),subplot_kw={'projection': ccrs.PlateCarree()} )
fig.subplots_adjust(bottom=0.02, top=0.92, left=0.02, right=0.87, wspace=0.05, hspace=0.05)
axes._autoscaleXon = False
axes._autoscaleYon = False

map1=axes.pcolormesh(obs_lons,obs_lats,anomaly_obs,vmin=-6,vmax=6,cmap='seismic',transform=ccrs.PlateCarree())
axes.contour(obs_lons,obs_lats,MHW_obs,levels=[0.5],colors=['maroon'],linewidths=[1],transform=ccrs.PlateCarree())

axes.set_title("Satellite Observations \n {}".format(target_date))

axes.coastlines()
axes.set_extent([-6, 36.5, 30, 44.])
axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='black', facecolor='lightgrey'))

cb_ax1 = fig.add_axes([0.90, 0.225, 0.01, 0.5])
cbar1 = fig.colorbar(map1, cax=cb_ax1, orientation='vertical',extend='both')
#cbar1.set_ticklabels([r"0",r"$\frac{1}{10}$",r"$\frac{1}{3}$",r"$\frac{1}{2}$",r"$\frac{2}{3}$",r"$\frac{9}{10}$",r"1"])
cbar1.set_label("Temperature Anomaly ($^oC$)")

plt.show()
plt.savefig("/output/Figure_2.png",dpi=300)
#plt.savefig("/work/oda/rm29119/Paper_OSR8_Figures/Figure_2.png",dpi=300)
