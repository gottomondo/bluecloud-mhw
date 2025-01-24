#!/usr/bin/env python
# coding: utf-8

# # Notebook: CMCC-MHW-Maps
# **Version**: v1.0-beta 
# 
# **Last update**: 2025-01-31
# 
# **Authors**: ...

# This notebook is designed for the operational analysis of Marine Heatwave (MHW) detection in the Mediterranean Sea, based on comparisons between climatological baselines and Sea Surface Temperature (SST) data from reprocessed (REP) or near-real-time (NRT) satellite observations. It connects to the Copernicus Marine Service (CMEMS) using the copernicusmarine API.
# 
# The notebook focus on 2D map visualization to enhance the understanding of the spatial dimensions of MHW events. Efficient processing of large spatial datasets is powered by xarray and numpy, while the wavesnspikes algorithm enables real-time detection of heatwaves.

import os, time, warnings
from datetime import date, datetime, timedelta

# import ipywidgets as widgets
# from IPython.display import display

import numpy as np
import xarray as xr
from wavesnspikes import wns
import copernicusmarine

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore") # Ignore all warnings

import sys
from config import CMEMS_datasets_options

args = sys.argv

# script_name = "mhw-method-maps.py"
script_name = args[0]

if len(sys.argv) != 9:
    print("Usage: python {}  {{ data_source }} {{ data_path }} {{ outputs_path }} {{ doi }} {{ lon_min }} {{ lon_max }} {{ lat_min }} {{ lat_max }}".format(script_name))
    print("Example: python ./{} SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_a_V2 /data /outputs 20240601 18 41.3 11.36 38.93".format(script_name))
    sys.exit(1)
    
# inputs
data_source = args[1]
data_path = args[2]
outputs_path = args[3]
doi = args[4]
lon_min, lon_max = float(args[5]), float(args[6])
lat_min, lat_max = float(args[7]), float(args[8])

# print(lat_min, lat_max)
# sys.exit(1)
# italy region
# lon_min, lon_max = 0, 20
# lat_min, lat_max = 34, 46
# lon_min, lon_max = -5, 1
# lat_min, lat_max = 34, 42
# lon_min, lon_max = 11.36, 38.93
# lat_min, lat_max = 18, 41.3

# class for generating mock object
class MockObj:
    def __init__(self, value):
        self.value = value
#
# objects conversions
#
date_picker = MockObj(value = datetime.strptime(doi, "%Y%m%d"))

lonmin_input = MockObj(value = lon_min)
lonmax_input = MockObj(value = lon_max)
latmin_input = MockObj(value = lat_min)
latmax_input = MockObj(value = lat_max)

# ### 1. Select the CMEMS dateset of interest
# - Mediterranean Satellite Reprocessed (REP) Sea Surface Temperature (SST) with data from 1982
# - Mediterranean Satellite Near Real Time SST (NRT) with data from 2008
# 
# - Climatology data: Long-term averages (from 1987–2021) to provide a baseline for comparing the CMEMS data.

# CMEMS_datasets_options = {'cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021':{'varname':  "analysed_sst", "prod_type": 'REP',
#                                                                          'grid_file':"prev/cell_areas_CMS.nc",
#                                                                          'clim_file':"CMS_SST_Climatology_19872021.nc"},
#                         'SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_a_V2': {'varname':  "analysed_sst", "prod_type": 'NRT',
#                                                                          'grid_file':"2024/cell_areas_CMS_NRT.nc",
#                                                                          'clim_file':"CMS_SST_Climatology_19872021_rect00625.nc"}}

# # CMEMS datasets selection
# print('Please select a Copernicus Marine Service (CMEMS) dataset:')
# datasets_dropdown = widgets.Dropdown(options = CMEMS_datasets_options.keys(), description = 'Dataset:', disabled = False)
# display(datasets_dropdown) # Display the dropdown

dataset_id = data_source

# paths
# clim_path  = "~/blue-cloud-dataspace/MEI/CMCC/MHW/input/clim/"
# clim_path  = "/workspace/VREFolders/MarineEnvironmentalIndicators/input_datasets/mhw/clim"
clim_path = data_path

# Set outputs dir
# out_dir    = "/workspace/MHW_figures/Maps/"
out_dir = outputs_path
os.makedirs(out_dir, exist_ok=True) # Check if the directory exists; create it if it doesn't

## MHW settings
time_dim   = 'time'
ndays_mhw  = 12 # number of days to consider in identifying and masking the MHWs
delta_days = timedelta(ndays_mhw-1)

# # 2. Load and set
# - Open climatology dataset
# - Query copernicusmarine api for selected dataset
# - Select date of interest
# - Select region of interest

# Climatology reference file
print("\nOpening climatology file...")
t0=time.time()
clim_rawdataset = xr.open_dataset(os.path.join(clim_path,CMEMS_datasets_options[dataset_id]['clim_file']))
print(f"\tDone ({time.time() - t0:.1f}s).")

varname    = CMEMS_datasets_options[dataset_id]['varname']
prod       = CMEMS_datasets_options[dataset_id]['prod_type']
print('Selected dataset -> %s'%(dataset_id))

# Querying copernicusmarine api
print("Querying copernicusmarine api...")
t0=time.time()
params = {"credentials_file": "bc2026_copernicusmarine-credentials",
          "dataset_id": dataset_id, "variables": [varname], "maximum_depth": 1.5,}  
cms_rawdataset = copernicusmarine.open_dataset(**params)
print('\tElapsed time: %ss'%(round(time.time()-t0,1)))
     
# Setting the date range
date_min = datetime.utcfromtimestamp(min(cms_rawdataset[time_dim].values).astype('datetime64[s]').astype(int)).date()
date_max = datetime.utcfromtimestamp(max(cms_rawdataset[time_dim].values).astype('datetime64[s]').astype(int)).date()
date_min_delta = date_min + delta_days
# date_picker    = widgets.DatePicker(description='Date', disabled=False,value=date_max, min=date_min_delta, max=date_max)
# print('\nPlease select the date of interest:\nDataset limits -> from %s to %s' %(date_min_delta, date_max))
# display(date_picker) # Displaying a DatePicker widget to select the date of interest

# Setting region of interest
print('\nInsert the coordinates of the region of interest:\nThe standard values are the dataset limits.')
n_dec = 3
# lon_min, lon_max = np.round(cms_rawdataset.longitude.min().item(),n_dec), np.round(cms_rawdataset.longitude.max().item(),n_dec)
# lat_min, lat_max = np.round(cms_rawdataset.latitude.min().item(),n_dec),  np.round(cms_rawdataset.latitude.max().item(),n_dec)

# lonmin_input = widgets.BoundedFloatText(description='Minimum:',value=lon_min,min=lon_min,max=lon_max,)
# lonmax_input = widgets.BoundedFloatText(description='Maximum:',value=lon_max,min=lon_min,max=lon_max,)
# latmin_input = widgets.BoundedFloatText(description='Minimum:',value=lat_min,min=lat_min,max=lat_max,)
# latmax_input = widgets.BoundedFloatText(description='Maximum:',value=lat_max,min=lat_min,max=lat_max,)
# # Display the widgets to select the region of interest
# print('Longitude:')
# display(widgets.VBox([lonmin_input, lonmax_input]))
# print('Latitude:')
# display(widgets.VBox([latmin_input, latmax_input]))


# ### 3. Filtering and Processing the Datasets
# - Date and region of interest filters
# - Compute Anomaly and MHW mask

def extract_clim_date_range(ds, dates_list, time_dim="days"):
    """
    Extract a specific range of dates from a climatology dataset.
    Parameters:
    - ds (xr.Dataset or xr.DataArray): The input dataset or data array with time_dim dimensions or coordinates.
    - dates_list (list of datetime): The list of the dates to extract
    - time_dim (str): The name of the time dimension in the dataset.
    Returns: The extracted subset of the dataset or data array as xr.Dataset or xr.DataArray.
    """
    # Helper functions to calculate climatological day-of-year with adjust for leap years
    def is_leap_year(year): return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
    def to_climatological_day(date):
        start_of_year = datetime(date.year, 1, 1)
        current_date  = datetime(date.year, date.month, date.day)
        dayofyear = (current_date - start_of_year).days
        if date.month == 2 and date.day == 29:           dayofyear = 58 # If 29 Feb return 28 Feb dayofyear value
        elif date.month > 2 and is_leap_year(date.year): dayofyear -= 1 # If leap year and date is after Feb 28, subtract one day
        return dayofyear
    
    extracted_data = [ds.where(ds[time_dim] == to_climatological_day(date), drop=True) for date in dates_list]
    result = xr.concat(extracted_data, dim=time_dim)
    result = result.assign_coords({time_dim: dates_list})
    return result

def filter_box(ds, lon_min, lon_max, lat_min, lat_max, lon_var='lon',lat_var='lat', output=''):
    """
    Filters an xarray dataset based on a bounding box of latitude and longitude.
    Parameters:
        ds (xarray.Dataset): The input dataset with 2D lat and lon coordinates.
        lon_min,... (float): coordinates limits the bounding box.
    Returns: The filtered xarray.Dataset.
    """
    # Create a mask for the bounding box
    mask = ((ds[lon_var] >= lon_min) & (ds[lon_var] <= lon_max) & (ds[lat_var] >= lat_min) & (ds[lat_var] <= lat_max))
    if output == 'mask': return  mask.astype(int)#.broadcast_like(ds)
    else: return ds.where(mask, drop=True)  # Apply the mask to the dataset and drop values outside the range

def intensity(SST,pc90,clim):
    """
    Computes the temperature anomaly by subtracting the climatological mean SST (clim) from the observed SST.
    The marine heatwave (MHW) intensity by comparing the observed SST to the 90th percentile of the climatology (pc90), masking values below the threshold, considering also the previous days of the target_date.
    """
    anomaly=SST-clim
    anomaly[np.isnan(anomaly)]=0   
    MHW=np.zeros(SST.shape)
    for i in range(SST.shape[1]):
        for j in range(SST.shape[2]):
            MHW[:,i,j]=wns(SST[:,i,j],pc90[:,i,j])[1]
    return anomaly, MHW


# Filtering and processing
dataset_cms  = cms_rawdataset.copy()
dataset_clim = clim_rawdataset.copy()
if 'depth' in dataset_cms.dims and dataset_cms.sizes['depth'] == 1: dataset_cms = dataset_cms.squeeze(dim='depth')

target_date = date_picker.value
dates_list  = [target_date-delta_days + timedelta(days=i) for i in range(ndays_mhw)]
print('Filtering by target date - %s days [%s to %s]...'%(ndays_mhw,dates_list[0],dates_list[-1]))
datestr = '%s%s%s'%(str(target_date.year),str(target_date.month).zfill(2),str(target_date.day).zfill(2))
dataset_cms  = dataset_cms.sel({time_dim:slice(dates_list[0], dates_list[-1])}) # Date filter
dataset_clim = extract_clim_date_range(dataset_clim, dates_list)

print('Filtering by region of interest [Lons -> %s to %s; Lats -> %s to %s]...'%(lonmin_input.value,lonmax_input.value,latmin_input.value,latmax_input.value))
dataset_cms  = filter_box(dataset_cms,lonmin_input.value,lonmax_input.value,latmin_input.value,latmax_input.value,lat_var='latitude',lon_var='longitude')
dataset_clim = filter_box(dataset_clim,lonmin_input.value,lonmax_input.value,latmin_input.value,latmax_input.value,lat_var='lat',lon_var='lon')

print('Computing Anomaly and Marine HeatWaves (MHW)...')
t0=time.time()
anomaly, MHW = intensity(dataset_cms[varname].values,dataset_clim.pc90.values,dataset_clim.clim.values)
print(f"\tDone ({time.time() - t0:.1f}s).")

output_file = os.path.join(out_dir,"MHWmap_CMEMS_%s_%s_Lons[%sto%s]Lats[%sto%s].nc"%(prod,datestr,lonmin_input.value,lonmax_input.value,latmin_input.value,latmax_input.value))
# output_file = os.path.join(out_dir,"t.nc")

print('Saving the results in a .nc file...')
dataset_anomaly = xr.Dataset({"anomaly": (["lat", "lon"], anomaly[-1,:,:]),"MHW": (["lat", "lon"], MHW[-1,:,:]),},
                             coords={"time": '%s-%s-%sT00:00:00'%(target_date.year,target_date.month,target_date.day),
                                     "lat": dataset_clim.lat.values, "lon": dataset_clim.lon.values, },)
dataset_anomaly.to_netcdf(output_file)
print(f"\tNetCDF file saved to {output_file}")


# ### Map plot
# - SST anomaly with MHW contours
# - matplotlib + cartopy

os.listdir(out_dir)


figtitle = 'CMEMS %s Satellite Observations\nSurface Temperature Anomaly and Marine Heat Waves\n%s'%(prod,target_date)
print('Plotting...')
projection = ccrs.PlateCarree()
fig, axes  = plt.subplots(nrows=1,ncols=1,figsize=(10,6),subplot_kw={'projection': projection} )
fig.subplots_adjust(bottom=0.02, top=0.92, left=0.02, right=0.87, wspace=0.05, hspace=0.05)
axes._autoscaleXon = axes._autoscaleYon = False
axes.set_title(figtitle)

map1=axes.pcolormesh(dataset_anomaly.lon,dataset_anomaly.lat,dataset_anomaly.anomaly,vmin=-6,vmax=6,cmap='seismic',transform=projection)
axes.contour(dataset_anomaly.lon,dataset_anomaly.lat,dataset_anomaly.MHW,levels=[0.5],colors=['maroon'],linewidths=[1],transform=projection)
axes.coastlines()
axes.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='black', facecolor='lightgrey'))
gl = axes.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linestyle='--', linewidth=0.2)
gl.xlabel_style = gl.ylabel_style = {'size': 10, 'color': 'black'}
gl.top_labels   = gl.right_labels = False  # Disable top and right labels

# Dynamically adjust the colorbar height to match the plot
bbox = axes.get_position()
cb_ax1 = fig.add_axes([bbox.x1 + 0.02, bbox.y0, 0.02, bbox.height]) # [right, bottom, width, height]
cbar1 = fig.colorbar(map1, cax=cb_ax1, orientation='vertical',extend='both')
cbar1.set_label("Temperature Anomaly ($^oC$)",rotation=270, labelpad=20)

fig.savefig(output_file.replace('.nc','.png'),dpi=300)
print(f"\tPNG figure saved at '{output_file.replace('.nc','.png')}'\n")




