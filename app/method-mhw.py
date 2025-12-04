#!/usr/bin/env python3
"""
MHW method implementation using Copernicus Marine Service data

    Examples of usage:
    1) Generate MHW map for Mediterranean full region 
        python method-mhw.py --outputs_path "./outputs" --data_source '["SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_a_V2"]' --id_output_type "mhw_map_anomalies" --working_domain '{ "box": [[ -18.125,  30.125, 36.325,  46.025 ]] }' --start_time "2025-10-08" --climatology "1987-2021" --data_path "./inputs/"

    2) Generate MHW map for Mediterranean sub region 
        python method-mhw.py --outputs_path "./outputs" --data_source '["SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_a_V2"]' --id_output_type "mhw_map_anomalies" --working_domain '{ "box": [[-4.99, 34, 1, 42]] }' --start_time "2025-10-08" --climatology "1987-2021" --data_path "./inputs/"

    3) Generate MHW categories map for Mediterranean sub region
        python method-mhw.py --outputs_path "./outputs" --data_source '["SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_a_V2"]' --id_output_type "mhw_map_categories" --working_domain '{ "box": [[-4.99, 34, 1, 42]] }' --start_time "2025-10-08" --climatology "1987-2021" --data_path "./inputs/"

    4) Generate MHW timeseries for Mediterranean sub region 
        python method-mhw.py --outputs_path "./outputs" --data_source '["SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_a_V2"]' --id_output_type "mhw_timeseries" --working_domain '{ "box": [[-4.99, 34, 1, 42]] }' --start_time "2025-06-01" --end_time "2025-09-01" --climatology "1987-2021" --data_path "./inputs/"
"""

from __future__ import annotations
import os
import time
import argparse
from datetime import datetime, timedelta, date
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ast
import warnings

# local utilities (same as notebook)
from conf.functions import (
    extract_clim_date_range,
    extract_clim_ref,
    standardize_dim_names,
    filter_box,
    classify_mhw_3D_numpy,
    classify_mhw_1D_numpy,
    extract_TS,
    plotly_fill_between_segments,
)
from conf.config import CMEMS_datasets_options, category_info, regions_strings

import copernicusmarine

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available, will use matplotlib fallback for timeseries")

warnings.filterwarnings("ignore") # Ignore all warnings

def _create_fig(title1='', title2='', title_fsize=10, projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7), subplot_kw={'projection': projection})
    fig.subplots_adjust(bottom=0.05, top=0.82, left=0.05, right=0.85)
    ax._autoscaleXon = ax._autoscaleYon = False
    ax.set_title(title1, loc='left', fontsize=title_fsize-1, fontweight='bold', pad=20)
    ax.set_title(title2, loc='right', fontsize=title_fsize-1, fontweight='bold', pad=20)
    ax.coastlines()
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='black', facecolor='lightgrey'))
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linestyle='--', linewidth=0.2)
    gl.xlabel_style = gl.ylabel_style = {'size': 10, 'color': 'black'}
    gl.top_labels = gl.right_labels = False
    return fig, ax

def parse_args():
    p = argparse.ArgumentParser(description="MHW method - Maps and Timeseries")
    p.add_argument("--data_path", type=str, required=True, help="Path to input data")
    p.add_argument("--outputs_path", type=str, required=True, help="Path to output data")
    p.add_argument("--data_source", type=str, required=True, help="Dataset ID (key from conf.config)")
    p.add_argument("--id_output_type", type=str, required=True,
                   choices=["mhw_map_anomalies", "mhw_timeseries", "mhw_map_categories"],
                   help="Output type")
    p.add_argument("--working_domain", type=str, required=True,
                   help="Working domain as JSON string. Format: "
                        "{ \"box\": [[lon_min, lat_min, lon_max, lat_max]], \"depth_layers\": [[depth_min, depth_max]] }")
    p.add_argument("--start_time", type=str, required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end_time", type=str, required=False, help="End date (YYYY-MM-DD) - Required for timeseries")
    p.add_argument("--climatology", default="1993-2022", type=str, required=True, help="Climatology key")
    p.add_argument("--ndays", type=int, default=12, required=False, help="Window length for MHW detection (maps only)")
    p.add_argument("--boundary_method", type=str, default="box", choices=["box", "region"], help="Boundary method (timeseries only)")
    p.add_argument("--region_name", type=str, required=False, help="Region name (if using region boundary method)")
    return p.parse_args()

def main():
    args = parse_args()

    # get inputs
    data_path = args.data_path
    outputs_dir = args.outputs_path
    dataset_id = ast.literal_eval(args.data_source)[0]
    id_output_type = args.id_output_type
    print(f"Data path: {data_path}")
    print(f"Outputs will be saved to: {outputs_dir}")
    print(f"Using data source: {dataset_id}")
    print(f"Output type: {id_output_type}")

    # Validate arguments based on output type
    if id_output_type == "mhw_timeseries":
        if not args.end_time:
            raise SystemExit("ERROR: --end_time is required for mhw_timeseries output type")

    climatology = args.climatology
    ndays_mhw = args.ndays # number of days to consider in identifying and masking the MHWs (maps only)

    # parse times
    start_time = datetime.strptime(args.start_time, "%Y-%m-%d").date()
    end_time = datetime.strptime(args.end_time, "%Y-%m-%d").date() if args.end_time else None

    # parse working_domain
    wd = ast.literal_eval(args.working_domain)
    lonmin_input, latmin_input, lonmax_input, latmax_input = wd['box'][0]
    if lonmin_input >= lonmax_input or latmin_input >= latmax_input:
        raise SystemExit("working_domain.box coordinates invalid: min must be < max")
    print(f"Using working domain box: Lons [{lonmin_input} to {lonmax_input}] Lats [{latmin_input} to {latmax_input}]")

    # validate dataset
    if dataset_id not in CMEMS_datasets_options:
        raise SystemExit(f"Dataset id '{dataset_id}' not found in conf.config CMEMS_datasets_options")
    meta = CMEMS_datasets_options[dataset_id]
    varname = meta.get("varname")
    prod = meta.get("prod_type")
    if prod == 'MFS': 
        prod2plot = 'Mediterranean Forecasting System (MFS)' if id_output_type == "mhw_timeseries" else 'Mediterranean Forecasting System'
    elif 'L4' in prod:
        prod2plot = prod + ' Satellite Observations'

    # check config has grid/area entries (needed for timeseries)
    grid_file_rel = meta.get("grid_file")
    area_file_rel = meta.get("area_file")
    if id_output_type == "mhw_timeseries" and (not grid_file_rel or not area_file_rel):
        raise SystemExit("ERROR: config missing 'grid_file' or 'area_file' for dataset (required for timeseries)")

    # resolve chosen climatology entry and file path
    try:
        clim_meta = meta.get("clim_file")
        clim_entry = clim_meta[climatology]
        clim_file_rel = clim_entry.get("file")
    except Exception:
        raise SystemExit(f"ERROR: climatology '{climatology}' not found in config for dataset")
    clim_path = os.path.join(data_path, str(clim_file_rel))

    # check files exist locally
    missing = []
    if not os.path.isfile(clim_path):
        missing.append(clim_path)
    
    # Only check area files for timeseries
    if id_output_type == "mhw_timeseries":
        grid_path = os.path.join(data_path, str(grid_file_rel))
        area_path = os.path.join(data_path, str(area_file_rel))
        if not os.path.isfile(grid_path):
            missing.append(grid_path)
        if not os.path.isfile(area_path):
            missing.append(area_path)

    if missing:
        print("ERROR: required files missing in data_path:")
        for p in missing:
            print("  -", p)
        raise SystemExit(1)

    print(f"Selected climatology: {climatology} -> {clim_path}")
    
    # Load climatology (and area files for timeseries)
    try:
        clim_rawdataset = xr.open_dataset(clim_path)
        if id_output_type == "mhw_timeseries":
            print("\nOpening cell area files...")
            area_rawdataset = xr.open_dataset(grid_path)
            area_clim_rawdataset = xr.open_dataset(area_path)
    except Exception as e:
        raise SystemExit(f"ERROR opening files: {e}")

    clim_ref = extract_clim_ref(clim_path)
    print("Opening climatology file... from", clim_ref)

    # query CMEMS
    print("Querying copernicusmarine api...")
    params = {"credentials_file": ".bc2026_copernicusmarine-credentials",
              "dataset_id": dataset_id, "variables": [varname], "maximum_depth": 1.5}
    if prod == 'MFS': 
        params["minimum_longitude"] = -6  # this limit is set because of the grid of the available region masks
    
    t0 = time.time()
    try:
        cmems_rawdataset = copernicusmarine.open_dataset(**params)
        if cmems_rawdataset is None:
            raise SystemExit(f"ERROR: copernicusmarine returned no dataset for id '{dataset_id}'")  
    except Exception as e:
        raise SystemExit(f"ERROR opening CMEMS dataset '{dataset_id}': {e}")
    
    print(f"\tElapsed time: {time.time() - t0:.1f}s")

    # determine available date range
    time_dim = 'time'
    import pandas as pd
    times = pd.to_datetime(cmems_rawdataset[time_dim].values)
    date_min = times.min().date()
    date_max = times.max().date()
    print(f'\nDataset limits -> from {date_min} to {date_max}')

    # Route to appropriate processing based on output type
    if id_output_type in ["mhw_map_anomalies", "mhw_map_categories"]:
        success = process_maps(args, cmems_rawdataset, clim_rawdataset, meta, outputs_dir, 
                              start_time, lonmin_input, latmin_input, lonmax_input, latmax_input,
                              date_min, date_max, varname, prod, prod2plot, clim_ref, 
                              time_dim, ndays_mhw, id_output_type)
    elif id_output_type == "mhw_timeseries":
        success = process_timeseries(args, cmems_rawdataset, clim_rawdataset, area_rawdataset, area_clim_rawdataset,
                                   meta, outputs_dir, start_time, end_time, lonmin_input, latmin_input, 
                                   lonmax_input, latmax_input, date_min, date_max, varname, prod, prod2plot, 
                                   clim_ref, time_dim)
    else:
        print(f"ERROR: Unknown output type '{id_output_type}'")
        success = False

    if not success:
        raise SystemExit(1)

def process_maps(args, cmems_rawdataset, clim_rawdataset, meta, outputs_dir, 
                start_time, lonmin_input, latmin_input, lonmax_input, latmax_input,
                date_min, date_max, varname, prod, prod2plot, clim_ref, 
                time_dim, ndays_mhw, id_output_type):
    """Process map-based outputs (anomalies or categories)"""
    
    date_min_delta = date_min + timedelta(ndays_mhw)
    target_date = start_time
    if target_date < date_min_delta or target_date > date_max:
        raise SystemExit(f"target date {target_date} out of available range [{date_min_delta} - {date_max}]")

    # prepare datasets
    dataset_cmems = cmems_rawdataset.copy()
    dataset_clim = clim_rawdataset.copy()

    if 'depth' in dataset_cmems.dims and dataset_cmems.sizes.get('depth', 0) == 1:
        dataset_cmems = dataset_cmems.squeeze(dim='depth')

    dates_list = np.arange(np.datetime64(target_date) - np.timedelta64(ndays_mhw, 'D'),
                           np.datetime64(target_date) + np.timedelta64(1, 'D'),
                           np.timedelta64(1, 'D'))
    datestr = f"{target_date.year}{str(target_date.month).zfill(2)}{str(target_date.day).zfill(2)}"

    print(f"Filtering by target date - {ndays_mhw} days [{dates_list[0]} to {dates_list[-1]}]...")
    dataset_cmems = dataset_cmems.sel({time_dim: slice(dates_list[0], dates_list[-1])})
    dataset_clim = extract_clim_date_range(dataset_clim, dates_list)
    dataset_cmems = standardize_dim_names(dataset_cmems)
    dataset_clim = standardize_dim_names(dataset_clim)

    print("Filtering by region of interest ...")
    dataset_cmems = filter_box(dataset_cmems,lonmin_input,lonmax_input,latmin_input,latmax_input,lat_var='latitude',lon_var='longitude')
    dataset_clim  = filter_box(dataset_clim,lonmin_input,lonmax_input,latmin_input,latmax_input,lat_var='latitude',lon_var='longitude')
   
    print("Computing Anomaly and Marine HeatWaves (MHW)...")
    t0 = time.time()
    anomaly, MHW, categories = classify_mhw_3D_numpy(dataset_cmems[varname].values,
                                                    dataset_clim.pc90.values,
                                                    dataset_clim.clim.values)
    print(f"\tDone ({time.time() - t0:.1f}s).")

    # assemble output dataset
    dataset_anomaly = xr.Dataset({
        "anomaly": (["lat", "lon"], anomaly[-1, :, :]),
        "MHW": (["lat", "lon"], MHW[-1, :, :]),
        "MHW_cats": (["lat", "lon"], categories[-1, :, :]),
    }, coords={
        "time": f"{target_date.isoformat()}T00:00:00",
        "lat": dataset_cmems.latitude.values,
        "lon": dataset_cmems.longitude.values,
    })

    os.makedirs(outputs_dir, exist_ok=True)
    output_file = os.path.join(outputs_dir, f"{id_output_type}_{datestr}_{prod}.nc")
    
    # plotting and saving
    success = False
    
    try:
        if (id_output_type == "mhw_map_anomalies"):
            print("Plotting SST Anomaly map...")
            title1 = f"Copernicus Marine Service {prod2plot}\nSea Surface Temperature (SST) Anomaly and Marine Heat Waves (MHW)"
            title2 = f"Date: {target_date}\nClimatology: {clim_ref}"
            fig, ax = _create_fig(title1 + "\n" + title2)
            map1 = ax.pcolormesh(dataset_anomaly.lon, dataset_anomaly.lat, dataset_anomaly.anomaly,
                                shading='nearest', vmin=-6, vmax=6, cmap='RdBu_r', transform=ax.projection)
            ax.contour(dataset_anomaly.lon, dataset_anomaly.lat, dataset_anomaly.MHW, levels=[0.5],
                    colors=['maroon'], linewidths=[1], transform=ax.projection)
            ax.set_extent([float(dataset_anomaly.lon.values.min()), float(dataset_anomaly.lon.values.max()),
                        float(dataset_anomaly.lat.values.min()), float(dataset_anomaly.lat.values.max())],
                        crs=ax.projection)
            bbox = ax.get_position()
            cb_ax1 = fig.add_axes([bbox.x1 + 0.01, bbox.y0, 0.015, bbox.height])
            cbar1 = fig.colorbar(map1, cax=cb_ax1, orientation='vertical', extend='both')
            cbar1.set_label("°C", rotation=0, labelpad=10)
            figpng = output_file.replace('.nc', '.png')
            fig.savefig(figpng, dpi=300)
            print(f"\tPNG figure saved at '{figpng}'")
            plt.close(fig)  # Clean up figure
            success = True
            
        elif (id_output_type == "mhw_map_categories"):
            # categories plot
            print('Plotting MHW Categories plot...')
            title1 = f"Copernicus Marine Service {prod2plot}\nSea Surface Marine Heat Waves (MHW) Categories"
            title2 = f"Date: {target_date}\nClimatology: {clim_ref}"

            fig2, ax2 = _create_fig(title1 + "\n" + title2)
            from matplotlib.colors import ListedColormap, BoundaryNorm
            levels = list(category_info.keys())
            colors = [category_info[k]["color"] for k in levels]
            cmap = ListedColormap(colors)
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
            mhw_cats = dataset_anomaly.MHW_cats.values.astype(float)
            mhw_cats[mhw_cats <= 0] = np.nan
            m = ax2.pcolormesh(dataset_anomaly.lon, dataset_anomaly.lat, mhw_cats,
                            cmap=cmap, norm=norm, shading="nearest", transform=ax2.projection, rasterized=True)
            ax2.set_extent([float(dataset_anomaly.lon.values.min()), float(dataset_anomaly.lon.values.max()),
                            float(dataset_anomaly.lat.values.min()), float(dataset_anomaly.lat.values.max())],
                        crs=ax2.projection)
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='s', color='w', label=info['name'],
                                    markerfacecolor=info['color'], markersize=8) for _, info in category_info.items()]
            ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1),
                    ncol=1, frameon=True, title='Categories:')
            fig2.savefig(output_file.replace('.nc', '_categories.png'), dpi=300)
            print(f"\tPNG figure saved at '{output_file.replace('.nc', '_categories.png')}'")
            plt.close(fig2)  # Clean up figure
            success = True
            
    except Exception as e:
        print(f"ERROR during plotting: {e}")
        success = False

    # Only save NetCDF if plotting was successful
    if success:
        print("Saving the results in a .nc file...")
        try:
            dataset_anomaly.to_netcdf(output_file)
            print(f"\tNetCDF file saved to {output_file}")
        except Exception as e:
            print(f"ERROR saving NetCDF file: {e}")
            success = False
    else:
        print("Skipping NetCDF save due to previous errors")
    
    return success

def process_timeseries(args, cmems_rawdataset, clim_rawdataset, area_rawdataset, area_clim_rawdataset,
                      meta, outputs_dir, date_start, date_end, lonmin_input, latmin_input, 
                      lonmax_input, latmax_input, date_min, date_max, varname, prod, prod2plot, 
                      clim_ref, time_dim):
    
    # MHW settings for timeseries
    ndays_min = 30
    max_years = 3
    ndays_max = 365 * max_years
    boundary_method = args.boundary_method
    region_name = args.region_name

    # Dates check
    if (date_end - date_start).days < ndays_min:
        print(f'Minimum range of {ndays_min} days not respected OR Start date is after End date.')
        print(f'Setting Start date to {ndays_min} days before End date.')
        date_start = date_end - timedelta(ndays_min)
    
    print(f'\nDate range: {date_start} to {date_end} ({(date_end - date_start).days} days)\n')
    
    if (date_end - date_start).days > ndays_max:
        print(f"WARNING: You are computing more than {ndays_max} days! Please take care of processing time.")

    # date string
    datestr = f'{date_start}_{date_end}'.replace('-', '')

    # Boundary method - simplified to use only box method for now
    filter_by_box = True
    if boundary_method == "region" and region_name:
        filter_by_box = False
        print(f"Using region: {region_name}")
    else:
        print(f'Using box filter -> Lons [{lonmin_input} to {lonmax_input}] Lats [{latmin_input} to {latmax_input}]')

    # Processing datasets
    dataset_CMEMS = cmems_rawdataset.copy()
    dataset_clim = clim_rawdataset.copy()
    cell_area = area_rawdataset.cell_area
    cell_area_clim = area_clim_rawdataset.cell_area

    if 'depth' in dataset_CMEMS.dims and dataset_CMEMS.sizes['depth'] == 1: 
        dataset_CMEMS = dataset_CMEMS.squeeze(dim='depth')

    print("This could take a while if you have selected a long time range...")
    print('Filtering...')
    print(f'\tDates filter -> {date_start} to {date_end} ({(date_end-date_start).days} days)')
    
    dates_list = np.arange(np.datetime64(date_start), np.datetime64(date_end) + np.timedelta64(1, 'D'), np.timedelta64(1, 'D'))
    dataset_CMEMS = dataset_CMEMS.sel({time_dim: slice(dates_list[0], dates_list[-1])})  # Date filter
    
    if 'L4' in prod:
        dataset_CMEMS = dataset_CMEMS.reindex({time_dim: dates_list.astype('datetime64[D]').astype('datetime64[ns]')})
    elif prod == 'MFS':
        dataset_CMEMS = dataset_CMEMS.reindex({time_dim: dates_list})  # assuring gaps filled with nan
        if np.datetime64(date.today()) in dates_list:
            print('\nFinding start date of MFS forecast through copernicusmarine api...')
            dataset_files = copernicusmarine.get(credentials_file=".bc2026_copernicusmarine-credentials",
                dataset_id=meta['dataset_id'] if 'dataset_id' in meta else args.data_source, dry_run=True)
            start_fc = sorted([f.filename.split('_')[0] for f in dataset_files.files if '_fc-' in f.filename])[0]
            start_fc = np.datetime64(datetime.strptime(start_fc, "%Y%m%d"), "ns") 
            print(f'\tForecast start in {start_fc}')
    
    dataset_clim = extract_clim_date_range(dataset_clim, dates_list)

    # Apply region/box filtering
    if filter_by_box: 
        print(f'\tBox filter   -> Lons [{lonmin_input} to {lonmax_input}] Lats [{latmin_input} to {latmax_input}]')
        region_mask = region_mask_clim = filter_box(dataset_CMEMS, lonmin_input, lonmax_input, latmin_input, latmax_input,
                                                   lat_var='latitude', lon_var='longitude', output='mask')
        if prod == 'MFS':
            region_mask_clim = filter_box(dataset_clim, lonmin_input, lonmax_input, latmin_input, latmax_input,
                                        lat_var='lat', lon_var='lon', output='mask')
    else:
        print(f'\tRegion -> {region_name}')
        reg_path = os.path.join(args.data_path, 'region')  # Adjust path as needed
        region_folder = meta.get('region_folder', '')
        region_mask = region_mask_clim = xr.open_dataset(os.path.join(reg_path, region_folder, region_name+'_region.nc')).index_region
        if prod == 'MFS':
            region_mask_clim = xr.open_dataset(os.path.join(reg_path, 'MFS', region_name+'_region.nc')).index_region

    # Standardize coordinates (lat,lon) dimensions names
    dataset_CMEMS = standardize_dim_names(dataset_CMEMS)
    region_mask = standardize_dim_names(region_mask)
    cell_area = standardize_dim_names(cell_area)
    dataset_clim = standardize_dim_names(dataset_clim)
    region_mask_clim = standardize_dim_names(region_mask_clim)
    cell_area_clim = standardize_dim_names(cell_area_clim)  

    # extract_TS for each variable of the datasets
    print('Applying masks and computing area-average...')
    t0 = time.time()
    averaged_vars = {}
    print('\tCMEMS data...')    
    for var_name, var_data in dataset_CMEMS.data_vars.items():
        if var_data.min() > 100: var_data = var_data - 273.15  # Kelvin units check
        averaged_vars[var_name] = xr.DataArray(extract_TS(var_data, region_mask, cell_area).filled(np.nan), 
                                                dims=[time_dim], coords={time_dim: var_data[time_dim]}, name=var_name)    
    print('\tCLIM data...')
    for var_name, var_data in dataset_clim.data_vars.items():
        if var_data.min() > 100: var_data = var_data - 273.15  # Kelvin units check
        averaged_vars[var_name] = xr.DataArray(extract_TS(var_data, region_mask_clim, cell_area_clim).filled(np.nan), 
                                                dims=[time_dim], coords={time_dim: var_data[time_dim]}, name=var_name)
    print(f'\tElapsed time: {time.time() - t0:.1f}s')
    processed_dataset = xr.Dataset(averaged_vars)  # Combine all variables into a single dataset

    print('Detecting Marine HeatWaves (MHW)...')
    t0 = time.time()
    MHW, categories, category_thresholds = classify_mhw_1D_numpy(processed_dataset[varname].values, 
                                                                processed_dataset.pc90.values, 
                                                                processed_dataset.clim.values,
                                                                return_thresholds=True)
    processed_dataset['MHW'] = xr.DataArray(MHW, dims=processed_dataset[varname].dims, coords=processed_dataset[varname].coords)
    processed_dataset['cats'] = xr.DataArray(categories, dims=processed_dataset[varname].dims, coords=processed_dataset[varname].coords)
    cats = list(category_thresholds.keys())  # [1, 2, 3, 4]
    stacked_thresholds = np.stack([category_thresholds[c] for c in cats], axis=0)
    processed_dataset['cats_thrs'] = xr.DataArray(stacked_thresholds, dims=('category',) + processed_dataset[varname].dims, 
                                                  coords={'category': cats, **processed_dataset[varname].coords})

    print(f'\tElapsed time: {time.time() - t0:.1f}s')

    # Simplified region string for filenames and detailed for titles
    if filter_by_box:
        region_str_file = "box"
        region_str_title = f"Box: [{lonmin_input}°, {latmin_input}°] to [{lonmax_input}°, {latmax_input}°]"
    else:
        region_str_file = region_name if region_name else "region"
        region_str_title = regions_strings.get(region_name, region_name) if region_name else "Custom Region"

    # Create outputs directory
    os.makedirs(outputs_dir, exist_ok=True)

    # PLOTTING
    success = False
    try:
        # Extract data
        x = processed_dataset[time_dim].values
        y = processed_dataset[varname].values
        pc90 = processed_dataset.pc90.values
        clim = processed_dataset.clim.values
        y_all = np.concatenate([y, clim])

        if PLOTLY_AVAILABLE:
            # Create plotly figure (original code)
            fig = go.Figure()
            # === Title === (include coordinates)
            title = f"<b>Copernicus Marine Service {prod2plot}<br>Sea Surface Temperature (SST) and Marine Heat Waves (MHW) events<br>{region_str_title}</b>"
            fig.add_annotation(dict(text=title, xref="paper", yref="paper", x=0, y=1.15, showarrow=False, align="left", font=dict(size=14)))

            # === Categories Filling ===
            present_categories = np.unique(categories)
            present_categories = present_categories[present_categories > 0].tolist()  # exclude non-events
            if 4 not in present_categories: 
                if len(present_categories) == 0: 
                    present_categories.append(1)
                else: 
                    present_categories.append(present_categories[-1]+1)
            
            for cat in present_categories:
                y_all = np.concatenate([y_all, category_thresholds[cat]])
                info = category_info[cat]
                cat_mask = categories >= cat 
                if cat_mask.any():
                    plotly_fill_between_segments(fig, x, category_thresholds[cat], y, cat_mask, info['color'],
                                               label=info['name'] + ' MHW', legrank=4)
                fig.add_trace(go.Scatter(x=x, y=category_thresholds[cat], mode='lines', name=info['name'],
                                       line=dict(color=info['color'], width=1.5, dash='dash'),
                                       hovertemplate='Date: %{x|%d %b}<br>Value: %{y:.2f}°C<extra></extra>',
                                       legendrank=4, legend='legend2'))
            
            # === Climatology ===
            fig.add_trace(go.Scatter(x=x, y=clim, mode='lines', name=f"Climatology ({clim_ref})",
                line=dict(color='black', width=1), hovertemplate='Date: %{x|%d %b}<br>Value: %{y:.2f}°C<extra></extra>',
                legendrank=2, legend='legend'))
            
            # === Daily SST ===
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name="Daily SST",
                line=dict(color='darkgrey', width=2.5), hovertemplate='Date: %{x}<br>Temp: %{y:.2f}°C<extra></extra>',
                legendrank=2, legend='legend'))
            
            # === Forecast ===
            if 'start_fc' in locals() and start_fc in x:
                forecast = processed_dataset.sel({time_dim: slice(start_fc, None)})
                fig.add_trace(go.Scatter(x=forecast[time_dim].values, y=forecast[varname].values,
                    mode='lines', name="Forecast", line=dict(color='steelblue', width=3),
                    hovertemplate='Date: %{x}<br>Temp: %{y:.2f}°C<extra></extra>', legendrank=2, legend='legend'))
                fig.add_vline(x=start_fc.astype(str), line_dash="dash", line_color="steelblue", opacity=0.5, line_width=1)
                
            # === Layout ===
            y_valid = y_all[np.isfinite(y_all)]
            y_min, y_max = sorted([np.min(y_valid), np.max(y_valid)])
            buffer_temp = 1
            y_min -= buffer_temp; y_max += buffer_temp

            x_dt = np.array(x).astype('datetime64[ns]').astype('datetime64[s]').astype(object)
            x_min = min(x_dt)
            x_max = max(x_dt)
            buffer_days = 1
            x_range = [x_min - timedelta(days=buffer_days), x_max + timedelta(days=buffer_days)]

            fig.update_layout(
                margin=dict(t=80, b=100, l=60, r=60),
                height=500, width=1000,
                template='plotly_white', modebar_remove=['resetScale2d'],
                yaxis=dict(title="Degrees Celsius [°C]", showline=True, linewidth=1, linecolor='black',
                            range=[y_min, y_max], minallowed=y_min, maxallowed=y_max, autorange=False),
                xaxis=dict(title=None, showline=True, linewidth=1, linecolor='black',
                            range=[x_min, x_max], minallowed=x_range[0], maxallowed=x_range[1]),
                legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center", 
                           title=dict(text='<b>Temperature:<b>', font=dict(size=11))),
                legend2=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", 
                            title=dict(text='<b>MHW Categories<br>and Thresholds:<b>', font=dict(size=11))),
            )

            # Set output file - SIMPLIFIED NAME
            output_file = os.path.join(outputs_dir, f"mhw_timeseries_{datestr}.nc")
            fig.write_html(output_file.replace('.nc', '.html'), include_plotlyjs='inline')
            print(f"\tHTML figure saved at '{output_file.replace('.nc', '.html')}'")
            
        else:
            # Matplotlib fallback (include coordinates in title)
            print("Using matplotlib fallback...")
            fig_plt, ax = plt.subplots(figsize=(12, 8))
            ax.plot(x, clim, 'k-', linewidth=1, label=f'Climatology ({clim_ref})')
            ax.plot(x, y, 'darkgrey', linewidth=2, label='Daily SST')
            ax.plot(x, pc90, 'orange', linewidth=1, linestyle='--', label='90th Percentile')
            
            # Fill MHW periods by category
            for cat in sorted(category_info.keys()):
                if cat in category_thresholds:
                    cat_mask = categories >= cat
                    if cat_mask.any():
                        info = category_info[cat]
                        thresh_line = category_thresholds[cat]
                        ax.fill_between(x, thresh_line, y, 
                                       where=(cat_mask & (y >= thresh_line)), 
                                       color=info['color'], alpha=0.7, label=f'{info["name"]} MHW')
            
            ax.set_ylabel('Temperature (°C)')
            ax.set_title(f'Copernicus Marine Service {prod2plot}\n'
                        f'SST and Marine Heat Waves - {region_str_title}\n'
                        f'Period: {date_start} to {date_end} | Climatology: {clim_ref}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_file = os.path.join(outputs_dir, f"mhw_timeseries_{datestr}.nc")
            # output_file = os.path.join(outputs_dir, f"MHWtimeseries_cmems_{prod}_{datestr}_{region_str}.nc").replace(' ', '')
            fig_plt.savefig(output_file.replace('.nc', '.png'), dpi=200, bbox_inches='tight')
            print(f"\tMatplotlib PNG saved: {output_file.replace('.nc', '.png')}")
            plt.close(fig_plt)

        success = True
    
    except Exception as e:
        print(f"ERROR during plotting: {e}")
        success = False

    # Save NetCDF if successful
    if success:
        print("Saving the results in a .nc file...")
        try:
            processed_dataset.to_netcdf(output_file)
            print(f"\tNetCDF file saved to {output_file}")
        except Exception as e:
            print(f"ERROR saving NetCDF file: {e}")
            success = False
    else:
        print("Skipping NetCDF save due to previous errors")
    
    return success

if __name__ == "__main__":
    main()