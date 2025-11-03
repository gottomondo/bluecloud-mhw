#!/usr/bin/env python3
"""
MHW method implementation using Copernicus Marine Service data

    Examples of usage:
    1) Generate MHW map for Mediterranean full region 
        python method-mhw.py --outputs_path "./out" --data_source '["SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_a_V2"]' --id_output_type "mhw_ta_map" --working_domain '{ "box": [[ -18.125,  30.125, 36.325,  46.025 ]] }' --start_time "2025-10-08" --climatology "1987-2021" --data_path "./input/"

    2) Generate MHW map for Mediterranean sub region 
        python method-mhw.py --outputs_path "./out" --data_source '["SST_MED_SST_L4_NRT_OBSERVATIONS_010_004_a_V2"]' --id_output_type "mhw_ta_map" --working_domain '{ "box": [[-4.99, 34, 1, 42]] }' --start_time "2025-10-08" --climatology "1987-2021" --data_path "./input/"

"""
from __future__ import annotations
import os
import time
import argparse
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ast

# local utilities (same as notebook)
from conf.functions import (
    extract_clim_date_range,
    extract_clim_ref,
    standardize_dim_names,
    filter_box,
    classify_mhw_3D_numpy,
)
from conf.config import CMEMS_datasets_options, category_info

import copernicusmarine

def _create_fig(title1='', title2='', title_fsize=10, projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6), subplot_kw={'projection': projection})
    fig.subplots_adjust(bottom=0.02, top=0.92, left=0.02, right=0.87)
    ax._autoscaleXon = ax._autoscaleYon = False
    ax.set_title(title1, loc='left',  fontsize=title_fsize, fontweight='bold')
    ax.set_title(title2, loc='right', fontsize=title_fsize, fontweight='bold')
    ax.coastlines()
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='black', facecolor='lightgrey'))
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linestyle='--', linewidth=0.2)
    gl.xlabel_style = gl.ylabel_style = {'size': 10, 'color': 'black'}
    gl.top_labels = gl.right_labels = False
    return fig, ax

def parse_args():
    p = argparse.ArgumentParser(description="method for mhw indicator")
    p.add_argument("--data_path", type=str, required=True, help="Path to input data")
    p.add_argument("--outputs_path", type=str, required=True, help="Path to output data")
    p.add_argument("--data_source", type=str, required=True, help="Dataset ID (key from conf.config)")
    # keep backward compatibility with --id_output_type
    p.add_argument("--id_output_type", type=str, required=False,
                   choices=["mhw_map_anomalies", "mhw_timeseries", "mhw_map_categories"],
                   help="Output type")
    p.add_argument("--working_domain", type=str, required=True,
                   help="Working domain as JSON string. Format: "
                        "{ \"box\": [[lon_min, lat_min, lon_max, lat_max]], \"depth_layers\": [[depth_min, depth_max]] }")
    p.add_argument("--start_time", type=str, required=True, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end_time", type=str, required=False, help="End date (YYYY-MM-DD)")
    p.add_argument("--climatology", default="1993-2022", type=str, required=True, help="Climatology key")
    p.add_argument("--ndays", type=int, default=12, required=False, help="Window length for MHW detection")
    return p.parse_args()

import json

def main():
    # settings 
    # clim_path = "clim"

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

    climatology = args.climatology
    ndays_mhw = args.ndays # number of days to consider in identifying and masking the MHWs

    # parse times
    start_time = datetime.strptime(args.start_time, "%Y-%m-%d").date()
    end_time = datetime.strptime(args.end_time, "%Y-%m-%d").date() if args.end_time else None

    # parse working_domain
    wd = ast.literal_eval(args.working_domain)
    lonmin_input, latmin_input, lonmax_input, latmax_input = wd['box'][0]
    if lonmin_input >= lonmax_input or latmin_input >= latmax_input:
        raise SystemExit("working_domain.box coordinates invalid: min must be < max")
    print(f"Using working domain box: Lons [{lonmin_input} to {lonmax_input}] Lats [{latmin_input} to {latmax_input}]")
    # return
    # write_inputs_file(outputs_dir, args, start_time, end_time)

    # validate dataset
    if dataset_id not in CMEMS_datasets_options:
        raise SystemExit(f"Dataset id '{dataset_id}' not found in conf.config CMEMS_datasets_options")
    meta = CMEMS_datasets_options[dataset_id]
    varname = meta.get("varname")
    prod = meta.get("prod_type")
    if prod == 'MFS': 
        prod2plot = 'Mediterranean Forecasting System'  
    elif 'L4' in prod:
        prod2plot = prod + ' Satellite Observations'

    # choose and validate climatology file (existing handling)
    # check config has grid/area entries
    grid_file_rel = meta.get("grid_file")
    area_file_rel = meta.get("area_file")
    if not grid_file_rel or not area_file_rel:
        raise SystemExit("ERROR: config missing 'grid_file' or 'area_file' for dataset")
    grid_path = os.path.join(data_path, str(grid_file_rel))
    area_path = os.path.join(data_path, str(area_file_rel))
    # check files exist locally
    missing = []
    if not os.path.isfile(grid_path):
        missing.append(grid_path)
    if not os.path.isfile(area_path):
        missing.append(area_path)

    # resolve chosen climatology entry and file path
    try:
        clim_meta = meta.get("clim_file")
        clim_entry = clim_meta[climatology]
        clim_file_rel = clim_entry.get("file")
    except Exception:
        raise SystemExit(f"ERROR: climatology '{climatology}' not found in config for dataset")
    clim_path = os.path.join(data_path, str(clim_file_rel))
    if not os.path.isfile(clim_path):
        missing.append(clim_path)

    if missing:
        print("ERROR: required files missing in data_path:")
        for p in missing:
            print("  -", p)
        print("Run the downloader to fetch grid/area/climatology for this dataset, then retry.")
        raise SystemExit(1)

    print(f"Selected climatology: {climatology} -> {clim_path}")
    try:
        clim_rawdataset = xr.open_dataset(clim_path)
    except Exception as e:
        raise SystemExit(f"ERROR opening climatology file '{clim_path}': {e}")
    clim_ref = extract_clim_ref(clim_path)
    print("Opening climatology file... from", clim_ref)
    # print(clim_rawdataset)
    # return
    # query CMEMS
    print("Querying copernicusmarine api...")
    params = {"credentials_file": ".bc2026_copernicusmarine-credentials",
              "dataset_id": dataset_id, "variables": [varname], "maximum_depth": 1.5}
    t0 = time.time()
    try:
        cmems_rawdataset = copernicusmarine.open_dataset(**params)
        if cmems_rawdataset is None:
            raise SystemExit(f"ERROR: copernicusmarine returned no dataset for id '{dataset_id}'")  
    except Exception as e:
        raise SystemExit(f"ERROR opening CMEMS dataset '{dataset_id}': {e}")
    
    print(f"\tElapsed time: {time.time() - t0:.1f}s")

    print("Opened CMEMS dataset:")
    print(copernicusmarine, cmems_rawdataset)
    # return

    # determine available date range and target date
    time_dim = 'time'

    import pandas as pd
    times = pd.to_datetime(cmems_rawdataset[time_dim].values)
    date_min = times.min().date()
    date_max = times.max().date()

    date_min_delta = date_min + timedelta(ndays_mhw)

    # choose target date: prefer provided end_time, otherwise latest available
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
    output_file = os.path.join(outputs_dir,
                               f"MHWmap_cmems_{prod}_{datestr}_Lons[{lonmin_input}to{lonmax_input}]_Lats[{latmin_input}to{latmax_input}].nc")
    # return
    # plotting and saving

    if (id_output_type != "mhw_map_anomalies"):
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
    elif (id_output_type == "mhw_map_categories"):
        # categories plot
        print("Plotting MHW Categories plot...")
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
    elif (id_output_type == "mhw_timeseries"):
        print("not implemented yet")
        pass

    print("Saving the results in a .nc file...")
    dataset_anomaly.to_netcdf(output_file)
    print(f"\tNetCDF file saved to {output_file}")

if __name__ == "__main__":
    main()
