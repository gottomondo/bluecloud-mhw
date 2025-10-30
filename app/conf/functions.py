## imported pkgs
import re
import numpy as np

## FUNCTIONS
def show_alert(message, alert_type="info"):
    """Display an alert message in a Jupyter Notebook.
    Args:message (str): The message to display.
        alert_type (str): The type of alert (info, success, warning, danger)."""
    colors = {"info": "#d9edf7","success": "#dff0d8","warning": "#fcf8e3","danger": "#f2dede"}
    color = colors.get(alert_type, "#d9edf7")
    display(HTML(f"""
    <div style="
        padding: 10px;
        margin: 5px 0;
        border: 1px solid transparent;
        border-radius: 4px;
        background-color: {color};
        font-size: 16px;">
        {message}
    </div>
    """))

# # Example usage
# show_alert("This is an info alert!", "info")
# show_alert("This is a success alert!", "success")
# show_alert("This is a warning alert!", "warning")
# show_alert("This is a danger alert!", "danger")


def extract_clim_ref(fname: str) -> str:
    """
    Extract a climatology reference string like '1987 to 2021' from a filename.
    Falls back to the last '_' chunk if no year pair is found.
    """
    years = re.findall(r"\d{4}", fname)
    if len(years) >= 2:
        return f"{years[0]} to {years[1]}"
    return fname.replace('.nc', '').split('_')[-1]

# def is_leap_year(y): return (y % 3 == 0 and y % 100 != 0) or (y % 400 == 0)
# def clim_day(dt):
#     dt = dt.astype('M8[D]').astype(object)
#     doy = dt.timetuple().tm_yday - 1
#     if dt.month == 2 and dt.day == 29:         return 58  # use Feb 28 (same as before)
#     if dt.month > 2 and is_leap_year(dt.year): return doy - 1
#     return doy
# def extract_clim_date_range(ds, dates_list, time_dim="days"):
#     """Extracts specific days from a 365-day climatology dataset."""
#     offset = 1 if time_dim in ds.coords and ds[time_dim].values.min() == 1 else 0
#     clim_days = np.array([clim_day(d) + offset for d in dates_list])
#     return ds.sel({time_dim: ds[time_dim].isin(clim_days)}).assign_coords({time_dim: dates_list})
#     # return subset.sel({time_dim: clim_days}).assign_coords({time_dim: dates_list})

def is_leap_year(y):
    return (y % 4 == 0) & ((y % 100 != 0) | (y % 400 == 0))

def clim_day(dt64_arr):
    dt64_arr = np.asarray(dt64_arr, dtype='datetime64[D]')
    dates = dt64_arr.astype('O')  # convert to datetime.date
    years = np.array([d.year for d in dates])
    months = np.array([d.month for d in dates])
    days = np.array([d.day for d in dates])

    doy = np.array([d.timetuple().tm_yday for d in dates]) - 1
    leap = is_leap_year(years)

    # Feb 29 → 58 (Feb 28)
    mask_feb29 = (months == 2) & (days == 29)
    doy[mask_feb29] = 58

    # After Feb in leap years → shift -1
    mask_post_feb_leap = (months > 2) & leap
    doy[mask_post_feb_leap] -= 1

    return doy

def extract_clim_date_range(ds, dates_list, time_dim="days"):
    """Extracts values from a 365-day climatology for given datetime64[ns] dates_list."""
    offset = 1 if time_dim in ds.coords and ds[time_dim].values.min() == 1 else 0
    clim_days = clim_day(dates_list) + offset

    # Use advanced indexing to preserve order and duplicates
    indexer = [int(np.where(ds[time_dim].values == d)[0]) for d in clim_days]
    extracted = ds.isel({time_dim: indexer})
    return extracted.assign_coords({time_dim: dates_list})


def filter_box(ds, lon_min, lon_max, lat_min, lat_max, lon_var='lon',lat_var='lat', output=''):
    """Filters an xarray dataset based on a bounding box of latitude and longitude.
    Parameters:
        ds (xarray.Dataset): The input dataset with 2D lat and lon coordinates.
        lon_min,... (float): coordinates limits the bounding box.
    Returns: The filtered xarray.Dataset."""
    mask = ((ds[lon_var] >= lon_min) & (ds[lon_var] <= lon_max) & (ds[lat_var] >= lat_min) & (ds[lat_var] <= lat_max))
    if output == 'mask': return  mask.astype(int)
    else: return ds.where(mask, drop=True)  # Apply the mask to the dataset and drop values outside the range

def standardize_dim_names(ds, lat_name='latitude', lon_name='longitude', time_name='time'):
    """Standardize the dimension names of a dataset.
    Parameters:
    - ds (xarray.Dataset or xarray.DataArray): Input dataset or data array.
    - *_name (str): standardized dimension name.
    Returns: - xarray.Dataset or xarray.DataArray: Dataset with standardized dimension names."""
    rename_map = {}
    if 'latitude'    in ds.dims: rename_map['latitude']  = lat_name
    elif 'Latitude'  in ds.dims: rename_map['Latitude']  = lat_name
    elif 'lat'       in ds.dims: rename_map['lat']       = lat_name
    elif 'y'         in ds.dims: rename_map['y']         = lat_name
    if 'longitude'   in ds.dims: rename_map['longitude'] = lon_name
    elif 'Longitude' in ds.dims: rename_map['Longitude'] = lon_name
    elif 'lon'       in ds.dims: rename_map['lon']       = lon_name
    elif 'x'         in ds.dims: rename_map['x']         = lon_name
    if 'time'        in ds.dims: rename_map['time']      = time_name
    elif 'days'      in ds.dims: rename_map['days']      = time_name
    elif 'dayofyear' in ds.dims: rename_map['dayofyear'] = time_name
    ds = ds.rename(rename_map) # Rename dimensions
    if time_name in ds.dims: ds = ds.transpose(time_name, lat_name, lon_name)
    else: ds = ds.transpose(lat_name, lon_name)
    return ds

# def extract_TS(data,region,area):
#     region_3D = np.tile(region,(data.shape[0],1,1))
#     area_3D   = np.tile(area,(data.shape[0],1,1))
#     area_mask_3D = np.ma.masked_where(region_3D==0,area_3D)
#     data_mask    = np.ma.masked_where(region_3D==0,data)
#     data_mask    = np.ma.masked_where(np.isnan(data),data_mask)
#     data_mask    = np.ma.masked_where(data==0,data_mask)
#     area_mask_3D.mask = data_mask.mask # forcing nan mask of data_mask into area_mask_3D (necessary for box masks)
#     return np.nansum(data_mask*area_mask_3D,axis=(1,2))/np.nansum(area_mask_3D,axis=(1,2)) # calculates area-average

def extract_TS(data, region, area):
    data = data.values  # (time, y, x)
    region = region.values  # (y, x)
    area = area.values      # (y, x)
    region_3D = region[np.newaxis, :, :]
    area_3D = area[np.newaxis, :, :]
    mask = (region_3D == 0) | np.isnan(data) | (data == 0)
    data_masked = np.ma.array(data, mask=mask)
    area_masked = np.ma.array(np.broadcast_to(area_3D, data.shape), mask=mask)
    weighted_sum = np.nansum(data_masked * area_masked, axis=(1, 2))
    total_area = np.nansum(area_masked, axis=(1, 2))
    return weighted_sum / total_area

## Detect function
# def wns(sst,pc90): # Input: time series of SST and 90th percentile
#     ### Identify Marine Heatspikes and Heatwaves ###
#     # Based on code from https://github.com/ecjoliver/marineHeatWaves #
#     import scipy.ndimage as ndimage
#     MHS=sst>pc90 # Define spike 

#     # Filter out MHS with short duration
#     events,n_events=ndimage.label(MHS)
#     start=[]
#     end=[]
#     for ev in range(1,n_events+1):
#         event_duration = (events == ev).sum()
#         if event_duration < 5: continue
#         init=int(np.argwhere(events==ev)[0])
#         fin=int(np.argwhere(events==ev)[-1])
#         start.append(init)
#         end.append(fin)
#     # Join MHWs which have short gaps between #
#     maxgap=2
#     gaps=np.array(start[1:])-np.array(end[0:-1])-1
#     if len(gaps)>0:
#         while gaps.min()<=maxgap:
#             ev=np.where(gaps<=maxgap)[0][0]
#             end[ev]=end[ev+1]
#             del start[ev+1]
#             del end[ev+1]
#             gaps=np.array(start[1:])-np.array(end[0:-1])-1
#             if len(gaps)==0: break
#     # Save MHW array
#     n_events=len(start)
#     MHW=np.zeros((MHS.size))
#     for ev in range(n_events):
#         event_duration=end[ev]-start[ev]+1
#         init=start[ev]
#         for i in range(event_duration): MHW[init+i]=1
#     return MHS, MHW

# def intensity(SST,pc90,clim):
#     """Computes the temperature anomaly by subtracting the climatological mean SST (clim) from the observed SST.
#     The marine heatwave (MHW) intensity by comparing the observed SST to the 90th percentile of the climatology (pc90), masking values below the threshold, considering also the previous days of the target_date."""
#     anomaly=SST-clim
#     anomaly[np.isnan(anomaly)]=0   
#     MHW=np.zeros(SST.shape)
#     for i in range(SST.shape[1]):
#         for j in range(SST.shape[2]):
#             MHW[:,i,j]=wns_cats(SST[:,i,j],pc90[:,i,j])[1]
#     return anomaly, MHW

# def classify_mhw_1D_numpy(sst, pc90, clim, min_duration=5, maxgap=2, return_thresholds=False, smooth_delta=True):
#     n = len(sst)
#     MHW = np.zeros(n, dtype=bool)
#     categories = np.zeros(n, dtype=int)

#     MHS = sst > pc90
#     delta = pc90 - clim
#     if smooth_delta:
#         # from scipy.ndimage import gaussian_filter1d
#         # delta = gaussian_filter1d(delta, sigma=7, mode='nearest')
#         from scipy.ndimage import uniform_filter1d
#         delta = uniform_filter1d(delta, size=30)  # 'wrap' ensures periodicity
        
#     delta[delta == 0] = np.nan  # prevent div by zero
#     ratio = (sst - clim) / delta
#     ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

#     # Label MHS events
#     events = []
#     i = 0
#     while i < n:
#         if MHS[i]:
#             start = i
#             while i < n and MHS[i]:
#                 i += 1
#             end = i - 1
#             if (end - start + 1) >= min_duration:
#                 events.append((start, end))
#         i += 1

#     # Merge events across maxgap
#     merged = []
#     if events:
#         start, end = events[0]
#         for s, e in events[1:]:
#             if s - end -1 <= maxgap:
#                 end = e
#             else:
#                 merged.append((start, end))
#                 start, end = s, e
#         merged.append((start, end))

#     for s, e in merged:
#         for i in range(s, e + 1):
#             MHW[i] = True
#             r = ratio[i]
#             if r >= 4:
#                 categories[i] = 4
#             elif r >= 3:
#                 categories[i] = 3
#             elif r >= 2:
#                 categories[i] = 2
#             elif r >= 1:
#                 categories[i] = 1
#             # MHW[i] = categories[i] > 0

#     if return_thresholds:
#         # delta = pc90 - clim
#         # category_thresholds = {c: clim + c * delta for c in range(1, 5)}
#         category_thresholds = {c: clim + c * delta for c in range(1, 5)}
#         return MHW, categories, category_thresholds
#     else: return MHW, categories

def classify_mhw_1D_numpy(sst, pc90, clim, min_duration=5, maxgap=2, return_thresholds=False, smooth_delta=True):
    from scipy.ndimage import label
    n = len(sst)
    MHW = np.zeros(n, dtype=bool)
    categories = np.full(n, np.nan)

    MHS = sst > pc90
    delta = pc90 - clim

    if smooth_delta:
        from scipy.ndimage import uniform_filter1d
        delta = uniform_filter1d(delta, size=30)

    delta[delta == 0] = np.nan
    ratio = (sst - clim) / delta
    ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 1: Label all segments (MHS or not)
    labeled, n_labels = label(MHS)

    # Step 2: Build boolean array with gap-merged events
    gap_merged = np.zeros_like(MHS, dtype=bool)
    if n_labels > 0:
        event_bounds = []
        for ev in range(1, n_labels + 1):
            idx = np.where(labeled == ev)[0]
            event_bounds.append((idx[0], idx[-1]))

        # Merge across small gaps
        merged_bounds = []
        s, e = event_bounds[0]
        for ns, ne in event_bounds[1:]:
            if ns - e - 1 <= maxgap:
                e = ne  # merge
            else:
                merged_bounds.append((s, e))
                s, e = ns, ne
        merged_bounds.append((s, e))  # final

        # Mark merged bounds
        for s, e in merged_bounds:
            if (e - s + 1) >= min_duration:
                gap_merged[s:e+1] = True

    # Step 3: Fill outputs
    MHW[:] = gap_merged
    for i in np.where(MHW)[0]:
        r = ratio[i]
        if r >= 4:
            categories[i] = 4
        elif r >= 3:
            categories[i] = 3
        elif r >= 2:
            categories[i] = 2
        elif r >= 1:
            categories[i] = 1

    if return_thresholds:
        category_thresholds = {c: clim + c * delta for c in range(1, 5)}
        return MHW, categories, category_thresholds
    else:
        return MHW, categories
    
def classify_mhw_3D_numpy(sst, pc90, clim):
    """
    Input: sst, pc90, clim - 3D NumPy arrays (time, lat, lon)
    Output: anomaly, MHW, categories - 3D arrays (time, lat, lon)
    """
    time_len, lat_len, lon_len = sst.shape
    npoints = lat_len * lon_len

    sst_flat = sst.transpose(1, 2, 0).reshape(npoints, time_len)
    pc90_flat = pc90.transpose(1, 2, 0).reshape(npoints, time_len)
    clim_flat = clim.transpose(1, 2, 0).reshape(npoints, time_len)

    MHW_flat = np.zeros((npoints, time_len), dtype=bool)
    cats_flat = np.zeros((npoints, time_len), dtype=int)

    for i in range(npoints):
        MHW_flat[i], cats_flat[i] = classify_mhw_1D_numpy(
            sst_flat[i], pc90_flat[i], clim_flat[i]
        )

    # Reshape back to (time, lat, lon)
    MHW = MHW_flat.reshape(lat_len, lon_len, time_len).transpose(2, 0, 1)
    cats = cats_flat.reshape(lat_len, lon_len, time_len).transpose(2, 0, 1)
    anomaly = sst - clim

    return anomaly, MHW, cats

# def detect_mhw_dataset(ds_sst, ds_pc90, ds_clim):
#     """
#     Apply MHW classification to a full dataset with dimensions (time, lat, lon).
#     Returns:
#         - anomaly: (time, lat, lon)
#         - MHW: boolean (time, lat, lon)
#         - categories: int (time, lat, lon)
#     """
#     import xarray as xr

#     anomaly = ds_sst - ds_clim

#     # Use apply_ufunc with return_thresholds=False
#     MHW, categories = xr.apply_ufunc(
#         lambda s, p, c: classify_mhw_1D(s, p, c, return_thresholds=False),
#         ds_sst, ds_pc90, ds_clim,
#         input_core_dims=[['time'], ['time'], ['time']],
#         output_core_dims=[['time'], ['time']],
#         vectorize=True,
#         #dask=None,#'parallelized',
#         output_dtypes=[bool, int],
#     )
#     MHW = MHW.transpose(*anomaly.dims)
#     categories = categories.transpose(*anomaly.dims)
#     return anomaly, MHW, categories

# def classify_mhw_1D(sst, pc90, clim, min_duration=5, maxgap=2, return_thresholds=True):
#     """Optimized detection of MHS and MHWs with Hobday-style pointwise categories.
#     Parameters:
#     -----------
#     sst : 1D np.array        Sea surface temperature time series.
#     pc90 : 1D np.array        90th percentile threshold time series.
#     clim : 1D np.array        Climatology time series.
#     min_duration : int        Minimum duration to qualify as a MHW (default 5).
#     maxgap : int        Maximum gap to merge between events (default 2).
#     Returns:
#     --------
#     MHW : 1D bool array        True where MHWs (duration >=5 and gap-merged) occur.
#     categories : 1D int array  0 for non-MHW days, 1–4 for Hobday categories (per time step).
#     category_thresholds : dict of 1D float arrays    Continuous temperature thresholds for each category (1–4), for every time step.    """

#     import scipy.ndimage as ndimage

#     # Step 1: Initial mask and anomaly
#     MHS     = sst > pc90
#     delta   = pc90 - clim
#     anomaly = sst - clim
#     ratio   = anomaly / delta  # could have NaNs or inf if delta == 0

#     # Step 2: Label candidate MHS events
#     labeled, n_events = ndimage.label(MHS)
#     sizes = ndimage.sum(MHS, labeled, index=np.arange(1, n_events + 1))
#     valid_events = np.where(sizes >= min_duration)[0] + 1  # event IDs (1-based)
#     mask_valid = np.isin(labeled, valid_events)

#     # Step 3: Merge events across small gaps
#     event_mask = mask_valid.astype(int)
#     labeled_valid, _ = ndimage.label(event_mask)
#     bounds = ndimage.find_objects(labeled_valid)
#     new_events = []
#     for sl in bounds:
#         if sl is None: continue
#         i0, i1 = sl[0].start, sl[0].stop - 1
#         new_events.append((i0, i1))
#     # Merge based on maxgap
#     # start, end = zip(*new_events)
#     # start, end = list(start), list(end)
#     # gaps = np.array(start[1:]) - np.array(end[:-1]) - 1
#     if new_events:
#         start, end = zip(*new_events)
#         start, end = list(start), list(end)
#         gaps = np.array(start[1:]) - np.array(end[:-1]) - 1
#     else:
#         start, end = [], []
#         gaps = np.array([])
#     while len(gaps) > 0 and gaps.min() <= maxgap:
#         ev = np.where(gaps <= maxgap)[0][0]
#         end[ev] = end[ev + 1]
#         del start[ev + 1]
#         del end[ev + 1]
#         gaps = np.array(start[1:]) - np.array(end[:-1]) - 1

#     # Step 4: Create MHW and category arrays
#     MHW = np.zeros_like(sst, dtype=bool)
#     categories = np.full_like(sst, np.nan, dtype=float)

#     for i0, i1 in zip(start, end):
#         this_ratio = ratio[i0:i1 + 1]
#         this_ratio = np.nan_to_num(this_ratio, nan=0.0, posinf=0.0, neginf=0.0)

#         cat_local = (
#             (this_ratio >= 4).astype(int) * 4 +
#             ((this_ratio >= 3) & (this_ratio < 4)).astype(int) * 3 +
#             ((this_ratio >= 2) & (this_ratio < 3)).astype(int) * 2 +
#             ((this_ratio >= 1) & (this_ratio < 2)).astype(int) * 1
#         )

#         categories[i0:i1 + 1] = cat_local
#         MHW[i0:i1 + 1] = cat_local > 0  # ← Only mark MHW where category ≥ 1
    
#     # categories = np.zeros_like(sst, dtype=int)
#     # for i0, i1 in zip(start, end):
#     #     # MHW[i0:i1 + 1] = True
#     #     this_ratio = ratio[i0:i1 + 1]
#     #     this_ratio = np.nan_to_num(this_ratio, nan=0.0, posinf=0.0, neginf=0.0)
#     #     # Vectorized assignment based on thresholds
#     #     categories[i0:i1 + 1] = (
#     #         (this_ratio >= 4).astype(int) * 4 +
#     #         ((this_ratio >= 3) & (this_ratio < 4)).astype(int) * 3 +
#     #         ((this_ratio >= 2) & (this_ratio < 3)).astype(int) * 2 +
#     #         ((this_ratio >= 1) & (this_ratio < 2)).astype(int) * 1
#     #     )

#     # Step 5: Continuous thresholds
#     if return_thresholds: 
#         category_thresholds = {c: clim + c * delta for c in range(1, 5)}
#         return MHW, categories, category_thresholds
#     else: return MHW, categories


# === fill_between emulation ===
def plotly_fill_between_segments(fig, x, y1, y2, mask, color="orange", label=None,legrank=2):
    import plotly.graph_objects as go
    """Fill between y1 and y2 where mask is True. All segments are grouped into one legend entry."""
    x = np.asarray(x)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    mask = np.logical_and(mask, y2 > y1)
    segments, current = [],[]
    for i in range(len(mask)):
        if mask[i]: current.append(i)
        elif current:
            segments.append(current)
            current = []
    if current: segments.append(current)

    def interpolate_crossing(i0, i1):
        dy2 = y2[i1] - y2[i0]
        dy1 = y1[i1] - y1[i0]
        dx_ns = (x[i1] - x[i0]).astype('timedelta64[ns]').astype(int)
        denom = dy2 - dy1
        alpha = 0.5 if denom == 0 else np.clip((y1[i0] - y2[i0]) / denom, 0, 1)
        x_interp = x[i0] + np.timedelta64(int(dx_ns * alpha), 'ns')
        y_interp = y1[i0] + dy1 * alpha
        return x_interp, y_interp
    # Define a unique legend group
    legendgroup = label or f"group_{color}"
    first_segment = True
    for seg in segments:
        idx = np.array(seg)
        x_seg = x[idx]
        y1_seg = y1[idx]
        y2_seg = y2[idx]
        # Add interpolated edges
        if idx[0] > 0 and not mask[idx[0] - 1]:
            xi, yi = interpolate_crossing(idx[0] - 1, idx[0])
            x_seg = np.insert(x_seg, 0, xi)
            y1_seg = np.insert(y1_seg, 0, yi)
            y2_seg = np.insert(y2_seg, 0, yi)
        if idx[-1] < len(mask) - 1 and not mask[idx[-1] + 1]:
            xi, yi = interpolate_crossing(idx[-1], idx[-1] + 1)
            x_seg = np.append(x_seg, xi)
            y1_seg = np.append(y1_seg, yi)
            y2_seg = np.append(y2_seg, yi)
        # Plot lower edge (invisible)
        fig.add_trace(go.Scatter(
            x=x_seg, y=y1_seg, mode='lines', line=dict(width=0),
            hoverinfo='skip', showlegend=False, legendgroup=legendgroup,legend='legend2'
        ))
        # Plot filled top
        fig.add_trace(go.Scatter(
            x=x_seg, y=y2_seg, mode='lines', line=dict(width=0), fill='tonexty',
            fillcolor=color, hoverinfo='skip',
            showlegend=first_segment and label is not None,
            name=label if first_segment else None,
            legendgroup=legendgroup,legendrank=legrank,legend='legend2'
        ))
        first_segment = False
        
# def plotly_fill_between_segments(fig, x, y1, y2, mask, color="orange", label=False):
#     """
#     Fill between y1 and y2 where mask is True, using interpolated edges for accurate crossing.
#     No legend label shown.
#     """
#     x = np.asarray(x)
#     y1 = np.asarray(y1)
#     y2 = np.asarray(y2)
#     mask = np.logical_and(mask, y2 > y1)
#     segments = []
#     current = []
#     showlegend=False
#     if label != False: showlegend=True
    
#     for i in range(len(mask)):
#         if mask[i]: current.append(i)
#         elif current:
#             segments.append(current)
#             current = []
#     if current:segments.append(current)

#     def interpolate_crossing(i0, i1):
#         # Linearly interpolate crossing between y2 and y1
#         dy2 = y2[i1] - y2[i0]
#         dy1 = y1[i1] - y1[i0]
#         dx_ns = (x[i1] - x[i0]).astype('timedelta64[ns]').astype(int)
#         denom = (dy2 - dy1)
#         if denom == 0:  alpha = 0.5  # fallback: midpoint
#         else:
#             alpha = (y1[i0] - y2[i0]) / denom
#             alpha = np.clip(alpha, 0, 1)
#         x_interp = x[i0] + np.timedelta64(int(dx_ns * alpha), 'ns')
#         y_interp = y1[i0] + dy1 * alpha
#         return x_interp, y_interp

#     for seg in segments:
#         idx = np.array(seg)
#         x_seg = x[idx]
#         y1_seg = y1[idx]
#         y2_seg = y2[idx]
#         # Add interpolated point at start
#         if idx[0] > 0 and not mask[idx[0] - 1]:
#             xi, yi = interpolate_crossing(idx[0] - 1, idx[0])
#             x_seg  = np.insert(x_seg, 0, xi)
#             y1_seg = np.insert(y1_seg, 0, yi)
#             y2_seg = np.insert(y2_seg, 0, yi)
#         # Add interpolated point at end
#         if idx[-1] < len(mask) - 1 and not mask[idx[-1] + 1]:
#             xi, yi = interpolate_crossing(idx[-1], idx[-1] + 1)
#             x_seg = np.append(x_seg, xi)
#             y1_seg = np.append(y1_seg, yi)
#             y2_seg = np.append(y2_seg, yi)

#         # Plot lower edge
#         fig.add_trace(go.Scatter(x=x_seg, y=y1_seg, mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
#         # Plot filled top
#         fig.add_trace(go.Scatter(x=x_seg, y=y2_seg, mode='lines', line=dict(width=0), fill='tonexty', 
#                                  fillcolor=color,hoverinfo='skip',showlegend=showlegend,name=label))
        
def wrap_and_smooth(da, window=30, time_dim='days'):
    if not isinstance(da, xr.DataArray): raise TypeError("wrap_and_smooth expects a DataArray")
    pad = window // 2
    # Circular padding using slices
    da_padded = xr.concat([da.isel({time_dim: slice(-pad, None)}), da, da.isel({time_dim: slice(0, pad)})], dim=time_dim)
    # Apply centered rolling mean
    smoothed = da_padded.rolling({time_dim: window}, center=True).mean()
    # Trim to original size
    return smoothed.isel({time_dim: slice(pad, pad + da.sizes[time_dim])})