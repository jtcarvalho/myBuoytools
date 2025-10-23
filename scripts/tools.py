
import os
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter


def remove_spikes_and_smooth(data, threshold_factor=3.0, max_wave_height=25.0, apply_smoothing=True, smooth_window=5):
    """
    Remove spikes, unrealistic values, and optionally smooth time series data
    
    Parameters:
    - data: pandas Series or numpy array with time series data
    - threshold_factor: multiplier for spike detection threshold
    - max_wave_height: maximum realistic wave height (meters)
    - apply_smoothing: whether to apply additional smoothing
    - smooth_window: window size for smoothing filter
    
    Returns:
    - cleaned and optionally smoothed data
    """
    if isinstance(data, np.ndarray):
        data = pd.Series(data)
    
    if len(data) < 5:
        return data
    
    # First, remove obviously unrealistic values
    realistic_mask = (data > 0) & (data < max_wave_height) & np.isfinite(data)
    n_unrealistic = (~realistic_mask).sum()
    
    if n_unrealistic > 0:
        # Fill unrealistic values with NaN
        data_temp = data.copy()
        data_temp[~realistic_mask] = np.nan
    else:
        data_temp = data.copy()
    
    # Apply spike detection on realistic values only
    if len(data_temp.dropna()) < 5:
        return data_temp
    
    # Calculate rolling statistics on valid data
    window = min(25, len(data_temp.dropna()) // 4)  # Adaptive window size
    if window < 3:
        window = 3
    
    rolling_median = data_temp.rolling(window=window, center=True, min_periods=1).median()
    rolling_std = data_temp.rolling(window=window, center=True, min_periods=1).std()
    
    # Identify additional spikes
    diff_from_median = np.abs(data_temp - rolling_median)
    threshold = threshold_factor * rolling_std
    
    # Replace spikes with rolling median
    spike_mask = (diff_from_median > threshold) & np.isfinite(data_temp)
    cleaned_data = data_temp.copy()
    cleaned_data[spike_mask] = rolling_median[spike_mask]
    
    # Apply additional smoothing if requested
    if apply_smoothing and len(cleaned_data.dropna()) > smooth_window:
        # Use rolling mean for smoothing
        smoothed_data = cleaned_data.rolling(window=smooth_window, center=True, min_periods=1).mean()
        return smoothed_data
    
    return cleaned_data


def preprocess_wave_data(observed, modeled, apply_smoothing=True):
    """
    Preprocess wave data by removing spikes and optionally smoothing
    
    Parameters:
    - observed: observed wave height data (numpy array or pandas Series)
    - modeled: modeled wave height data (numpy array or pandas Series)
    - apply_smoothing: whether to apply smoothing
    
    Returns:
    - tuple of (cleaned_observed, cleaned_modeled)
    """
    # Convert to pandas Series if needed
    if isinstance(observed, np.ndarray):
        observed = pd.Series(observed)
    if isinstance(modeled, np.ndarray):
        modeled = pd.Series(modeled)
    
    # Process both datasets
    cleaned_observed = remove_spikes_and_smooth(observed, apply_smoothing=apply_smoothing)
    cleaned_modeled = remove_spikes_and_smooth(modeled, apply_smoothing=apply_smoothing)
    
    return cleaned_observed.values, cleaned_modeled.values


def calculate_nbias(observados, modelados, preprocess=True):
    """
    Calculate normalized bias (NBIAS) between observed and modeled data
    
    Parameters:
    - observados: observed data
    - modelados: modeled data  
    - preprocess: whether to apply spike removal and smoothing
    """
    if len(observados) == 0 or len(modelados) == 0:
        return np.nan

    # Apply preprocessing if requested
    if preprocess:
        observados, modelados = preprocess_wave_data(observados, modelados, apply_smoothing=True)

    n_bias_percent_values = []
    for i in range(len(observados)):
        if not np.isnan(observados[i]) and not np.isnan(modelados[i]) and observados[i] != 0:
            bias = modelados[i] - observados[i]
            n_bias = bias / observados[i]
            n_bias_percent = n_bias * 100
            n_bias_percent_values.append(n_bias_percent)

    if len(n_bias_percent_values) > 0:
        n_bias_percent_mean = np.mean(n_bias_percent_values)
    else:
        n_bias_percent_mean = np.nan

    return n_bias_percent_mean


def calculate_nrmse(observados, modelados, preprocess=True):
    """
    Calculate normalized root mean square error (NRMSE) between observed and modeled data
    
    Parameters:
    - observados: observed data
    - modelados: modeled data
    - preprocess: whether to apply spike removal and smoothing
    """
    if len(observados) == 0 or len(modelados) == 0:
        return np.nan

    # Apply preprocessing if requested
    if preprocess:
        observados, modelados = preprocess_wave_data(observados, modelados, apply_smoothing=True)

    n_rmse_percent_values = []
    for i in range(len(observados)):
        if not np.isnan(observados[i]) and not np.isnan(modelados[i]) and observados[i] != 0:
            rmse = np.sqrt((modelados[i] - observados[i]) ** 2)
            n_rmse = rmse / observados[i]
            n_rmse_percent = n_rmse * 100
            n_rmse_percent_values.append(n_rmse_percent)

    if len(n_rmse_percent_values) > 0:
        n_rmse_percent_mean = np.mean(n_rmse_percent_values)
    else:
        n_rmse_percent_mean = np.nan

    return n_rmse_percent_mean


def getStationInfo_emodnet(buoy_path, arq):
    file_path = os.path.join(buoy_path, arq)
    arq1 = xr.open_dataset(file_path, engine='netcdf4')
    try:
        station_info = {
            'Latitude': arq1['latitude'][0].values,
            'Longitude': arq1['longitude'][0].values
        }
    except Exception as e:
        print(f"Failed to fetch information for file: {arq}. Error: {e}")
        station_info = None
    finally:
        arq1.close()
    return station_info

def getStationInfo_ndbc(buoy_path, arq):
    file_path = os.path.join(buoy_path, arq)
    try:
        arq1 = xr.open_dataset(file_path)
        station_info = {
            'Latitude': arq1['latitude'].values,
            'Longitude': arq1['longitude'].values
        }
    except Exception as e:
        print(f"Failed to fetch information for file: {arq}. Error: {e}")
        station_info = None
    finally:
        arq1.close()

    return station_info


def create_custom_cmap(colors, n_colors):
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_colors)
    return custom_cmap


def plot_map_nbias(all_longitudes, all_latitudes, n_bias_percent_values):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)

    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = plt.FixedLocator(np.arange(-80, 81, 20))
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.top_labels = False
    gl.right_labels = False

    # Filter out None values
    valid_n_bias_percent_values = [value for value in n_bias_percent_values if value is not None]
    n_bias_percent_mean = np.nanmean(valid_n_bias_percent_values)
    
    lon_text = 58  
    lat_text = 50   
    ax.text(lon_text, lat_text, r"$\overline{\mathrm{NBIAS}}=$"+f"{n_bias_percent_mean:.2f}%", color='k', transform=ccrs.PlateCarree(), fontsize=12)

    colors1 = ['deepskyblue', 'lightgreen', 'khaki', 'orange', 'red']
    n_colors = 32
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors1, N=n_colors)
    norm = mcolors.TwoSlopeNorm(vmin=-40, vcenter=0, vmax=40)
    sc = plt.scatter(all_longitudes, all_latitudes, c=valid_n_bias_percent_values, cmap=custom_cmap, s=50, alpha=0.7, norm=norm)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('NBIAS (%)')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Buoy Positions - NBIAS')
    plt.grid(True)
    plt.tight_layout()
    save_name = f'../figs/ndbcBuoysNBIAS.jpeg'  
    plt.savefig(save_name, dpi=300)


def plot_map_nrmse(all_longitudes, all_latitudes, n_rmse_percent_values):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)

    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = plt.FixedLocator(np.arange(-80, 81, 20))
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.top_labels = False
    gl.right_labels = False

    # Filter out None values
    valid_n_rmse_percent_values = [value for value in n_rmse_percent_values if value is not None]
    n_rmse_percent_mean = np.nanmean(valid_n_rmse_percent_values)
    
    lon_text = 58  
    lat_text = 50   
    ax.text(lon_text, lat_text, r"$\overline{\mathrm{NRMSE}}=$"+f"{n_rmse_percent_mean:.2f}%", color='k', transform=ccrs.PlateCarree(), fontsize=12)

    colors1 = ['deepskyblue', 'lightgreen', 'khaki', 'orange', 'red']
    n_colors = 32
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors1, N=n_colors)
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=30, vmax=60)
    sc = plt.scatter(all_longitudes, all_latitudes, c=valid_n_rmse_percent_values, cmap=custom_cmap, s=50, alpha=0.7, norm=norm)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('NRMSE (%)')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Buoy Positions - NRMSE')
    plt.grid(True)
    plt.tight_layout()
    save_name = f'../figs/ndbcBuoysNRMSE.jpeg'  
    plt.savefig(save_name, dpi=300)


def check_data_ndbc(buoy, model):
    try:
        # Open files
        buoy_data = xr.open_dataset(buoy)
        model_data = xr.open_dataset(model)

        # Check common dates     
        valid_dates_buoy = buoy_data['time'].where(~np.isnan(buoy_data['hs']), drop=True)
        common_dates = pd.Index(valid_dates_buoy).intersection(model_data['time'])
        return common_dates if not common_dates.empty else None

    except Exception as e:
        print(f"An error occurred while checking the NetCDF file: {e}")
        return None
    finally:
        buoy_data.close()
        model_data.close()

def check_data_emodnet(buoy, model):
    buoy_data = None
    model_data = None

    try:
        buoy_data = xr.open_dataset(buoy, engine='netcdf4')
        model_data = xr.open_dataset(model, engine='netcdf4')

        reference_date = pd.to_datetime('2020-01-01')
        buoy_data['time'] = reference_date + pd.to_timedelta(buoy_data.time.values, unit='h')
        buoy_data['VHM0'] = xr.where(buoy_data['VHM0'] == 9.96921e+36, np.nan, buoy_data['VHM0'])
        valid_dates_buoy = buoy_data['time'].where(~np.isnan(buoy_data['VHM0']), drop=True)
        common_dates = pd.Index(valid_dates_buoy).intersection(pd.Index(model_data['time'].values))

        return common_dates if not common_dates.empty else None, buoy_data, model_data
    
    except Exception as e:
        print(f"An error occurred while checking the NetCDF file: {e}")
        return None, buoy_data, model_data
    
    finally:
        if buoy_data is not None:
            buoy_data.close()
        if model_data is not None:
            model_data.close()


def flatten_list(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, (list, np.ndarray)):
            if isinstance(item, np.ndarray) and item.ndim == 0:
                # Convert 0-d array to a list
                flattened_list.append(item.tolist())
            else:
                flattened_list.extend(flatten_list(item))  # Recursively flatten the list
        else:
            flattened_list.append(item)
    return flattened_list



# def replace_comma():
#        text = open("data/buoy/ISPRA2440.csv", "r")
#        text = ''.join([i for i in text]).replace(",", ".")
#        x = open("dir.csv", "w")
#        x.writelines(text)
#        x.close()
#        return



# def getVectComponents(direction):
#     return -np.sin(direction), -np.cos(direction)

