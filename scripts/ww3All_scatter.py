#%%
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
from tools import *


# Define your variables and paths
buoy_path1= '/data/cmcc/jc11022/buoys/ndbc/'
model_path1 = f'/data/cmcc/jc11022/ww3/points_ndbc'  #link the files from simulation path
stations1 = [os.path.splitext(arq)[0] for arq in os.listdir(buoy_path1) if arq.endswith('.nc')]

# Definindo suas variáveis e caminhos
buoy_path2= '/data/cmcc/jc11022/buoys/emodnet/'
model_path2 = f'/data/cmcc/jc11022/ww3/points_emodnet' #link the files from simulation path
#stations = ['42055','42067','6201051','6201050']  
arqs_nc2 = [arq for arq in os.listdir(buoy_path2) if arq.endswith('.nc')]
stations2 = [os.path.splitext(arq)[0] for arq in os.listdir(buoy_path2) if arq.endswith('.nc')]

all_latitudes1 = []
all_longitudes1 = []
n_bias_percent_values1 = []
n_rmse_percent_values1 = []
all_latitudes2 = []
all_longitudes2 = []
n_bias_percent_values2 = []
n_rmse_percent_values2 = []


for station_id in stations1:
    buoy_file1 = os.path.join(buoy_path1, f"{station_id}.nc")
    model_file1 = os.path.join(model_path1, f"ww3_{station_id}.nc")

    common_dates1 = check_data_ndbc(buoy_file1, model_file1)

    if common_dates1 is not None:
        with xr.open_dataset(buoy_file1) as buoy_data, xr.open_dataset(model_file1) as model_data:
            buoy_data_common1 = buoy_data.sel(time=common_dates1)
            model_data_common1 = model_data.sel(time=common_dates1)

            observed1 = buoy_data_common1['hs'].values
            modeled1 = model_data_common1['hs'].values

            n_bias_percent1 = calculate_nbias(observed1, modeled1)
            n_rmse_percent1 = calculate_nrmse(observed1, modeled1)

            station_info1 = getStationInfo_ndbc(buoy_path1, f"{station_id}.nc")
                    
            print(f"Station {station_id}: NBIAS Percentage = {n_bias_percent1}")
                    
            if station_info1 is not None:
                all_latitudes1.append(station_info1['Latitude'])
                all_longitudes1.append(station_info1['Longitude'])
                n_bias_percent_values1.append(n_bias_percent1)
                n_rmse_percent_values1.append(n_rmse_percent1)


for station_id in stations2:
    buoy_file2 = os.path.join(buoy_path2, f"{station_id}.nc")
    model_file2 = os.path.join(model_path2, f"ww3_{station_id}.nc")

    common_dates2, buoy_data2, model_data2= check_data_emodnet(buoy_file2, model_file2)

    if common_dates2 is not None:
        buoy_data_common2 = buoy_data2.sel(time=common_dates2)
        model_data_common2 = model_data2.sel(time=common_dates2)

        observed2 = buoy_data_common2['VHM0'].values
        modeled2 = model_data_common2['hs'].values

        n_bias_percent_value2 = calculate_nbias(observed2, modeled2)
        n_rmse_percent_value2 = calculate_nrmse(observed2, modeled2)

        station_info2 = getStationInfo_emodnet(buoy_path2, f"{station_id}.nc")
        latitudes2 = station_info2['Latitude']
        longitudes2 = station_info2['Longitude']

        print(f"Station {station_id}: NBIAS Percentage = {n_bias_percent_value2}")

        # Adicionando os valores de latitude e longitude para cada boia
        all_latitudes2.append(latitudes2)
        all_longitudes2.append(longitudes2)
        
        n_bias_percent_values2.append(n_bias_percent_value2)
        n_rmse_percent_values2.append(n_rmse_percent_value2)


    else:
        print(f"No common dates for {station_id}. Going to next station...")



# Flatten the lists
all_longitudes = flatten_list(all_longitudes1) + flatten_list(all_longitudes2)
all_latitudes = flatten_list(all_latitudes1) + flatten_list(all_latitudes2)
n_bias_percent_values = flatten_list(n_bias_percent_values1) + flatten_list(n_bias_percent_values2)
n_rmse_percent_values = flatten_list(n_rmse_percent_values1) + flatten_list(n_rmse_percent_values2)
# Debug prints to check the contents
print("Longitudes:", all_longitudes)
print("Latitudes:", all_latitudes)
print("NBIAS Values:", n_bias_percent_values)
print("NRMSE Values:", n_rmse_percent_values)
# Check if lengths match
assert len(all_longitudes) == len(all_latitudes) == len(n_bias_percent_values), "Arrays must be of the same length."

plot_map_nbias(all_longitudes, all_latitudes, n_bias_percent_values)
plot_map_nrmse(all_longitudes, all_latitudes, n_rmse_percent_values)

# %%

