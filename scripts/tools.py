
def calculate_nbias(observados, modelados):
    if len(observados) == 0 or len(modelados) == 0:
        return np.nan

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


def calculate_nrmse(observados, modelados):
    if len(observados) == 0 or len(modelados) == 0:
        return np.nan

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

