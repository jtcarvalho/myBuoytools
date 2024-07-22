import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def check_data(buoy, satellite):
    """
    Check for common dates between buoy and satellite data.
    """
    try:
        # Open files
        buoy_data = xr.open_dataset(buoy)
        satellite_data = xr.open_dataset(satellite)

        # Check common dates
        valid_dates_buoy = buoy_data['time'].where(~np.isnan(buoy_data['hs']), drop=True)
        valid_dates_sat = satellite_data['time'].where(~np.isnan(satellite_data['hs']), drop=True)
        #valid_dates_sat = satellite_data['time'].where(~np.isnan(satellite_data['WWM_1']), drop=True)
        common_dates = pd.Index(valid_dates_sat.values).intersection(pd.Index(buoy_data['time'].values))
        #common_dates = pd.Index(valid_dates_buoy.values).intersection(pd.Index(satellite_data['time'].values))

        return common_dates if not common_dates.empty else None

    except Exception as e:
        print(f"An error occurred while checking the NetCDF files: {e}")
        return None

    finally:
        buoy_data.close()
        satellite_data.close()

def load_data(buoy_file, satellite_file):
    """
    Load data from NetCDF files.
    """
    buoy_data = xr.open_dataset(buoy_file)
    satellite_data = xr.open_dataset(satellite_file)
    
    return buoy_data, satellite_data

def filter_data_by_dates(data, dates):
    """
    Filter data to include only the specified dates.
    """
    return data.sel(time=dates)

def plot_time_series(buoy_data, satellite_data, common_dates, output_file):
    """
    Plot time series data and save as PNG.
    """
    # Filter data by common dates
    buoy_filtered = filter_data_by_dates(buoy_data, common_dates)
    satellite_filtered = filter_data_by_dates(satellite_data, common_dates)
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(buoy_filtered['time'], buoy_filtered['hs'], label='Buoy', color='b')
    plt.plot(satellite_filtered['time'], satellite_filtered['WWM_1'], label='WWM', color='g')
    
    # Add title and labels
    #plt.title('Comparison for Buoy x WWM  point at East of Martinique')
    plt.title('Comparison for Buoy x WWM (1.43)  point at East of Martinique')
    plt.xlabel('Data')
    plt.ylabel('Hs')
    plt.ylim(0, 5)  # Set y-axis range
    plt.legend()
    
    # Save the plot as a PNG file
    plt.savefig(output_file)
    plt.close()

# Main program
if __name__ == "__main__":
    # File paths
    buoy_file = '/data/cmcc/jc11022/buoys/ndbc/41040.nc'
    #satellite_file = '/work/cmcc/jc11022/simulations/uGlobWW3/WWM/martinica/output/points/ww3_41040.nc'
    #satellite_file = '/work/cmcc/jc11022/simulations/uGlobWW3/WWM/martinica_143/output/points/ww3_41040.nc'
    satellite_file = '/data/cmcc/jc11022/ww3/points/ww3_41040.nc'

    # Check common dates
    common_dates = check_data(buoy_file, satellite_file)
    
    if common_dates is not None:
        # Load data
        buoy_data, satellite_data = load_data(buoy_file, satellite_file)
        
        # Output file
        #output_file = '/work/cmcc/jc11022/projects/ww3Tools/myBuoytools/figs/WWMxBuoy_exp2.png'
        #output_file = '/work/cmcc/jc11022/projects/ww3Tools/myBuoytools/figs/WWMxBuoy_exp1_beta143.png'
        output_file = '/work/cmcc/jc11022/projects/ww3Tools/myBuoytools/figs/ww3xbuoy_41040.png'
        # Plot and save the time series comparison
        plot_time_series(buoy_data, satellite_data, common_dates, output_file)
        
        print(f"Gráfico salvo como {output_file}")
    else:
        print("Não há datas comuns entre os dados da boia e do satélite.")
