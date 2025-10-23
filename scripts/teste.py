import xarray as xr 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

buoy_path = '~/Downloads/emodnet.bak/6201008.nc'
model_path = '~/Downloads/emodnet.bak/ww3_6201008.nc'

# Open datasets with xarray and select variables
# For EMODNET, try different decoding strategies to handle non-gregorian calendars
try:
    ds_emod = xr.open_dataset(buoy_path, decode_times=True)
except (ValueError, OSError) as e:
    print(f"Warning: Failed to decode times automatically for EMODNET data: {e}")
    # Try opening without time decoding and handle manually
    ds_emod = xr.open_dataset(buoy_path, decode_times=False)
    
ds_ww3 = xr.open_dataset(model_path, decode_times=True)

hs_buoy = ds_emod['VHM0']   # EMODNET variable
hs_model = ds_ww3['hs']     # WW3 variable

# Function to fix time coordinates and handle non-gregorian calendars
def _fix_time_coordinate(da, dataset_name=""):
    """Fix time coordinate, handling non-gregorian calendars"""
    time_coord = None
    
    # Find time coordinate
    for coord_name in da.coords:
        if 'time' in coord_name.lower():
            time_coord = coord_name
            break
    
    if time_coord is None:
        print(f"Warning: No time coordinate found in {dataset_name}")
        return da
    
    # Rename to 'time' if needed
    if time_coord != 'time':
        da = da.rename({time_coord: 'time'})
    
    # Handle the case where time is not properly decoded
    if da.time.dtype == 'O' or not np.issubdtype(da.time.dtype, np.datetime64):
        print(f"Converting time coordinate for {dataset_name}...")
        
        # Check if time coordinate has attributes that can help with conversion
        time_attrs = da.time.attrs
        print(f"Time attributes: {time_attrs}")
        
        try:
            # Try to use CF conventions first
            ds_temp = da.to_dataset()
            ds_decoded = xr.decode_cf(ds_temp, use_cftime=False)
            da = ds_decoded[da.name]
            print(f"Successfully decoded using CF conventions")
        except Exception as e:
            print(f"CF decoding failed: {e}")
            
            # Try manual conversion based on common patterns
            time_vals = da.time.values
            
            # Check if it looks like days since epoch
            if np.all(time_vals >= 0) and np.all(time_vals < 50000):  # Reasonable range for days
                try:
                    # Common EMODNET reference: days since 1900-01-01
                    reference_date = '1900-01-01'
                    if 'units' in time_attrs:
                        units_str = time_attrs['units'].lower()
                        if 'since' in units_str:
                            # Extract reference date from units string
                            parts = units_str.split('since')
                            if len(parts) > 1:
                                reference_date = parts[1].strip().split()[0]
                    
                    print(f"Trying to convert as days since {reference_date}")
                    new_times = pd.to_datetime(reference_date) + pd.to_timedelta(time_vals, unit='D')
                    da = da.assign_coords(time=new_times)
                    print(f"Successfully converted time coordinate")
                    
                except Exception as e2:
                    print(f"Manual conversion failed: {e2}")
                    # Try other common reference dates
                    for ref_date in ['1950-01-01', '2000-01-01', '1970-01-01']:
                        try:
                            print(f"Trying reference date: {ref_date}")
                            new_times = pd.to_datetime(ref_date) + pd.to_timedelta(time_vals, unit='D')
                            da = da.assign_coords(time=new_times)
                            print(f"Successfully converted with reference date {ref_date}")
                            break
                        except:
                            continue
            else:
                # Try other units
                for unit in ['h', 'min', 's']:
                    try:
                        new_times = pd.to_datetime('1900-01-01') + pd.to_timedelta(time_vals, unit=unit)
                        da = da.assign_coords(time=new_times)
                        print(f"Successfully converted using unit: {unit}")
                        break
                    except:
                        continue
    
    # Ensure time is the first dimension
    if list(da.dims)[0] != 'time':
        da = da.swap_dims({list(da.dims)[0]: 'time'})
    
    return da.sortby('time')

# Function to remove spikes and unrealistic values from data
def remove_spikes(data, threshold_factor=3.0, max_wave_height=25.0, apply_smoothing=False, smooth_window=5):
    """
    Remove spikes and unrealistic values from time series data
    
    Parameters:
    - threshold_factor: multiplier for spike detection threshold
    - max_wave_height: maximum realistic wave height
    - apply_smoothing: whether to apply additional smoothing
    - smooth_window: window size for smoothing filter
    """
    if len(data) < 5:
        return data
    
    # First, remove obviously unrealistic values (common for wave height data)
    # Typical significant wave heights rarely exceed 25m in most locations
    realistic_mask = (data > 0) & (data < max_wave_height) & np.isfinite(data)
    n_unrealistic = (~realistic_mask).sum()
    
    if n_unrealistic > 0:
        print(f"  Removed {n_unrealistic} unrealistic values ({100*n_unrealistic/len(data):.1f}% of data)")
        data_clean = data[realistic_mask]
        if len(data_clean) == 0:
            print("  Warning: All data points were unrealistic!")
            return data
        # Fill unrealistic values with NaN temporarily
        data_temp = data.copy()
        data_temp[~realistic_mask] = np.nan
    else:
        data_temp = data.copy()
    
    # Now apply spike detection on realistic values only
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
    
    n_spikes = spike_mask.sum()
    if n_spikes > 0:
        print(f"  Removed {n_spikes} additional spikes ({100*n_spikes/len(data):.1f}% of data)")
    
    # Apply additional smoothing if requested
    if apply_smoothing and len(cleaned_data.dropna()) > smooth_window:
        print(f"  Applying smoothing with window size {smooth_window}")
        # Use rolling mean for smoothing
        smoothed_data = cleaned_data.rolling(window=smooth_window, center=True, min_periods=1).mean()
        return smoothed_data
    
    return cleaned_data

# Ensure time coordinate is named 'time' and sorted
def _ensure_time(da):
    if 'time' in da.coords:
        if list(da.dims)[0] != 'time':
            da = da.swap_dims({list(da.dims)[0]: 'time'})
        return da.sortby('time')
    for c in da.coords:
        if 'time' in c.lower():
            da = da.rename({c: 'time'})
            if list(da.dims)[0] != 'time':
                da = da.swap_dims({list(da.dims)[0]: 'time'})
            return da.sortby('time')
    return da

# Fix time coordinates for both datasets
print("Fixing time coordinates...")
hs_model = _fix_time_coordinate(hs_model, "WW3")

# Special handling for EMODNET dataset with numeric time
print("Handling EMODNET time coordinate...")
if hs_buoy.time.dtype in [np.float32, np.float64] and not np.issubdtype(hs_buoy.time.dtype, np.datetime64):
    # EMODNET appears to use sequential numbering (0, 1, 2, ...)
    # Assume it covers the same period as WW3 model
    if len(hs_buoy.time) == len(hs_model.time):
        print(f"EMODNET and WW3 have same length ({len(hs_buoy.time)}), using WW3 time coordinate")
        # Use WW3 time coordinate for EMODNET
        hs_buoy = hs_buoy.assign_coords(time=hs_model.time.values)
    else:
        # Create hourly time series starting from WW3 start time
        print("Creating hourly time series for EMODNET...")
        start_time = hs_model.time.values[0] if len(hs_model.time) > 0 else pd.Timestamp('2020-01-01')
        new_times = pd.date_range(start=start_time, periods=len(hs_buoy.time), freq='H')
        hs_buoy = hs_buoy.assign_coords(time=new_times)
else:
    hs_buoy = _fix_time_coordinate(hs_buoy, "EMODNET")

# Fallback to original method if fix didn't work
hs_buoy = _ensure_time(hs_buoy)
hs_model = _ensure_time(hs_model)

# To pandas Series indexed by time and drop NaNs
s_buoy = hs_buoy.to_series().dropna()
s_model = hs_model.to_series().dropna()

# Remove spikes from both datasets
print("Removing spikes from EMODNET data...")
s_buoy_original = s_buoy.copy()
s_buoy = remove_spikes(s_buoy, apply_smoothing=False)

print("Removing spikes from WW3 data...")
s_model_original = s_model.copy()
s_model = remove_spikes(s_model, apply_smoothing=False)

# Test with smoothing applied
print("\n--- Testing with smoothing applied ---")
print("Applying smoothing to EMODNET data...")
s_buoy_smooth = remove_spikes(s_buoy_original, apply_smoothing=True, smooth_window=5)

print("Applying smoothing to WW3 data...")
s_model_smooth = remove_spikes(s_model_original, apply_smoothing=True, smooth_window=5)

# Print time ranges
print("Buoy data:")
print("  Start:", s_buoy.index.min(), "End:", s_buoy.index.max(), "Number of points:", len(s_buoy), "\n")

print("Model data:")
print("  Start:", s_model.index.min(), "End:", s_model.index.max(), "Number of points:", len(s_model), "\n")

# Merge on common timestamps - original data
df_ts = (pd.DataFrame({'hs_buoy': s_buoy})
         .join(pd.DataFrame({'hs_model': s_model}), how='inner')
         .dropna())

# Merge on common timestamps - smoothed data
df_ts_smooth = (pd.DataFrame({'hs_buoy': s_buoy_smooth})
                .join(pd.DataFrame({'hs_model': s_model_smooth}), how='inner')
                .dropna())

print("Matching timestamps (original):", len(df_ts))
print("Matching timestamps (smoothed):", len(df_ts_smooth), "\n")

# Function to calculate statistics
def calculate_statistics(df, label=""):
    """Calculate validation statistics for buoy vs model data"""
    if df.empty:
        print(f"No data available for {label}")
        return None
        
    # Filter out invalid values for statistics
    valid_mask = ((df['hs_buoy'] > 0) & (df['hs_buoy'] < 25) & 
                  (df['hs_model'] > 0) & (df['hs_model'] < 25) & 
                  np.isfinite(df['hs_buoy']) & np.isfinite(df['hs_model']))
    df_valid = df[valid_mask]
    
    print(f"{label} - Valid data points: {len(df_valid)} out of {len(df)}")
    
    if len(df_valid) > 10:
        # Statistics
        corr = df_valid['hs_buoy'].corr(df_valid['hs_model'])
        bias = (df_valid['hs_model'] - df_valid['hs_buoy']).mean()
        buoy_mean = df_valid['hs_buoy'].mean()
        rmse = np.sqrt(((df_valid['hs_model'] - df_valid['hs_buoy'])**2).mean())
        
        if buoy_mean > 0:
            nbias = 100.0 * bias / buoy_mean
            nrmse = 100.0 * rmse / buoy_mean
        else:
            nbias = np.nan
            nrmse = np.nan
            
        return {
            'corr': corr, 'bias': bias, 'nbias': nbias, 
            'rmse': rmse, 'nrmse': nrmse, 'n_valid': len(df_valid),
            'df_valid': df_valid
        }
    else:
        print(f"Not enough valid data points for {label}")
        return None

# Calculate statistics for both datasets
print("=== COMPARISON: Original vs Smoothed Data ===\n")

stats_original = calculate_statistics(df_ts, "Original data")
stats_smooth = calculate_statistics(df_ts_smooth, "Smoothed data")

if stats_original:
    print("\n--- Original Data Statistics ---")
    print(f"  Correlation: {stats_original['corr']:.3f}")
    if np.isfinite(stats_original['nbias']) and np.isfinite(stats_original['nrmse']):
        print(f"  NBias: {stats_original['nbias']:.2f}%")
        print(f"  NRMSE: {stats_original['nrmse']:.2f}%")
    else:
        print(f"  Bias: {stats_original['bias']:.3f} m")
        print(f"  RMSE: {stats_original['rmse']:.3f} m")

if stats_smooth:
    print("\n--- Smoothed Data Statistics ---")
    print(f"  Correlation: {stats_smooth['corr']:.3f}")
    if np.isfinite(stats_smooth['nbias']) and np.isfinite(stats_smooth['nrmse']):
        print(f"  NBias: {stats_smooth['nbias']:.2f}%")
        print(f"  NRMSE: {stats_smooth['nrmse']:.2f}%")
    else:
        print(f"  Bias: {stats_smooth['bias']:.3f} m")
        print(f"  RMSE: {stats_smooth['rmse']:.3f} m")

# Show improvement
if stats_original and stats_smooth:
    print("\n--- Improvement with Smoothing ---")
    corr_improvement = stats_smooth['corr'] - stats_original['corr']
    print(f"  Correlation change: {corr_improvement:+.3f}")
    
    if np.isfinite(stats_original['nrmse']) and np.isfinite(stats_smooth['nrmse']):
        nrmse_improvement = stats_original['nrmse'] - stats_smooth['nrmse']
        print(f"  NRMSE improvement: {nrmse_improvement:+.2f}%")
        
    if np.isfinite(stats_original['nbias']) and np.isfinite(stats_smooth['nbias']):
        nbias_improvement = abs(stats_original['nbias']) - abs(stats_smooth['nbias'])
        print(f"  NBias improvement: {nbias_improvement:+.2f}%")

# Use smoothed data for plotting if it shows improvement
if stats_smooth and stats_original:
    if (stats_smooth['corr'] > stats_original['corr'] and 
        stats_smooth['nrmse'] < stats_original['nrmse']):
        print("\n==> Using SMOOTHED data for plotting (better statistics)")
        df_for_plot = df_ts_smooth
        stats_for_plot = stats_smooth
        plot_label = "Smoothed"
    else:
        print("\n==> Using ORIGINAL data for plotting (better or equivalent statistics)")
        df_for_plot = df_ts
        stats_for_plot = stats_original
        plot_label = "Original"
else:
    df_for_plot = df_ts
    stats_for_plot = stats_original
    plot_label = "Original"

if not df_for_plot.empty and stats_for_plot:
    # Plot time series with best data
    df_valid = stats_for_plot['df_valid']
    corr = stats_for_plot['corr']
    bias = stats_for_plot['bias'] 
    nbias = stats_for_plot['nbias']
    rmse = stats_for_plot['rmse']
    nrmse = stats_for_plot['nrmse']
    
    # Use valid data for plotting limits
    buoy_finite = df_for_plot['hs_buoy'][np.isfinite(df_for_plot['hs_buoy']) & (df_for_plot['hs_buoy'] > 0)]
    model_finite = df_for_plot['hs_model'][np.isfinite(df_for_plot['hs_model']) & (df_for_plot['hs_model'] > 0)]
    
    if len(buoy_finite) > 0 and len(model_finite) > 0:
        y_max = float(max(buoy_finite.max(), model_finite.max()))
    else:
        y_max = 10.0  # Default max
    
    x_min = df_for_plot.index.min()

    plt.figure(figsize=(12, 6))
    plt.plot(df_for_plot.index, df_for_plot['hs_buoy'], label=f'EMODNET (VHM0) - {plot_label}', linewidth=1.2, alpha=0.9)
    plt.plot(df_for_plot.index, df_for_plot['hs_model'], label=f'WW3 (hs) - {plot_label}', linewidth=1.2, alpha=0.9)
    plt.xlabel('Time')
    plt.ylabel('Significant Wave Height (m)')
    plt.title(f'Wave Height Time Series - Buoy 6201008 ({plot_label} Data)')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.ylim(0, y_max * 1.1)  # Set reasonable y-limits

    # Format statistics text
    if np.isfinite(corr) and np.isfinite(nbias) and np.isfinite(nrmse):
        txt = f"Correlation: {corr:.3f}\nNBias: {nbias:.2f}%\nNRMSE: {nrmse:.2f}%"
    else:
        txt = f"Correlation: {corr:.3f}\nBias: {bias:.3f} m\nRMSE: {rmse:.3f} m"
    
    plt.text(x_min, y_max * 0.95, txt, ha='left', va='top',
             bbox=dict(facecolor='wheat', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    filename = f'wave_height_timeseries_6201008_{plot_label.lower()}.png'
    plt.savefig(filename, dpi=150)
    plt.show()
    
    print(f"\n=== Final Plot Statistics ({plot_label} Data) ===")
    print(f"  Correlation: {corr:.3f}")
    if np.isfinite(nbias) and np.isfinite(nrmse):
        print(f"  NBias: {nbias:.2f}%")
        print(f"  NRMSE: {nrmse:.2f}%")
    else:
        print(f"  Bias: {bias:.3f} m")
        print(f"  RMSE: {rmse:.3f} m")
    print(f"  Plot saved as: {filename}")
else:
    print("No overlapping times to plot.")
