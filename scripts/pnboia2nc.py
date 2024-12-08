#%%
# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd

# Input CSV file

#arqname='metocean_bacia-de-santos-ax24'
#arqname='metocean_bacia-de-campos-ax23'
arqname = 'metocean_bacia-de-santos-bm01'
arq = f'../data/buoys/pnboia/{arqname}.csv'

# Read CSV file
df = pd.read_csv(arq, delimiter=',', skiprows=0, header=0)

# Variables to process and their new names
variables = {
    'swvht1': 'hs',    # Significant wave height
    'tp1': 'tp',       # Peak period
    'mxwvht1': 'mxHs', # Maximum wave height
    'wvdir1': 'dir'    # Wave direction
}

# Define the full time range
start_date = '2020-01-01 00:00:00'
end_date = '2023-12-31 23:00:00'
date_range = pd.date_range(start_date, end_date, freq='1h')

# Convert string dates to datetime
date_time = pd.to_datetime(df['date_time'], format='%Y-%m-%d %H:%M:%S')

# Create a DataFrame with NaN for the full time range
data_complete = pd.DataFrame(index=date_range, columns=variables.values())

# Process each variable
for original_var, new_var in variables.items():
    if original_var in df.columns:
        # Convert to numeric values
        values = pd.to_numeric(df[original_var], errors='coerce').values
        # Create a DataFrame for the existing data
        data_partial = pd.DataFrame(values, index=date_time, columns=[new_var])
        # Combine the existing data with the full time range
        combined_data = data_partial.reindex(data_complete.index).combine_first(data_complete[[new_var]])
        # Replace invalid values (< 0) with NaN
        data_complete[new_var] = combined_data[new_var].where(combined_data[new_var] >= 0)

# Convert the DataFrame to an xarray Dataset
ds = xr.Dataset.from_dataframe(data_complete)
ds = ds.set_coords('index')
ds = ds.rename({'index': 'time'})

# Save to NetCDF
output_path = f'../data/{arqname}.nc'
ds.to_netcdf(output_path)
print(f'Data saved to: {output_path}')

# Plot the data (e.g., variable 'hs')
plt.figure(figsize=(12, 6))

# Scatter plot
plt.scatter(ds['time'], ds['hs'], color='blue', marker='o', s=1, label='hs')

# Set axis labels and title
plt.title('Significant Wave Height (hs)', fontsize=14)
plt.xlabel('Time', fontsize=12)
plt.ylabel('hs (m)', fontsize=12)

# Format x-axis ticks for better readability
plt.xlim(date_range[0], date_range[-1])
plt.xticks(pd.date_range(start=start_date, end=end_date, freq='6MS'), rotation=45)
plt.grid()

# Show plot
plt.tight_layout()
plt.show()


