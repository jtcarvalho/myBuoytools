#%%
import os,sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# Definindo suas variáveis e caminhos
buoy_path = '/data/cmcc/jc11022/buoys/emodnet/'
exp = 'expb2_143_psi'
model_path = f'/work/cmcc/jc11022/simulations/uGlobWW3/{exp}/output/points/'   
station_id='6100197'

buoy_file = os.path.join(buoy_path, f"{station_id}.nc")
model_file = os.path.join(model_path, f"ww3_{station_id}.nc")

model_data = xr.open_dataset(model_file)
buoy_data = xr.open_dataset(buoy_file)

# buoy_data['VHM0'] = buoy_data['VHM0'].astype(str)
# buoy_data['VHM0'] = xr.where(buoy_data['VHM0'] != '_', buoy_data['VHM0'], np.nan)
# buoy_data['VHM0'] = buoy_data['VHM0'].astype(float)
# Converter o tempo para o formato datetime
reference_date = pd.to_datetime('2020-01-01')  
buoy_data['time'] = reference_date + pd.to_timedelta(buoy_data.time.values, unit='h')
buoy_data['VHM0'] = xr.where(buoy_data['VHM0'] == 9.96921e+36, np.nan, buoy_data['VHM0'])

# plt.plot(buoy_data.time,buoy_data.VHM0.values)
# plt.show()
# plt.plot(model_data.time,model_data.hs.values)
# plt.show()

#%%
# Extraindo as datas onde VHM0 não é NaN
valid_dates_buoy = buoy_data['time'].where(~np.isnan(buoy_data['VHM0']), drop=True)

# Encontrando as datas comuns entre os conjuntos de dados considerando apenas as datas válidas de Hs/VHM0
common_dates = pd.Index(valid_dates_buoy).intersection(model_data['time'])

#common_dates = np.intersect1d(buoy_data['time'].values, model_data['time'].values)
buoy_data_common = buoy_data.sel(time=common_dates)
model_data_common = model_data.sel(time=common_dates)

fig = plt.figure(figsize=(10, 6))

plt.plot(buoy_data_common.time,buoy_data_common.VHM0.values,'r')
plt.plot(model_data_common.time,model_data_common.hs.values,'b')
plt.savefig('../figs/teste.png', dpi=300)
# %%
