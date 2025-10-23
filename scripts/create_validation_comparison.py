#!/usr/bin/env python3
"""
Script para criar figura comparativa dos mapas de validação geral e P99 (apenas NBIAS e NRMSE)
Subplots: (a) Validação geral, (b) Validação P99
Colormap NRMSE: rainbow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Arquivos
stats_file = os.path.join(script_dir, 'summary_statistics.csv')
metrics_file = os.path.join(script_dir, 'validation_metrics.csv')
stations_file = os.path.join(script_dir, '../stations_match.csv')
output_file = os.path.join(script_dir, 'validation_comparison.png')

# Carregar dados gerais
stats = pd.read_csv(stats_file)
stations = pd.read_csv(stations_file)

df = pd.merge(stats, stations[['estacao', 'lon_obs', 'lat_obs']], on='estacao', how='left')
df = df.dropna(subset=['lon_obs', 'lat_obs'])

df['nrmse'] = (df['rmse'] / df['hs_obs_mean']) * 100
df['nbias'] = (df['bias'] / df['hs_obs_mean']) * 100

# Carregar dados P99
# Se não houver validation_metrics.csv, usa as coordenadas aproximadas
if os.path.exists(metrics_file):
    metrics = pd.read_csv(metrics_file)
    df_p99 = pd.merge(stats, metrics[['station', 'longitude', 'latitude']], left_on='estacao', right_on='station', how='left')
else:
    df_p99 = df.copy()
    df_p99['longitude'] = df_p99['lon_obs']
    df_p99['latitude'] = df_p99['lat_obs']

df_p99 = df_p99.dropna(subset=['longitude', 'latitude'])
df_p99['nbias_p99'] = (df_p99['bias_p99'] / df_p99['p99_obs']) * 100
df_p99['nrmse_p99'] = (df_p99['rmse_p99'] / df_p99['p99_obs']) * 100

# Figura
fig = plt.figure(figsize=(18, 10))



# --- Subplot (a) NRMSE Geral ---
ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
ax1.set_global()
ax1.coastlines(resolution='110m', linewidth=0.8, color='black')
ax1.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
ax1.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
gl1 = ax1.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl1.top_labels = False
gl1.right_labels = False
gl1.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
gl1.ylocator = plt.FixedLocator(np.arange(-90, 91, 30))
scatter1 = ax1.scatter(df['lon_obs'], df['lat_obs'], c=df['nrmse'], cmap='rainbow', s=150, alpha=0.8, edgecolors='black', linewidth=0.8, vmin=0, vmax=60, zorder=5, transform=ccrs.PlateCarree())
cbar1 = plt.colorbar(scatter1, ax=ax1, orientation='vertical', pad=0.05, shrink=0.55)
cbar1.set_label('NRMSE (%)', fontsize=11, fontweight='bold')
# Subplot label (a) and mean value (lower left)
nrmse_mean = df['nrmse'].mean()
ax1.text(0.64, 0.76, r'$\overline{\mathrm{NRMSE}} = %.2f\%%$' % nrmse_mean, transform=ax1.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left')#, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.text(0.01, 0.07, '(a)', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='left')#, bbox=dict(boxstyle='round', facecolor='black', alpha=0.7), color='white')



# --- Subplot (b) NRMSE P99 ---
ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
ax2.set_global()
ax2.coastlines(resolution='110m', linewidth=0.8, color='black')
ax2.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
ax2.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
gl2 = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl2.top_labels = False
gl2.right_labels = False
gl2.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
gl2.ylocator = plt.FixedLocator(np.arange(-90, 91, 30))
scatter2 = ax2.scatter(df_p99['longitude'], df_p99['latitude'], c=df_p99['nrmse_p99'], cmap='rainbow', s=150, alpha=0.8, edgecolors='black', linewidth=0.8, vmin=0, vmax=60, zorder=5, transform=ccrs.PlateCarree())
cbar2 = plt.colorbar(scatter2, ax=ax2, orientation='vertical', pad=0.05, shrink=0.55)
cbar2.set_label('NRMSE P99 (%)', fontsize=11, fontweight='bold')
# Subplot label (b) and mean value (lower left)
nrmse_p99_mean = df_p99['nrmse_p99'].mean()
ax2.text(0.64, 0.76, r'$\overline{\mathrm{NRMSE}} = %.2f\%%$' % nrmse_p99_mean, transform=ax2.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left')#)
ax2.text(0.01, 0.07, '(b)', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='left')# bbox=dict(boxstyle='round', facecolor='black', alpha=0.7), color='white')



# --- Subplot (c) NBIAS Geral ---
ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
ax3.set_global()
ax3.coastlines(resolution='110m', linewidth=0.8, color='black')
ax3.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
ax3.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
gl3 = ax3.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl3.top_labels = False
gl3.right_labels = False
gl3.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
gl3.ylocator = plt.FixedLocator(np.arange(-90, 91, 30))
scatter3 = ax3.scatter(df['lon_obs'], df['lat_obs'], c=df['nbias'], cmap='RdBu_r', s=150, alpha=0.8, edgecolors='black', linewidth=0.8, vmin=-40, vmax=40, zorder=5, transform=ccrs.PlateCarree())
cbar3 = plt.colorbar(scatter3, ax=ax3, orientation='vertical', pad=0.05, shrink=0.55)
cbar3.set_label('NBIAS (%)', fontsize=11, fontweight='bold')
# Subplot label (c) and mean value (lower left)
nbias_mean = df['nbias'].mean()
ax3.text(0.64, 0.76, r'$\overline{\mathrm{NBIAS}} = %.2f\%%$' % nbias_mean, transform=ax3.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left')#, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax3.text(0.01, 0.07, '(c)', transform=ax3.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='left')#, bbox=dict(boxstyle='round', facecolor='black', alpha=0.7), color='white')



# --- Subplot (d) NBIAS P99 ---
ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree())
ax4.set_global()
ax4.coastlines(resolution='110m', linewidth=0.8, color='black')
ax4.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
ax4.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
gl4 = ax4.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl4.top_labels = False
gl4.right_labels = False
gl4.xlocator = plt.FixedLocator(np.arange(-180, 181, 60))
gl4.ylocator = plt.FixedLocator(np.arange(-90, 91, 30))
scatter4 = ax4.scatter(df_p99['longitude'], df_p99['latitude'], c=df_p99['nbias_p99'], cmap='RdBu_r', s=150, alpha=0.8, edgecolors='black', linewidth=0.8, vmin=-40, vmax=40, zorder=5, transform=ccrs.PlateCarree())
cbar4 = plt.colorbar(scatter4, ax=ax4, orientation='vertical', pad=0.05, shrink=0.55)
cbar4.set_label('NBIAS P99 (%)', fontsize=11, fontweight='bold')
# Subplot label (d) and mean value (lower left)
nbias_p99_mean = df_p99['nbias_p99'].mean()
ax4.text(0.64, 0.76, r'$\overline{\mathrm{NBIAS}} = %.2f\%%$' % nbias_p99_mean, transform=ax4.transAxes, fontsize=11, fontweight='bold', va='bottom', ha='left')#, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax4.text(0.01, 0.07, '(d)', transform=ax4.transAxes, fontsize=14, fontweight='bold', va='bottom', ha='left')#, bbox=dict(boxstyle='round', facecolor='black', alpha=0.7), color='white')

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Figura comparativa salva: {output_file}")
plt.show()
