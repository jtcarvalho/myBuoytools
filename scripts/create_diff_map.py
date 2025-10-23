# -*- coding: utf-8 -*-
"""
Gera um mapa mostrando a diferença percentual entre os valores 'high' e 'coarse' para cada estação.
"""
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from shapely.geometry import Point
import requests
import os
from geopy.distance import geodesic
# Função para baixar o shapefile da costa se não existir
def download_coastline_shapefile():
    url = "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_coastline.zip"
    out_dir = "aux/ne_10m_coastline"
    shp_path = os.path.join(out_dir, "ne_10m_coastline.shp")
    if not os.path.exists(shp_path):
        import zipfile
        import io
        print("Downloading Natural Earth coastline shapefile...")
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(out_dir)
    return shp_path
# Carregar shapefile da costa
coast_shp = download_coastline_shapefile()
coast_gdf = gpd.read_file(coast_shp)
# Função para calcular a menor distância de um ponto à costa (em km)
def min_distance_to_coast(lon, lat, coast_gdf):
    point = Point(lon, lat)
    # Coastline geometries are in WGS84
    min_dist = coast_gdf.distance(point).min()
    # Convert degrees to km (approx)
    # 1 degree ~ 111 km
    return min_dist * 111


# Carregar os dados
coarse = pd.read_csv('./validation_metrics_coarse.csv')
high = pd.read_csv('./validation_metrics_high.csv')

# Mesclar os dataframes pelas colunas de identificação
merged = pd.merge(coarse, high, on=['station', 'type'], suffixes=('_coarse', '_high'))

# Calcular diferenças percentuais

metrics = [
    ('nrmse_pct', 'NRMSE (%)'),
    ('nbias_pct', 'NBIAS (%)'),
    ('correlation', 'Correlation (R)')
]
diffs = {}
improved = {}

for metric, label in metrics:
    # Calcular distância à costa para cada estação
    merged['dist_to_coast'] = [min_distance_to_coast(lon, lat, coast_gdf) for lon, lat in zip(merged['longitude_coarse'], merged['latitude_coarse'])]
    diff = (merged[f'{metric}_high'] - merged[f'{metric}_coarse']) / merged[f'{metric}_coarse'] * 100
    diffs[metric] = diff
    total_mean = diff.mean()
    if metric == 'correlation':
        improved_mask = merged[f'{metric}_high'] > merged[f'{metric}_coarse']
        improved_pct = improved_mask.sum() / len(merged) * 100
        mean_improvement = diff[improved_mask].mean() if improved_mask.any() else 0
        worse_mask = merged[f'{metric}_high'] < merged[f'{metric}_coarse']
        mean_worsening = diff[worse_mask].mean() if worse_mask.any() else 0
        print(f"{label}: High is better in {improved_pct:.1f}% of stations, with a mean increase of {mean_improvement:.2f}% in correlation.")
        print(f"{label}: High is worse in {(worse_mask.sum()/len(merged)*100):.1f}% of stations, with a mean decrease of {mean_worsening:.2f}% in correlation.")
        print(f"{label}: Total mean difference (all stations): {total_mean:.2f}%\n")
        # Location stats
        if improved_mask.any():
            print(f"Mean location (improved): lon={merged['longitude_coarse'][improved_mask].mean():.2f}, lat={merged['latitude_coarse'][improved_mask].mean():.2f}")
            print(f"Mean distance to coast (improved): {merged['dist_to_coast'][improved_mask].mean():.2f} km")
        if worse_mask.any():
            print(f"Mean location (worse): lon={merged['longitude_coarse'][worse_mask].mean():.2f}, lat={merged['latitude_coarse'][worse_mask].mean():.2f}\n")
            print(f"Mean distance to coast (worse): {merged['dist_to_coast'][worse_mask].mean():.2f} km\n")
    else:
        improved_mask = merged[f'{metric}_high'] < merged[f'{metric}_coarse']
        improved_pct = improved_mask.sum() / len(merged) * 100
        mean_reduction = abs(diff[improved_mask].mean()) if improved_mask.any() else 0
        worse_mask = merged[f'{metric}_high'] > merged[f'{metric}_coarse']
        mean_worsening = abs(diff[worse_mask].mean()) if worse_mask.any() else 0
        print(f"{label}: High is better in {improved_pct:.1f}% of stations, with a mean reduction of {mean_reduction:.2f}%.")
        print(f"{label}: High is worse in {(worse_mask.sum()/len(merged)*100):.1f}% of stations, with a mean worsening of {mean_worsening:.2f}%.")
        print(f"{label}: Total mean difference (all stations): {total_mean:.2f}%\n")
        # Location stats
        if improved_mask.any():
            print(f"Mean location (improved): lon={merged['longitude_coarse'][improved_mask].mean():.2f}, lat={merged['latitude_coarse'][improved_mask].mean():.2f}")
            print(f"Mean distance to coast (improved): {merged['dist_to_coast'][improved_mask].mean():.2f} km")
        if worse_mask.any():
            print(f"Mean location (worse): lon={merged['longitude_coarse'][worse_mask].mean():.2f}, lat={merged['latitude_coarse'][worse_mask].mean():.2f}\n")
            print(f"Mean distance to coast (worse): {merged['dist_to_coast'][worse_mask].mean():.2f} km\n")
    improved[metric] = improved_pct

# Plotar mapas empilhados
fig, axes = plt.subplots(3, 1, figsize=(12, 18), subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(hspace=0.25)



for i, (metric, label) in enumerate(metrics):
    ax = axes[i]
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    # Ocean is white, no borders

    if metric in ['nrmse_pct', 'nbias_pct']:
        # bwr colormap, lighter extremes
        from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
        import numpy as np
        # Create a custom bwr with lighter extremes
        bwr = plt.get_cmap('bwr')
        colors = bwr(np.linspace(0, 1, 256))
        colors[:30, :3] = 1.0  # lighter blue
        colors[-30:, :3] = 1.0  # lighter red
        custom_bwr = LinearSegmentedColormap.from_list('custom_bwr', colors)
        norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
        sc = ax.scatter(
            merged['longitude_coarse'], merged['latitude_coarse'],
            c=diffs[metric], cmap=custom_bwr, norm=norm, s=60, edgecolor='k'
        )
        cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.01, aspect=20)
        cbar.set_label(f'Percent difference {label} (high - coarse)', fontsize=11)
        cbar.set_ticks([-75, -50, -25, 0, 25, 50, 75])
        cbar.ax.tick_params(labelsize=10)
    else:
        # Correlation: monotonic green, 0 white, 100 green
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0, vmax=100)
        cmap = plt.get_cmap('Greens')
        corr_diff = diffs[metric].clip(lower=0)
        sc = ax.scatter(
            merged['longitude_coarse'], merged['latitude_coarse'],
            c=corr_diff, cmap=cmap, norm=norm, s=60, edgecolor='k'
        )
        cbar = plt.colorbar(sc, ax=ax, orientation='vertical', pad=0.01, aspect=20)
        cbar.set_label('Percent improvement in Correlation (R) (high - coarse)', fontsize=11)
        cbar.ax.tick_params(labelsize=10)
    ax.set_title(f'Percent difference {label} between high and coarse', fontsize=13)

plt.suptitle('Percent difference between high and coarse by station (NRMSE, NBIAS, Correlation)', fontsize=16, y=0.98)
plt.savefig('figs/validation_diff_maps.png', dpi=300, bbox_inches='tight')
plt.show()
