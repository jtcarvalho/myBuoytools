import os
import glob
import json
import numpy as np
import xarray as xr
from natsort import natsorted

def nearest(ds, x, y):
    """Find the index of the nearest point in the dataset."""
    return np.argmin(np.abs(ds.longitude.values - x) + np.abs(ds.latitude.values - y))
    #return np.argmin(np.abs(ds.SCHISM_hgrid_node_x.values - x) + np.abs(ds.SCHISM_hgrid_node_y.values - y))

def buoy_extraction(base, x, y, outname, variables, grid_type):
    """Extract equivalent buoy point data from WW3 files."""
    buffer = []
    for f in natsorted(glob.glob(base)):
        print(f)
        ds = xr.open_dataset(f)
        
        if grid_type == 'unstructured':
            ds = ds.isel(node=nearest(ds, x, y))[variables] #ww3
            #ds = ds.isel(nSCHISM_hgrid_node=nearest(ds, x, y))[variables]  #WWM        
            #ds=ds.interp(node(ds,x,y),method='linear')[variables] #another way
        elif grid_type == 'structured':
            ds = ds.sel(lat=y, lon=x, method='nearest')[variables]
        else:
            raise ValueError("grid_type must be 'unstructured' or 'structured'")
        
        print(x, y)
        print('#-----#')
        buffer.append(ds)

    out = xr.concat(buffer, dim='time')
    out.to_netcdf(f'{outname}.nc')

def load_buoy_points(file_path):
    """Load buoy points from a JSON file."""
    with open(file_path, 'r') as file:
        points = json.load(file)
    return points

def process_buoys(points, input_path, output_path, variables, grid_type):
    """Process all buoy points and save extracted data."""
    for name, info in points.items():
        pto = name
        x = info['x']
        y = info['y']
        print("Processing buoy id:", pto)
        fileout = f'ww3_{pto}'
        buoy_extraction(input_path, x, y, os.path.join(output_path, fileout), variables, grid_type)

def configure_experiment(exp_type):
    """Configure the experiment based on the chosen type."""
    if exp_type == 'unstructured_ww3':
        #exp = 'expb2_143_psi'
        exp = 'highResExperiments/gloH_article_exps/exp1_era5'
        #exp = 'expb2_133_psi_zalp0015'
        grid_type = 'unstructured'
        variables = ['hs']
        input_path = f'/work/cmcc/jc11022/simulations/uGlobWW3/{exp}/output/ww3.2020*.nc'
        #output_path = f'/work/cmcc/jc11022/simulations/uGlobWW3/{exp}/output/points_emodnet/'
        #output_path = f'/work/cmcc/jc11022/simulations/uGlobWW3/{exp}/output/points_ndbc/'
        output_path = f'/work/cmcc/jc11022/simulations/uGlobWW3/{exp}/output/points_ndbc/' 

    elif exp_type == 'unstructured_wwm':
        #exp = 'martinica'
        exp = 'martinica_143'
        grid_type = 'unstructured'
        variables = ['WWM_1']
        #input_path = f'/work/cmcc/lm09621/work/WWMV/schismWwmIce/run_experiment_coarse/outputs_final/GESWP_ERA5_schismwwm_*.nc'#exp2
        input_path = f'/work/cmcc/lm09621/work/WWMV/schismWwmIce/run_experiment_coarse/outputs_final_betamax1.43/GESWP_ERA5_schismwwm_*.nc'#exp1_beta143
        output_path = f'/work/cmcc/jc11022/simulations/uGlobWW3/WWM/{exp}/output/points/'

    elif exp_type == 'structured_ww3':
        exp = 'global_reg'
        grid_type = 'structured'
        variables = ['hs']
        input_path = f'/work/cmcc/jc11022/simulations/uGlobWW3/{exp}/output/ww3.*.nc'
        output_path = f'/work/cmcc/jc11022/simulations/uGlobWW3/{exp}/output/points/'
    elif exp_type == 'satellite':
        exp = 'regridUnst'
        grid_type = 'structured'
        variables = ['WWM_1']
        input_path = f'/work/cmcc/ww3_cst-dev/tools/regridUnst/WWM/REG_2019*.nc'
        output_path = f'/work/cmcc/jc11022/simulations/uGlobWW3/{exp}/points/'
    else:
        raise ValueError("exp_type must be 'unstructured_ww3', 'structured_ww3', or 'satellite'")
    
    return input_path, output_path, variables, grid_type

def main():
    """Main function to run the processing."""
    # Points file
    #points_file = '../aux/pointsEMODNET.info'
    points_file = '../aux/pointsNDBC.info'
    #points_file = './pointsComparison.info'

    # Load buoy points
    points = load_buoy_points(points_file)

    # Choose experiment type: 'unstructured_ww3', 'structured_ww3', or 'satellite'
    exp_type = 'unstructured_ww3'  # Change this as needed

    # Configure experiment
    input_path, output_path, variables, grid_type = configure_experiment(exp_type)
    
    # Process buoys
    process_buoys(points, input_path, output_path, variables, grid_type)

if __name__ == "__main__":
    main()
