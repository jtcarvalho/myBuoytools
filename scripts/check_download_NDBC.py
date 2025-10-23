#%%
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def parse_filename(filename):
    """Parse station ID and year from filename like '46050w2014.txt'."""
    # Remove the .txt extension
    name = filename.replace('.txt', '')
    
    # Find the 'w' character that separates station ID from year
    w_position = name.find('w')
    
    if (w_position == -1):
        return None, None  # 'w' not found in filename
    
    station_id = name[:w_position]
    year = name[w_position+1:]
    
    return station_id, year


def create_station_year_table(directory_path):
    """Create a table with station IDs and their available years."""
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return None
    
    # Dictionary to store station info
    stations_data = {}
    
    # List all .txt files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            station_id, year = parse_filename(filename)
            
            if station_id and year:
                if station_id not in stations_data:
                    stations_data[station_id] = []
                stations_data[station_id].append(year)
    
    # Create DataFrame with stations and their years
    data = []
    
    for station_id, years_list in sorted(stations_data.items()):
        years_str = ', '.join(sorted(years_list))
        data.append({
            'Station ID': station_id,
            'Available Years': years_str,
            'Count': len(years_list)
        })
    
    df = pd.DataFrame(data)
    
    return df


def find_stations_with_recent_data(station_table=None, directory_path=None, start_year=2019, end_year=2024):
    """
    Find stations that have data for all years in the specified range (inclusive).
    
    Parameters:
    -----------
    station_table : DataFrame, optional
        DataFrame containing station information. If None, it will be created.
    directory_path : str, optional
        Path to directory with spectrum files. Required if station_table is None.
    start_year : int, optional
        Start year of the range to check (default: 2019)
    end_year : int, optional
        End year of the range to check (default: 2024)
    
    Returns:
    --------
    DataFrame
        DataFrame containing only stations with data for all years in the range.
    """
    # Create station_table if not provided
    if station_table is None:
        if directory_path is None:
            raise ValueError("Either station_table or directory_path must be provided")
        station_table = create_station_year_table(directory_path)
    
    # Create a list of years to check (as strings)
    years_to_check = [str(year) for year in range(start_year, end_year + 1)]
    
    # Function to check if a station has all required years
    def has_all_years(years_str):
        available_years = [y.strip() for y in years_str.split(',')]
        return all(year in available_years for year in years_to_check)
    
    # Filter stations
    filtered_stations = station_table[station_table['Available Years'].apply(has_all_years)]
    
    return filtered_stations


def fetch_station_metadata(station_ids):
    """
    Fetch latitude, longitude and name information for a list of station IDs.
    Uses NDBC XML metadata which is more structured than HTML.
    
    Parameters:
    -----------
    station_ids : list
        List of station IDs to fetch metadata for
        
    Returns:
    --------
    DataFrame
        DataFrame with station_id, name, latitude, and longitude
    """
    print(f"Fetching metadata for {len(station_ids)} stations...")
    
    # Initialize lists to store station info
    latitudes = []
    longitudes = []
    names = []
    
    # Dictionary of known stations - MOVE IT HERE, outside of the loop
    known_stations = {
        # Atlântico Norte
        '41001': {'lat': 34.7, 'lon': -72.7, 'name': 'East of Hatteras'},
        '41002': {'lat': 32.3, 'lon': -75.4, 'name': 'South Hatteras'},
        '41004': {'lat': 32.5, 'lon': -79.1, 'name': 'EDISTO'},
        '41008': {'lat': 31.4, 'lon': -80.9, 'name': 'GRAYS REEF'},
        '41009': {'lat': 28.5, 'lon': -80.2, 'name': 'Canaveral'},
        '41010': {'lat': 28.9, 'lon': -78.5, 'name': 'CANAVERAL EAST'},
        '41013': {'lat': 33.4, 'lon': -77.7, 'name': 'FRYING PAN SHOALS'},
        '41025': {'lat': 35.0, 'lon': -75.4, 'name': 'Diamond Shoals'},
        '41040': {'lat': 14.5, 'lon': -53.0, 'name': 'North Atlantic'},
        '41043': {'lat': 21.1, 'lon': -64.8, 'name': 'Puerto Rico'},
        '41044': {'lat': 21.6, 'lon': -58.6, 'name': 'NTAS'},
        '41046': {'lat': 23.8, 'lon': -68.4, 'name': 'East Bahamas'},
        '41047': {'lat': 27.5, 'lon': -71.5, 'name': 'NE Bahamas'},
        '41049': {'lat': 27.5, 'lon': -62.9, 'name': 'South Bermuda'},
        
        # Golfo do México
        '42001': {'lat': 25.9, 'lon': -89.7, 'name': 'Mid-Gulf of Mexico'},
        '42002': {'lat': 26.1, 'lon': -93.8, 'name': 'Western Gulf of Mexico'},
        '42012': {'lat': 30.1, 'lon': -87.6, 'name': 'Orange Beach'},
        '42019': {'lat': 27.9, 'lon': -95.4, 'name': 'Freeport, TX'},
        '42020': {'lat': 27.0, 'lon': -96.7, 'name': 'Corpus Christi, TX'},
        '42036': {'lat': 28.5, 'lon': -84.5, 'name': 'West Tampa'},
        '42040': {'lat': 29.2, 'lon': -88.2, 'name': 'Mobile South'},
        '42055': {'lat': 22.1, 'lon': -94.0, 'name': 'Bay of Campeche'},
        '42056': {'lat': 19.9, 'lon': -85.1, 'name': 'Yucatan Basin'},
        '42059': {'lat': 15.2, 'lon': -67.5, 'name': 'Eastern Caribbean'},
        '42060': {'lat': 16.4, 'lon': -63.3, 'name': 'NE Caribbean'},
        
        # Costa Leste/Nova Inglaterra
        '44005': {'lat': 43.2, 'lon': -69.1, 'name': 'GULF OF MAINE'},
        '44007': {'lat': 43.5, 'lon': -70.1, 'name': 'Portland'},
        '44008': {'lat': 40.5, 'lon': -69.3, 'name': 'NANTUCKET'},
        '44009': {'lat': 38.5, 'lon': -74.7, 'name': 'Delaware Bay'},
        '44011': {'lat': 41.1, 'lon': -66.6, 'name': 'Georges Bank'},
        '44013': {'lat': 42.3, 'lon': -70.7, 'name': 'Boston'},
        '44014': {'lat': 36.6, 'lon': -74.8, 'name': 'Virginia Beach'},
        '44018': {'lat': 41.3, 'lon': -70.4, 'name': 'SE Cape Cod'},
        '44020': {'lat': 41.4, 'lon': -70.2, 'name': 'Nantucket Sound'},
        '44025': {'lat': 40.3, 'lon': -73.2, 'name': 'Long Island'},
        '44065': {'lat': 40.4, 'lon': -73.7, 'name': 'New York Harbor'},
        
        # # Grandes Lagos
        # '45001': {'lat': 43.2, 'lon': -86.6, 'name': 'Lake Michigan'},
        # '45002': {'lat': 45.3, 'lon': -86.4, 'name': 'N Lake Michigan'},
        # '45003': {'lat': 45.4, 'lon': -82.8, 'name': 'Lake Huron'},
        # '45004': {'lat': 47.6, 'lon': -86.6, 'name': 'NE Lake Superior'},
        # '45005': {'lat': 41.7, 'lon': -82.4, 'name': 'W Lake Erie'},
        # '45006': {'lat': 43.2, 'lon': -82.0, 'name': 'S Lake Huron'},
        # '45007': {'lat': 42.7, 'lon': -87.0, 'name': 'S Lake Michigan'},
        # '45008': {'lat': 44.3, 'lon': -82.4, 'name': 'S Lake Huron'},
        # '45012': {'lat': 43.6, 'lon': -77.4, 'name': 'Lake Ontario'},
        
        # Costa Oeste
        '46001': {'lat': 56.3, 'lon': -148.0, 'name': 'Gulf of Alaska'},
        '46005': {'lat': 46.1, 'lon': -131.0, 'name': 'Washington'},
        '46006': {'lat': 41.0, 'lon': -138.0, 'name': 'N Pacific'},
        '46011': {'lat': 35.0, 'lon': -121.0, 'name': 'Santa Maria'},
        '46013': {'lat': 38.2, 'lon': -123.3, 'name': 'Bodega Bay'},
        '46014': {'lat': 39.2, 'lon': -124.0, 'name': 'Point Arena'},
        '46022': {'lat': 40.8, 'lon': -124.5, 'name': 'Eel River'},
        '46025': {'lat': 33.7, 'lon': -119.1, 'name': 'Santa Monica Basin'},
        '46026': {'lat': 37.8, 'lon': -122.8, 'name': 'San Francisco'},
        '46027': {'lat': 41.9, 'lon': -124.4, 'name': 'St. Georges'},
        '46028': {'lat': 35.8, 'lon': -121.9, 'name': 'Cape San Martin'},
        '46029': {'lat': 46.1, 'lon': -124.5, 'name': 'Columbia River Bar'},
        '46035': {'lat': 57.0, 'lon': -177.7, 'name': 'Bering Sea'},
        '46041': {'lat': 47.4, 'lon': -124.8, 'name': 'Wash. Coast'},
        '46047': {'lat': 32.4, 'lon': -119.5, 'name': 'Tanner Banks'},
        '46050': {'lat': 44.6, 'lon': -124.5, 'name': 'Stonewall Bank'},
        '46053': {'lat': 34.3, 'lon': -119.8, 'name': 'E Santa Barbara'},
        '46054': {'lat': 34.3, 'lon': -120.5, 'name': 'W Santa Barbara'},
        '46059': {'lat': 38.0, 'lon': -130.0, 'name': 'California'},
        '46060': {'lat': 30.2, 'lon': -138.0, 'name': 'W Hawaii'},
        '46066': {'lat': 52.8, 'lon': -155.0, 'name': 'Aleutian Basin'},
        
        # Havaí
        '51000': {'lat': 23.5, 'lon': -154.0, 'name': 'NW Hawaii'},
        '51001': {'lat': 23.4, 'lon': -162.2, 'name': 'NW Hawaii'},
        '51002': {'lat': 17.1, 'lon': -157.8, 'name': 'SW Hawaii'},
        '51003': {'lat': 19.2, 'lon': -160.8, 'name': 'W Hawaii'},
        '51004': {'lat': 17.6, 'lon': -152.4, 'name': 'SE Hawaii'},
        '51101': {'lat': 24.3, 'lon': -162.1, 'name': 'NW Hawaii'}
    }
    
    # For each station, get the metadata
    for i, station_id in enumerate(station_ids):
        print(f"Checking station {station_id} ({i+1}/{len(station_ids)})")
        
        # Try to get metadata from NDBC's XML feed
        xml_url = f"https://www.ndbc.noaa.gov/data/stations/{station_id}.xml"
        try:
            xml_response = requests.get(xml_url)
            
            if xml_response.status_code == 200:
                try:
                    # Parse XML content
                    xml_soup = BeautifulSoup(xml_response.content, 'xml')
                    
                    # Get station info from XML
                    lat_elem = xml_soup.find('lat')
                    lon_elem = xml_soup.find('lon')
                    name_elem = xml_soup.find('name')
                    
                    if lat_elem and lon_elem:
                        lat = float(lat_elem.text)
                        lon = float(lon_elem.text)
                        latitudes.append(lat)
                        longitudes.append(lon)
                        
                        if name_elem:
                            names.append(name_elem.text)
                        else:
                            names.append(f"Station {station_id}")
                            
                        print(f"Found coordinates from XML: {lat}, {lon}")
                        continue
                except Exception as e:
                    print(f"Error parsing XML for station {station_id}: {e}")
            
            # Fallback to HTML parsing if XML doesn't work
            meta_url = f"https://www.ndbc.noaa.gov/station_page.php?station={station_id}"
            meta_response = requests.get(meta_url)
            
            meta_soup = BeautifulSoup(meta_response.content, 'html.parser')
            location_text = None
            
            # Look for the location information in the page
            for p in meta_soup.find_all('p'):
                if 'Location:' in p.text:
                    location_text = p.text.strip()
                    print(f"Found location text: {location_text}")
                    break
            
            # Try to find coordinates in a table (newer NDBC format)
            if not location_text:
                tables = meta_soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            if 'Position' in cells[0].text:
                                location_text = cells[1].text.strip()
                                print(f"Found location in table: {location_text}")
                                break
            
            lat = None
            lon = None
            
            # Try multiple parsing approaches for location text
            if location_text:
                # Various parsing approaches...
                coords_patterns = [
                    r'(\d+\.\d+).*?([NS]).*?(\d+\.\d+).*?([EW])',  # 34.7 N 72.7 W
                    r'(\d+\.\d+)°\s*([NS]).*?(\d+\.\d+)°\s*([EW])',  # 34.7° N 72.7° W
                    r'(\d+\.\d+)\s*([NS]).*?,\s*(\d+\.\d+)\s*([EW])'  # 34.7N, 72.7W
                ]
                
                for pattern in coords_patterns:
                    import re
                    match = re.search(pattern, location_text)
                    if match:
                        lat = float(match.group(1))
                        if match.group(2) == 'S':
                            lat = -lat
                        lon = float(match.group(3))
                        if match.group(4) == 'W':
                            lon = -lon
                        break
            
            if lat is not None and lon is not None:
                print(f"Extracted coordinates: {lat}, {lon}")
                latitudes.append(lat)
                longitudes.append(lon)
                
                # Try to get station name
                title = meta_soup.find('title')
                if title and '-' in title.text:
                    name = title.text.split('-')[1].strip()
                    names.append(name)
                else:
                    names.append(f"Station {station_id}")
            else:
                print(f"Failed to extract coordinates for station {station_id}")
                # Fall back to the NDBC historical station list for common buoys
                # This is a small dictionary of known stations
                
                if station_id in known_stations:
                    station_info = known_stations[station_id]
                    latitudes.append(station_info['lat'])
                    longitudes.append(station_info['lon'])
                    names.append(station_info['name'])
                    print(f"Used known coordinates for station {station_id}")
                else:
                    latitudes.append(None)
                    longitudes.append(None)
                    names.append(f"Station {station_id}")
                
        except Exception as e:
            print(f"Error extracting metadata for station {station_id}: {e}")
            latitudes.append(None)
            longitudes.append(None)
            names.append(f"Station {station_id}")
        
        # Add a small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    # Create a DataFrame with the results
    metadata_df = pd.DataFrame({
        'Station ID': station_ids,
        'Name': names,
        'Latitude': latitudes,
        'Longitude': longitudes
    })
    
    return metadata_df


def plot_station_locations(stations_df, output_path=None):
    """
    Plot station locations on a map using cartopy.
    
    Parameters:
    -----------
    stations_df : DataFrame
        DataFrame containing station information with Latitude and Longitude columns
    output_path : str, optional
        Path to save the map image. If None, the map will only be displayed.
    """
    # Check for latitude and longitude columns (case insensitive)
    lat_col = None
    lon_col = None
    
    for col in stations_df.columns:
        if col.lower() == 'latitude':
            lat_col = col
        elif col.lower() == 'longitude':
            lon_col = col
    
    if lat_col is None or lon_col is None:
        print("Error: Could not find latitude and longitude columns in the data.")
        print(f"Available columns: {list(stations_df.columns)}")
        return None
    
    # Filter out stations with missing coordinates
    valid_stations = stations_df.dropna(subset=[lat_col, lon_col])
    
    print(f"Plotting {len(valid_stations)} stations with valid coordinates")
    
    # Create figure and axes with a suitable projection
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    
    # Set global view
    ax.set_global()
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot stations
    scatter = ax.scatter(
        valid_stations[lon_col], 
        valid_stations[lat_col],
        transform=ccrs.PlateCarree(),
        marker='o',
        color='red',
        s=50,
        alpha=0.7,
        label='NDBC Stations'
    )
    
    # Add station ID labels (optional - can be commented out if too cluttered)
    id_col = 'Station ID' if 'Station ID' in valid_stations.columns else valid_stations.columns[0]
    for _, row in valid_stations.iterrows():
        ax.text(
            row[lon_col] + 0.5, 
            row[lat_col] + 0.2,
            row[id_col],
            transform=ccrs.PlateCarree(),
            fontsize=8
        )
    
    # Add title and legend
    plt.title('NDBC Stations with Spectral Wave Data (2019-2024)')
    plt.legend(loc='lower left')
    
    # Save figure if output path is provided
    if output_path:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Map saved to {output_path}")
    
    # Display the map
    plt.tight_layout()
    plt.show()
    
    return ax


def main():
    """Main function to organize workflow and provide menu options."""
    # Default path to the directory containing spec files
    default_directory = '/Users/jtakeo/googleDrive/myProjects/myBuoytools/data/juno'
    
    while True:
        print("\n=== NDBC Station Analysis Tool ===")
        print("1. Create station year inventory")
        print("2. Find stations with recent data (2019-2024)")
        print("3. Fetch station metadata (coordinates)")
        print("4. Plot station locations on map")
        print("5. Run complete analysis pipeline")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-5): ")
        
        if choice == '0':
            print("Exiting program.")
            break
            
        elif choice == '1':
            directory_path = input(f"Enter directory path (default: {default_directory}): ") or default_directory
            
            station_table = create_station_year_table(directory_path)
            if station_table is not None:
                print(f"Found {len(station_table)} stations with spectrum data.")
                
                # Ask if user wants to see the table
                if input("Display table? (y/n): ").lower() == 'y':
                    print(station_table)
                
                # Ask if user wants to save the table
                if input("Save table to CSV? (y/n): ").lower() == 'y':
                    output_path = input("Enter output path (default: station_years_inventory.csv): ") or 'station_years_inventory.csv'
                    station_table.to_csv(output_path, index=False)
                    print(f"Table saved to '{output_path}'")
            
        elif choice == '2':
            # First, check if we have a station table already
            if 'station_table' not in locals():
                print("No station table loaded. Creating one first.")
                directory_path = input(f"Enter directory path (default: {default_directory}): ") or default_directory
                station_table = create_station_year_table(directory_path)
                if station_table is None:
                    continue
            
            # Allow custom year range
            start_year = int(input("Enter start year (default: 2019): ") or "2019")
            end_year = int(input("Enter end year (default: 2024): ") or "2024")
            
            recent_stations = find_stations_with_recent_data(station_table, start_year=start_year, end_year=end_year)
            if recent_stations is not None:
                print(f"Found {len(recent_stations)} stations with data for all years {start_year}-{end_year}.")
                
                # Ask if user wants to see the results
                if input("Display results? (y/n): ").lower() == 'y':
                    print(recent_stations)
                
                # Ask if user wants to save the results
                if input("Save results to CSV? (y/n): ").lower() == 'y':
                    output_path = input(f"Enter output path (default: recent_stations_{start_year}_{end_year}.csv): ") or f"recent_stations_{start_year}_{end_year}.csv"
                    recent_stations.to_csv(output_path, index=False)
                    print(f"Recent stations list saved to '{output_path}'")
        
        elif choice == '3':
            # Check if we have recent stations already
            if 'recent_stations' not in locals():
                print("No stations selected for metadata. Please run option 2 first or load a CSV file.")
                load_option = input("Load from CSV file? (y/n): ")
                if load_option.lower() == 'y':
                    csv_path = input("Enter CSV path: ")
                    try:
                        recent_stations = pd.read_csv(csv_path)
                    except Exception as e:
                        print(f"Error loading CSV: {e}")
                        continue
                else:
                    continue
            
            # Get the list of station IDs
            station_ids = recent_stations['Station ID'].tolist()
            
            # Fetch metadata for these stations
            metadata = fetch_station_metadata(station_ids)
            
            # Merge with the recent_stations DataFrame
            recent_stations_with_location = recent_stations.merge(metadata, on='Station ID')
            
            print("\nStations with location information:")
            if input("Display results? (y/n): ").lower() == 'y':
                print(recent_stations_with_location)
            
            # Ask if user wants to save the results
            if input("Save results to CSV? (y/n): ").lower() == 'y':
                output_path = input("Enter output path (default: recent_stations_with_location.csv): ") or "recent_stations_with_location.csv"
                recent_stations_with_location.to_csv(output_path, index=False)
                print(f"Stations with location saved to '{output_path}'")
        
        elif choice == '4':
            # Check if we have stations with location data
            if 'recent_stations_with_location' not in locals():
                print("No stations with location data. Please run option 3 first or load a CSV file.")
                load_option = input("Load from CSV file? (y/n): ")
                if load_option.lower() == 'y':
                    csv_path = input("Enter CSV path: ")
                    try:
                        recent_stations_with_location = pd.read_csv(csv_path)
                    except Exception as e:
                        print(f"Error loading CSV: {e}")
                        continue
                else:
                    continue
            
            # Ask if user wants to save the plot
            save_option = input("Save plot to file? (y/n): ")
            if save_option.lower() == 'y':
                output_path = input("Enter output path (default: station_map.png): ") or "station_map.png"
                plot_station_locations(recent_stations_with_location, output_path)
            else:
                plot_station_locations(recent_stations_with_location)
        
        elif choice == '5':
            print("Running complete analysis pipeline...")
            
            # Get directory path
            directory_path = input(f"Enter directory path (default: {default_directory}): ") or default_directory
            
            # Step 1: Create station table
            print("\nStep 1: Creating station year inventory...")
            station_table = create_station_year_table(directory_path)
            if station_table is None:
                print("Error creating station table. Aborting pipeline.")
                continue
            
            # Step 2: Find recent stations
            print("\nStep 2: Finding stations with recent data...")
            recent_stations = find_stations_with_recent_data(station_table)
            if recent_stations is None or len(recent_stations) == 0:
                print("No stations with recent data found. Aborting pipeline.")
                continue
            
            # Step 3: Fetch metadata
            print("\nStep 3: Fetching station metadata...")
            station_ids = recent_stations['Station ID'].tolist()
            metadata = fetch_station_metadata(station_ids)
            recent_stations_with_location = recent_stations.merge(metadata, on='Station ID')
            
            # Step 4: Plot stations
            print("\nStep 4: Plotting station locations...")
            output_path = input("Enter map output path (default: station_map.png): ") or "station_map.png"
            plot_station_locations(recent_stations_with_location, output_path)
            
            # Save all results
            print("\nSaving all results...")
            station_table.to_csv('station_years_inventory.csv', index=False)
            recent_stations.to_csv('recent_stations_2019_2024.csv', index=False)
            recent_stations_with_location.to_csv('recent_stations_with_location.csv', index=False)
            
            print("\nComplete analysis pipeline finished!")
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()

# %%
