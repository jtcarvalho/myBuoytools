#%%
import csv
import os

def main():
    """
    Read NDBC station data and format as WW3 points file.
    Format: lon, lat 'station_id' with spaces in IDs replaced by underscores.
    """
    # Path to the input CSV file
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          "data", "stations_NDBC_2019-2023.csv")
    
    # Path to the output text file
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "data", "ndbc_ww3points.txt")
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"Error: Input file not found: {csv_path}")
        return
    
    # Check if output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Counter for processed stations
    count = 0
    
    try:
        # Open output file for writing
        with open(output_path, 'w') as out_file:
            # Process the CSV file
            with open(csv_path, 'r') as f:
                print(f"Reading from: {csv_path}")
                
                # Create CSV reader WITH header row (don't skip first line)
                reader = csv.DictReader(f)
                
                # Process each row
                for row in reader:
                    # Skip rows with missing coordinates
                    if not row.get('Latitude') or not row.get('Longitude'):
                        continue
                        
                    # Extract station ID
                    station_id = row.get('Station ID', '')
                    
                    # Replace spaces with underscores if needed
                    station_id = station_id.replace(' ', '_')
                    
                    # Get coordinates
                    try:
                        lat = float(row['Latitude'])
                        lon = float(row['Longitude'])
                        
                        # Write formatted output to file
                        line = f"{lon}, {lat} '{station_id}'\n"
                        out_file.write(line)
                        count += 1
                        
                        # Print the first few lines to verify data
                        if count <= 3:
                            print(f"Sample output: {line.strip()}")
                            
                    except (ValueError, KeyError) as e:
                        print(f"Error processing row {station_id}: {e}")
                        # Skip rows with invalid coordinates
                        continue
        
        if count > 0:
            print(f"Successfully processed {count} stations")
            print(f"Output saved to: {output_path}")
        else:
            print("Warning: No data was processed. Check your CSV file.")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
# %%
