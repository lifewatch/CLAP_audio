import pandas as pd
import numpy as np
from tqdm import tqdm


# Define the path to the Excel file
file_path = r'C:\Users\wout.decrop\data\DCLDE\metadata\buoydata.xlsx'

# Read the Excel file to get the sheet names
excel_file = pd.ExcelFile(file_path)
sheet_names = excel_file.sheet_names  # List all sheet names

# Initialize lists to store DataFrames for locations and record times
location_dfs = []
record_time_dfs = []

# Loop through each sheet and read it into a DataFrame, categorize based on sheet name
for sheet in sheet_names:
    if "31" not in sheet:
        continue
    if 'locations' in sheet.lower():
        df = pd.read_excel(file_path, sheet_name=sheet)
        location_dfs.append(df)
    elif 'record time' in sheet.lower():
        df = pd.read_excel(file_path, sheet_name=sheet)
        record_time_dfs.append(df)

# Concatenate all location DataFrames and all record time DataFrames
df_locations_combined = pd.concat(location_dfs, ignore_index=True)
df_record_time_combined = pd.concat(record_time_dfs, ignore_index=True)

# Merge the concatenated DataFrames on the common column 'buoy_id'
df_merged = pd.merge(df_locations_combined, df_record_time_combined, on='buoy_id', how='inner')

# Display the first few rows of the merged dataframe
print(df_merged.head())


def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of Earth in kilometers. Use 3956 for miles
    r = 6371
    
    # Calculate the result
    return c * r




# Define the path to the Excel file
file_path = r'C:\Users\wout.decrop\data\DCLDE\metadata\CWI vessel track.xlsx'

# Read the Excel file
dfs = pd.read_excel(file_path, sheet_name="2018-07-31")
# Combine all dataframes into a single dataframe
df= dfs #pd.concat(dfs.values(), ignore_index=True)

# Print the first few rows of the combined dataframe
# print(df_combined.head())
# Display the first few rows of the dataframe
print(df.head())


# Calculate the distance for each row in the merged DataFrame
distance_results = []

# Iterate through each row in the ship DataFrame `df`
# Iterate through each row in the ship DataFrame `df` using tqdm for progress
for idx, ship_row in tqdm(df.iterrows(), total=df.shape[0], desc='Calculating distances for ship locations'):
    ship_lat = ship_row['ship_loc_lat']
    ship_lon = ship_row['ship_loc_lon']

    
    # Calculate the distance to each buoy in `df_merged`
    for buoy_idx, buoy_row in df_merged.iterrows():
        buoy_lat = buoy_row['buoy_lat']
        buoy_lon = buoy_row['buoy_lon']
        distance = haversine(ship_lon, ship_lat, buoy_lon, buoy_lat)
        if distance>11:
            continue
        # Append the result to the list
        distance_results.append({
            'ship_loc_time_utc': ship_row['ship_loc_time_utc'],
            'ship_loc_lon': ship_lon,
            'ship_loc_lat': ship_lat,
            'buoy_id': buoy_row['buoy_id'],
            'buoy_time_utc': buoy_row['buoy_time_utc'],
            'buoy_lon': buoy_lon,
            'buoy_lat': buoy_lat,
            'distance_km': distance
        })
    # break

# Convert the results list to a DataFrame
df_distances = pd.DataFrame(distance_results)

# Display the first few rows of the distance DataFrame
print(df_distances.head())

df_distances = df_distances[df_distances['distance_km'] < 11]


file_path = r'C:\Users\wout.decrop\environments\Imagine\imagine_environment\CLAP_audio\DCLDE\filtered_distances_31.csv'

# Save the filtered DataFrame to a CSV file
df_distances.to_csv(file_path, index=False)

print(f"Filtered DataFrame saved to {file_path}")



