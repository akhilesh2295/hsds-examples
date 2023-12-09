# %%
# %matplotlib inline
import h5pyd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.spatial import cKDTree

# %%
f = h5pyd.File("/nrel/nsrdb/v3/tmy/nsrdb_tmy-2020.h5", 'r') #nrel-pds-nsrdb/v3/tmy

# %%
list(f)
# f.attrs['version']   # attributes can be used to provide desriptions of the content

# %%
# Datasets are stored in a 2d array of time x location
dset = f['wind_speed']
dset.shape

# %%
# Extract datetime index for datasets
time_index = pd.to_datetime(f['time_index'][...].astype(str))
time_index # Temporal resolution is 60min

# %%
# Locational information is stored in either 'meta' or 'coordinates'
meta = pd.DataFrame(f['meta'][...])
meta.head()

# %%
dset.attrs['psm_scale_factor'] # Irradiance values have been truncated to integer precision

# %%
import os
os.getcwd()


# %%
# Get lat lon coordinates for counties in Texas from Excel file

df = pd.read_csv('../datasets/Texas_Counties_lat_lon_data.csv')


# %%
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Assuming you have already defined your DataFrame 'df' and the necessary functions and variables

dset_coords = f['coordinates'][...]
tree = cKDTree(dset_coords)
def nearest_site(tree, lat_coord, lon_coord):
    lat_lon = np.array([lat_coord, lon_coord])
    dist, pos = tree.query(lat_lon)
    return pos

# Create an empty list to store the nearest point indices
nearest_indices = []

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Get the latitude and longitude coordinates from the current row
    lat_coord = row['Latitude']
    lon_coord = row['Longitude']
    
    # Find the nearest point index using the 'nearest_site' function and the tree
    nearest_index = nearest_site(tree, lat_coord, lon_coord)
    
    # Add the nearest point index to the list
    nearest_indices.append(nearest_index)

# Add the nearest indices as a new column in the DataFrame
df['NearestIndex'] = nearest_indices


# %%
df['NearestIndex']

# %%
tseries = dset[:, df['NearestIndex'][0]] / dset.attrs['psm_scale_factor']

# %%
tseries

# %%
def calculate_tseries(row):
    tseries = dset[:, row['NearestIndex']] / dset.attrs['psm_scale_factor']
    return tseries

# Apply the function to create the new column
df['tseries'] = df.apply(calculate_tseries, axis=1) 

# %%
df.to_pickle("bad_TX_county_data_df_wind.pkl")


# %%
import ast
import pandas as pd

df = pd.read_pickle("bad_TX_county_data_df_wind.pkl")
# df['list'] = df['ghi'].apply(lambda x: ast.literal_eval(x.split(":")[1]))

# %%
# Unlike the gridded WTK data the NSRDB is provided as sparse time-series dataset.
# The quickest way to find the nearest site it using a KDtree

# dset_coords = f['coordinates'][...]
# tree = cKDTree(dset_coords)
# def nearest_site(tree, lat_coord, lon_coord):
#     lat_lon = np.array([lat_coord, lon_coord])
#     dist, pos = tree.query(lat_lon)
#     return pos

# NewYorkCity = (40.7128, -74.0059)
# NewYorkCity_idx = nearest_site(tree, NewYorkCity[0], NewYorkCity[1] )

# print("Site index for New York City: \t\t {}".format(NewYorkCity_idx))
# print("Coordinates of New York City: \t {}".format(NewYorkCity))
# print("Coordinates of nearest point: \t {}".format(dset_coords[NewYorkCity_idx]))

# %%
# nearest_indices = []

# # Iterate over each row in the DataFrame
# for index, row in df.iterrows():
#     # Get the latitude and longitude coordinates from the current row
#     lat_coord = row['Latitude']
#     lon_coord = row['Longitude']
    
#     # Find the nearest point index using the 'nearest_site' function and the tree
#     nearest_index = nearest_site(tree, lat_coord, lon_coord)
    
#     # Add the nearest point index to the list
#     nearest_indices.append(nearest_index)

# # Add the nearest indices as a new column in the DataFrame
# df['NearestIndex'] = nearest_indices

# %%
# Get the entire timeseries data for a point in NYC
# tseries = dset[:, NewYorkCity_idx] / dset.attrs['psm_scale_factor']

# # %%
# plt.plot(time_index, tseries)
# plt.ylabel("ghi")
# plt.title("NYC ghi in 2012")

# %%



