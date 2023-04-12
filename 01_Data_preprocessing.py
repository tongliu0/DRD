import osmnx as ox
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ast
from functools import partial, reduce

### Part 1. Load and Merge the Data

def get_shapefile_data(file_path):
    # open file and reset crs (in meter units)
    sf = gpd.read_file(file_path)
    # get the centriod for all grid cells
    sf['centroid'] = sf['geometry'].centroid
    sf['longitude'] = sf['centroid'].x
    sf['latitude'] = sf['centroid'].y
    data = sf.drop(columns = ['geometry', 'centroid'])
    print(sf.crs)
    return data

def recal_centroid(df):
    df['longitude'] = (df['left'] + df['right'])/2
    df['latitude'] = (df['top'] + df['bottom'])/2
    return df

# Load LULC data
LULC1 = get_shapefile_data("Data/Shape_files/ESA_LULC_Phoenix_01_trees.shp")
LULC2 = get_shapefile_data("Data/Shape_files/ESA_LULC_Phoenix_02_shrubland.shp")
LULC3 = get_shapefile_data("Data/Shape_files/ESA_LULC_Phoenix_03_grassland.shp")
LULC4 = get_shapefile_data("Data/Shape_files/ESA_LULC_Phoenix_05_builtup.shp")
LULC5 = get_shapefile_data("Data/Shape_files/ESA_LULC_Phoenix_06_barren.shp")
LULC6 = get_shapefile_data("Data/Shape_files/ESA_LULC_Phoenix_08_openwater.shp")
LULC7 = get_shapefile_data("Data/Shape_files/ESA_LULC_Phoenix_04_cropland.shp")
# Load Slope/Aspect data
slope = get_shapefile_data("Data/Shape_files/USGS_DEM_Phoenix_slope.shp")
aspect = get_shapefile_data("Data/Shape_files/USGS_DEM_Phoenix_aspect.shp")
elevation = get_shapefile_data("Data/Shape_files/USGS_DEM_Phoenix_elevation.shp")
# Load LST data
LST = get_shapefile_data("Data/Shape_files/Landsat8_Phoenix_LST.shp")
# Load Building/Street/POI data
building = get_shapefile_data("Data/Shape_files/OSM_Phoenix_BuildingDensity.shp")
street = get_shapefile_data("Data/Shape_files/OSM_Phoenix_StreetGridDensity.shp")
poi = get_shapefile_data("Data/Shape_files/SafeGraph_Phoenix_POI_Density.shp")

# recalculate centriod for geometry CRS EPSG 4326
building = recal_centroid(building)
street = recal_centroid(street)
poi = recal_centroid(poi)

list1 = [LST,LULC1,LULC2,LULC3,LULC4,LULC5,LULC6,LULC7,slope,aspect,elevation]
merge = partial(pd.merge, on=['longitude','latitude'], how='outer')
df1 = reduce(merge, list1)
df1['lon_id'] = df1['longitude'].rank(method='dense',ascending=True)
df1['lat_id'] = df1['latitude'].rank(method='dense',ascending=False)
df1 = df1.loc[(df1['lon_id']<=83) & (df1['lat_id']<=156)]

list2 = [building,street,poi]
merge = partial(pd.merge, on=['longitude','latitude','top','bottom','left','right','id'], how='outer')
df2 = reduce(merge, list2)
df2['lon_id'] = df2['longitude'].rank(method='dense',ascending=True)
df2['lat_id'] = df2['latitude'].rank(method='dense',ascending=False)
df2 = df2[['lon_id','lat_id','poi_num','bldg_area','nodes_num']]

# merge all data
list3 = [df2,df1]
merge = partial(pd.merge, on=['lon_id','lat_id'], how='outer')
df = reduce(merge, list3)

# Classify the aspect
def aspect_class(aspect_value):
  aspect = []
  for i in range(len(aspect_value)):
    x = aspect_value[i]
    if x == -1:
      aspect.append("Flat")
    elif x >= 0 and x < 22.5:
      aspect.append("North")
    elif x >= 22.5 and x < 67.5:
      aspect.append("Northeast")
    elif x >= 67.5 and x < 112.5:
      aspect.append("East")
    elif x >= 112.5 and x < 157.5:
      aspect.append("Southeast")
    elif x >= 157.5 and x < 202.5:
      aspect.append("South")
    elif x >= 202.5 and x < 247.5:
      aspect.append("Southwest")
    elif x >= 247.5 and x < 292.5:
      aspect.append("West")
    elif x >= 292.5 and x < 337.5:
      aspect.append("Northwest")
    elif x >= 337.5 and x <= 360:
      aspect.append("North")
  return aspect
df['aspect_class'] = aspect_class(df['aspect'])

### Part 2. Normalization

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore')
# Encode the Aspect column
aspect_class = pd.DataFrame(encoder.fit_transform(df[['Aspect']]).toarray())
print(encoder.categories_)
aspect_class.columns = ['East', 'North', 'Northeast', 'Northwest', 'South', 'Southeast','Southwest', 'West']
df = pd.concat([df, aspect_class], axis=1)

# Get the features to scale
feature_to_log = df.loc[:,('Elevation','bldg_area','nodes_num','poi_num')].values

# Log transformation
feature_log = np.log(feature_to_log+1)

# Get other features (features not needed to be transformed)
feature_others = df.loc[:,('tree','shrubland','grassland','builtup',
                          'openwater','slope','East', 'North', 'Northeast', 'Northwest',
                          'South', 'Southeast','Southwest', 'West')].values

# Concatenate the matix along columns
feature_matrix = np.concatenate((feature_others,feature_log), axis=1)
print(feature_matrix.shape)
print(feature_matrix[0])# Extract the Edge Features

# Generte the adjacent matrix
# Remap the grid cell id, starting from 0
df['id'] = df.reset_index().index
arr = df.loc[:, ('lat_id','lon_id')].values.reshape(lat_cell_num*lon_cell_num,2)

def create_adj_matrix(arr):
  adj_matrix = np.empty(shape=(lat_cell_num*lon_cell_num, lat_cell_num*lon_cell_num, 1), dtype=np.int32)
  for i in range(5016):
    for j in range(5016):
      lat_1 = arr[i][0]
      lon_1 = arr[i][1]
      lat_2 = arr[j][0]
      lon_2 = arr[j][1]
      euc_dist = (lat_2 - lat_1)**2 + (lon_2 - lon_1)**2
      if euc_dist <= 1:    # <4 if one node is linked to 8 nodes
        adj_matrix[i][j] = 1
      else:
        adj_matrix[i][j] = 0
  return adj_matrix

adj_matrix = create_adj_matrix(arr)

# Convert edge index(adjacent matrix) to source node and target node
# (this step for StellarGraph input: "line")
edge_index = adj_matrix.reshape((lat_cell_num*lon_cell_num, lat_cell_num*lon_cell_num))

def generate_edge_from_index(edge_index):
  source = []
  target = []
  all_nodes = np.array(list(range(5016)))
  for i in all_nodes:
    left_nodes = np.delete(all_nodes, i)
    for j in left_nodes:
      if edge_index[i][j] == 1 and i < j:
        source.append(i)
        target.append(j)
  return source, target

sources, targets = generate_edge_from_index(edge_index)

input_edges = pd.DataFrame({'source': sources, 'target': targets}, dtype='int64')

delta = []
def get_edge_weights(input_edges, delta_x, delta_y):
  for i in range(len(input_edges)):
    source_id = input_edges['source'][i]
    target_id = input_edges['target'][i]
    if abs(source_id - target_id) == 1:
      delta.append((delta_x[source_id] - delta_x[target_id])*(-1))
    else:
       delta.append(delta_y[source_id] - delta_y[target_id])
  return delta

degree = get_edge_weights(input_edges, df['delta_x'].values, df['delta_y'].values)

input_edges['delta'] = degree
input_edges['weight'] = (1 - input_edges['delta']).round(4)
input_edges = input_edges.drop(columns=['delta'])

# Save the data
input_edges.to_csv('Data/Graph_EdgewithWeights_grids_Phoenix.csv')
np.savetxt('Data/Graph_NodeFeatures_grids_Phoenix.csv', feature_matrix, delimiter=',')
