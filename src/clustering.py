# this script contains all functions used for clustering of the data and preparing the data for the optimisation model

# import libraries
import os

import sys
from termcolor import colored, cprint
import numpy as np
import pandas as pd
import geopandas as gpd
#import contextily as ctx
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Point, Polygon, LineString
from scipy.spatial import Delaunay
import networkx as nx
from pyproj import CRS
from sklearn.cluster import KMeans
import json
import time
from data import *


def read_geo_data_from_disk(case_study_name: str, config: dict):
    """
    Read the building data for a given casestudy from the disk and return a geodataframe.

    :param case_study_name: name of the case study
    :param config: configuration dictionary
    :return: geodataframe with building data
    :rtype: geopandas.GeoDataFrame

    """

    # load the data for buildings and drop nan values
    # input: path_building_data: path to the .geojson file with the building data

    cprint("Start: load building for case study: " + case_study_name)

    CaseStudies_dir = config['CaseStudies_dir']
    path_buildings_gdf = os.path.join(CaseStudies_dir,case_study_name, config['building_data']['buildings_gdf'])

    gdf_building_data = gpd.read_file(path_buildings_gdf)

    # remove the entries with missing data
    gdf_building_data = gdf_building_data.dropna()

    gdf_building_data = gdf_building_data.to_crs(epsg=3857)

    cprint("Done: load building data", 'green')

    return gdf_building_data


def read_heatGenUnits_from_disk(case_study_name: str, scenario_name: str, config: dict):
    #load heat generation units
   
    CaseStudies_dir = config['CaseStudies_dir']
    path_heatGenUnits = os.path.join(CaseStudies_dir,case_study_name, config['scenario_dir'], scenario_name, config['heat_source_data']['root_dir'])
    cprint("Start: load heat generation units from: " + path_heatGenUnits)
    df_heat_gen = pd.read_excel(os.path.join(path_heatGenUnits,config['heat_source_data']['heat_sources']), skiprows=[1])

    # add an id to the heat gen units
    #df_heat_gen['heat_unit_id'] = 'heat_unit_' + df_heat_gen.index.astype(str)

    gdf_heat_gen = gpd.GeoDataFrame(df_heat_gen, geometry=gpd.points_from_xy(df_heat_gen.lon, df_heat_gen.lat))
    gdf_heat_gen.crs = CRS('EPSG:4326')
    gdf_heat_gen = gdf_heat_gen.to_crs(epsg=3857)

    cprint("Done: load heat generation units", 'green')

    return gdf_heat_gen

def read_wasteHeatProfiles_from_disk(gdf_heat_gen, case_study_name: str, scenario_name: str, config: dict):
    CaseStudies_dir = config['CaseStudies_dir']
    path_heatGenUnits = os.path.join(CaseStudies_dir,case_study_name, config['scenario_dir'], scenario_name, config['heat_source_data']['root_dir'])
    df_waste_heat_profiles = pd.read_excel(os.path.join(path_heatGenUnits,config['heat_source_data']['heat_profiles']), skiprows=[1])

    WHunits = gdf_heat_gen[gdf_heat_gen['isWH']==1]

    for unit in WHunits['unit']:
        if unit not in df_waste_heat_profiles.columns:
            cprint(('Warning: No wast heat profile found for unit: ', unit) ,'red')
            return None
        else:
            cprint("Done: load waste heat profiles", 'green')
            return df_waste_heat_profiles

def read_building_TS_from_disk(case_study_name: str, config: dict):
    CaseStudies_dir = config['CaseStudies_dir']
    path_building_TS = os.path.join(CaseStudies_dir,case_study_name, config['building_data']['building_TS'])
    df_builidng_TS = pd.read_csv(path_building_TS, sep=',')

    return  df_builidng_TS


def get_centroids(gdf_building_data):
    # calculate the centroids of the buildings
    # input: gdf_building_data: geodataframe with building data
    # output: gdf_building_data: geodataframe with building data and centroids

    cprint("Start: calculate centroids")

    gdf_building_data['centroid'] = gdf_building_data.centroid

    cprint("Done: calculate centroids", 'green')




def cluster_buildings(gdf_building_data, cost_parameter: dict):
    # cluster the buildings intou para_cluster_size number of clusters based on location
    # input: gdf_building_data: geodataframe with building data
    #        parm_cluster_size: number of clusters to be formed
    # output: gdf_clustered_buildings: geodataframe with data for each cluster, in 2D projection

    # load the costs and parameter from the corresponding file
    parm_cluster_size = int(cost_parameter['param_cluster_size'])  
    cost_DH_connect_building = cost_parameter['cost_DH_connect_building']
    cost_DH_connect_area = cost_parameter['cost_DH_connect_area']
    cost_DH_connect_power = cost_parameter['cost_DH_connect_power']  


    cprint("Start: cluster buildings into " + str(parm_cluster_size) + " clusters")

    # calculate the kmeans clustering
    kmeans = KMeans(n_clusters=parm_cluster_size)
    gdf_building_data['cluster'] = kmeans.fit_predict(gdf_building_data['centroid'].apply(lambda x: [x.x, x.y]).tolist())

    # calculate the yearly heating costs per building
    gdf_building_data['YearlyHeatingCosts'] = gdf_building_data['YearlyDemand'] * gdf_building_data['LocalHeatProdCosts'] 

    # copy required data to sum up to a new geodataframe 
    df_clustered_buildings = gdf_building_data[['cluster','MaxDemand','GeneralisedThermCap','GeneralisedThermCond','YearlyHeatingCosts','YearlyDemand','heated_area','number_of_dwellings']].copy()
    #df_clustered_buildings.drop(columns=['geometry','centroid','building_id','year_of_construction', 'building_primary','building_secondary','building_category' ], inplace=True) # streetname as column not included in clustering

  
    # calculate the sum for each cluster
    df_clustered_buildings = df_clustered_buildings.groupby('cluster').sum()

    # overwrite the columns that are not sums
    df_clustered_buildings['geometry'] = gdf_building_data.dissolve(by='cluster').centroid

    # calculate the convex hull for each cluster
    df_clustered_buildings['convex_hull'] = gdf_building_data.dissolve(by='cluster').convex_hull

    # calculate the bounding box for each cluster
    df_clustered_buildings['district_area'] = df_clustered_buildings['convex_hull'].apply(lambda x: x.area)

    # convert the convex hull to a polygon in wkt to have only one geometry column
    df_clustered_buildings['convex_hull'] = df_clustered_buildings['convex_hull'].apply(lambda x: x.wkt)
   
    # add the number of buildings in each cluster
    df_clustered_buildings['NumOfBuildings'] = gdf_building_data.groupby('cluster').size()

    # calculated weighted average costs for the local heating production
    df_clustered_buildings['LocalHeatProdCost'] = df_clustered_buildings['YearlyHeatingCosts'] / df_clustered_buildings['YearlyDemand']
    #df_clustered_buildings.drop(columns=['YearlyHeatingCosts'], inplace=True)

    # create a list with the building_ids for each cluster
    df_clustered_buildings['building_id'] = gdf_building_data.groupby('cluster')['building_id'].agg(list)

    # add an id for the clusters
    df_clustered_buildings['cluster_id'] = 'heat_node_' + df_clustered_buildings.index.astype(str)

    # calculate the maxDHPower for each cluster
    df_clustered_buildings['maxDHPower'] = df_clustered_buildings['MaxDemand'] * 1.1

    # calculate the DH connection costs for each cluster
    df_clustered_buildings['CostsDHconnect'] = df_clustered_buildings['NumOfBuildings'] * cost_DH_connect_building + df_clustered_buildings['district_area'] * cost_DH_connect_area + df_clustered_buildings['maxDHPower'] * cost_DH_connect_power  

    # convert the data to geodataframe
    gdf_clustered_buildings = gpd.GeoDataFrame(df_clustered_buildings, geometry='geometry')
    gdf_clustered_buildings.crs = CRS('EPSG:3857')

    cprint("Done: cluster buildings", 'green')

    return gdf_clustered_buildings


def merge_heatGenUnits(gdf_clustered_buildings, gdf_heatGenUnits):
    # merge the data for the heat generation units with the data for the clusters of buildings
    # input: gdf_clustered_buildings: geodataframe with data for each cluster
    #        gdf_heatGenUnits: geodataframe with data for the heat generation units
    # output: gdf_clustered_buildings: geodataframe with data for each cluster and the heat generation units

    gdf_heatGenUnits['cluster_id'] = 'heat_unit_' + gdf_heatGenUnits['unit'].astype(str)
   
    gdf_heat_nodes = pd.concat([gdf_clustered_buildings, gdf_heatGenUnits.loc[:,['cluster_id','geometry']]], ignore_index=True)

    cprint("Done: merge heat generation units", 'green')

    return gdf_heat_nodes


def cluster_heat_demand(gdf_allheatnodes, df_building_TS):
    # sum up the heating demand time series of the buildings by the calculated clusters
    
    df_cluster_TS = pd.DataFrame()
    df_cluster_TS['hour'] = df_building_TS['hour']

    for cluster_id in gdf_allheatnodes.cluster_id:
        df_cluster_TS[cluster_id] = 0  # Placeholder for actual data
        temp_building_ids = gdf_allheatnodes[gdf_allheatnodes['cluster_id'] == cluster_id]['building_id'].values[0]

        if not temp_building_ids is np.nan:
            df_cluster_TS[cluster_id] = df_building_TS[temp_building_ids].sum(axis=1)
        else:
            df_cluster_TS[cluster_id] = 0  
    
    return df_cluster_TS


def propose_network(gdf_heat_nodes):
    # propose a network of heat pipes between the clusters of buildings based on the delaunay triangulation and minimum spanning tree   
    # input: gdf_clustered_buildings: geodataframe with data for each cluster
    # output: gdf_network: geodataframe with data for the network

    cprint("Start: routing a network between the clusters")

    # calculate the delaunay triangulation
    tri = Delaunay(gdf_heat_nodes['geometry'].apply(lambda x: (x.x, x.y)).tolist())

    # create a dictionary with the node id and index
    node_dict = {idx: node for idx, node in enumerate(gdf_heat_nodes['cluster_id'])}

    def get_edge_distance(node_from,node_to):
        point_from = gdf_heat_nodes[gdf_heat_nodes['cluster_id'] == node_from].geometry
        point_to = gdf_heat_nodes[gdf_heat_nodes['cluster_id'] == node_to].geometry

        #calculate the distance between the points
        distance = round(point_from.iloc[0].distance(point_to.iloc[0]),0)
        return distance

    # create a graph from the delaunay triangulation
    G = nx.Graph()

    # add the nodes to the graph
    G.add_nodes_from(gdf_heat_nodes['cluster_id'])

    # add the edges to the graph
    for simplex in tri.simplices:
        G.add_edge(node_dict[simplex[0]], node_dict[simplex[1]], weight=get_edge_distance(node_dict[simplex[0]],node_dict[simplex[1]]))
        G.add_edge(node_dict[simplex[1]], node_dict[simplex[2]], weight=get_edge_distance(node_dict[simplex[1]],node_dict[simplex[2]]))
        G.add_edge(node_dict[simplex[2]], node_dict[simplex[0]], weight=get_edge_distance(node_dict[simplex[2]],node_dict[simplex[0]]))

    # calculate the minimum spanning tree
    T = nx.minimum_spanning_tree(G)
    #T=G
    
    # convert the graph to a geodataframe
    df_edges = pd.DataFrame(T.edges(data=True), columns=['node_from','node_to','data'])

    # add the geometry to the dataframe
    df_edges['geometry'] = df_edges.apply(lambda x: LineString([gdf_heat_nodes[gdf_heat_nodes['cluster_id'] == x['node_from']].geometry.iloc[0], gdf_heat_nodes[gdf_heat_nodes['cluster_id'] == x['node_to']].geometry.iloc[0]]), axis=1)

    # add a column with the distance of the edge
    df_edges['distance'] = df_edges['data'].apply(lambda x: x['weight'])

    # add the reversed geometry of all edges (needed for the opt model)
    df_reversed_edges = df_edges.copy()
    df_reversed_edges.columns = ['node_to','node_from','data','geometry','distance']
    
    df_heat_network = pd.concat([df_edges, df_reversed_edges], ignore_index=True)

    # delete duplicate edges
    df_heat_network.drop_duplicates(subset=['node_from','node_to'], inplace=True)

    # add a line id 
    df_heat_network['pipe_id'] = 'pipe_' + df_heat_network.index.astype(str)

    # change to geodataframe
    gdf_heat_network = gpd.GeoDataFrame(df_heat_network, geometry='geometry')
    gdf_heat_network.crs = CRS('EPSG:3857')

    cprint("Done: propose network", 'green')

    return gdf_heat_network


def save_data(gdf_heat_nodes,df_cluster_TS,gdf_heat_network,gdf_heat_gen,df_waste_heat_profiles, case_study_name, scenario_name, config): 
    CaseStudies_dir = config['CaseStudies_dir']
    model_data_path = os.path.join(CaseStudies_dir,case_study_name, config['scenario_dir'], scenario_name, config['data_dir'], config['model_data']['root_dir'])
    
    # ckeck if the path exists
    if not os.path.exists(model_data_path):
        os.makedirs(model_data_path)
            
    # save the data for the optimisation model
    cprint("Start: save data for the optimisation model to: "  + model_data_path)
    gdf_heat_nodes.to_file(os.path.join(model_data_path,config['model_data']['heat_nodes']), driver='GeoJSON')
    df_cluster_TS.to_csv(os.path.join(model_data_path,config['model_data']['heat_dem']), index=False)
    gdf_heat_network.to_file(os.path.join(model_data_path,config['model_data']['heat_network']), driver='GeoJSON')
    gdf_heat_gen.rename(columns={'cluster_id':'heat_unit_id'}, inplace=True) # rename the column to unit_id for more clearity in the optimization model
    gdf_heat_gen.to_file(os.path.join(model_data_path,config['model_data']['heat_gen_units']), driver='GeoJSON')
    df_waste_heat_profiles.to_csv(os.path.join(model_data_path,config['model_data']['wh_profiles']), index=False)

    cprint("Done: save data for the optimisation model", 'green')


def add_hot_water_demand(building_cluster, cluster_TS, cost_parameter):
    # add a simplified hot water demand to the heat demand time series
    # input: building_cluster: geodataframe with data for each cluster
    #        cluster_TS: dataframe with the heat demand time series
    # output: cluster_TS: dataframe with the heat demand time series and the hot water demand

    # load the costs and parameter from the corresponding file
    hot_water_demand = cost_parameter['daily_hot_water_demand'] #hot water demand per dwelling per day

    cprint("Start: add hot water demand to the heat demand time series")

    # calculate the hot water demand for each cluster
    building_cluster['YearlyHWD'] = hot_water_demand * building_cluster['number_of_dwellings'] * 356

    # distribute the yearlyHWD to the hourly demand and add it to the existing heat demand
    for cluster_id in cluster_TS.columns[1:]:
        cluster_TS[cluster_id] = cluster_TS[cluster_id] + building_cluster[building_cluster['cluster_id'] == cluster_id]['YearlyHWD'].values[0] / 8760
   
    cprint("Done: add hot water demand", 'green')

    return cluster_TS


# run the whole script
if __name__ == "__main__":
    start = time.time()

    # set here the case study and scenario name
    case_study_name = 'Puertollano_open_data'
    model_name = 'realistic_costs'

    # load the general config file
    config = load_config()

    # generate a new scenario if the scenario does not exist
    generate_new_scenario(case_study_name,model_name,config)

    # prepre the parameter file
    prepare_parameter_file(case_study_name, model_name, config)

    # load prameter and costs as dictionary
    cost_parameter = load_cost_parameter(case_study_name, model_name, config)

    # load the required data sets
    buildings = read_geo_data_from_disk(case_study_name, config)
    buildings_TS = read_building_TS_from_disk(case_study_name, config)
    generation_units = read_heatGenUnits_from_disk(case_study_name, model_name, config)
    waste_heat_profiles = read_wasteHeatProfiles_from_disk(generation_units, case_study_name, model_name, config)
    
    # cluster the buidlings 
    get_centroids(buildings) # calculate the centroids of the buildings
    building_cluster = cluster_buildings(buildings, cost_parameter)

    # cluster the heat demand time series
    cluster_TS = cluster_heat_demand(building_cluster, buildings_TS)

    # add hot water demand to the heat demand time series
    cluster_TS = add_hot_water_demand(building_cluster, cluster_TS, cost_parameter)

    # merge clusters with generion units
    all_heat_nodes = merge_heatGenUnits(building_cluster, generation_units)

    # create a network
    network = propose_network(all_heat_nodes)

    # save the data to the scenario folder for the optimisation model
    save_data(all_heat_nodes,cluster_TS,network,generation_units,waste_heat_profiles, case_study_name, model_name, config)

    elapsed = time.time() - start
    cprint("Elapsed time: " + str(elapsed) + " seconds", 'green')
    cprint("Done: clustering and preparing data for the optimisation model", 'green')