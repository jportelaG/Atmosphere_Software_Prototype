
import os

import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
from shapely.geometry.linestring import LineString
from shapely import wkt

from data import *


def extract_node_centroids(model_input_data):
    # extract the centroids of the nodes from the input data
    gdf_heat_nodes = model_input_data['heat_nodes']
    gdf_heat_nodes.crs = 'EPSG:3857'
    # change crs
    gdf_heat_nodes = gdf_heat_nodes.to_crs('EPSG:4326')

    # creat a dictionary with node names as keys and centroids as values
    node_centroids = {}
    for index, row in gdf_heat_nodes.iterrows():
        node_centroids[row['cluster_id']] = row['geometry']

    return node_centroids

def extract_node_shapes(model_input_data):
    # extract the shapes of the nodes from the input data
    gdf_heat_nodes = model_input_data['heat_nodes']
    gdf_heat_nodes.crs = 'EPSG:3857'

    # change crs
    gdf_heat_nodes = gdf_heat_nodes.to_crs('EPSG:4326')

    # creat a dictionary with node names as keys and centroids as values
    node_shapes = {}
    for index, row in gdf_heat_nodes.iterrows():
        node_shapes[row['cluster_id']] = wkt.loads(row['convex_hull'])

    # change the crs of the shapes
    node_shapes_gdf = gpd.GeoDataFrame(geometry=list(node_shapes.values()), crs='EPSG:3857')
    node_shapes_gdf = node_shapes_gdf.to_crs('EPSG:4326')
    node_shapes = dict(zip(node_shapes.keys(), node_shapes_gdf.geometry))

    return node_shapes



def plot_investment_decisions(model_input_data, model_output_data, dict_nodes, dict_shapes, figure_path):
    ## shapes of the clusters
    # extract the investment decisions from the output data
    df_DHconnect = model_output_data['vDHconnect'].copy()
    df_DHconnect.rename(columns={'index_set_1': 'node_id'}, inplace=True)
    # set node as index
    df_DHconnect.set_index('node_id', inplace=True)

    # merge the coordinates to the nodes
    df_DHconnect['geometry'] = df_DHconnect.index.map(dict_shapes)

    gdf_DHconnect = gpd.GeoDataFrame(df_DHconnect, geometry=df_DHconnect['geometry'])

    ## pipes 
    # extract the investment decisions from the output data
    df_BuildPipe = model_output_data['vBinBuildPipe'].copy()
    df_BuildPipe.rename(columns={'value': 'invest'}, inplace=True)
    df_PipeMFInf = model_output_data['vPipeMassFlowInv'].copy()
    df_PipeMFInf.rename(columns={'value': 'mass_flow'}, inplace=True)
    # merge the mass flow to the investment decisions by the index set 1 and index set 2
    df_BuildPipe = df_BuildPipe.merge(df_PipeMFInf, on=['index_set_1', 'index_set_2'], how='left')

    #rename columns    
    df_BuildPipe.rename(columns={'index_set_1': 'node_from', 'index_set_2': 'node_to'}, inplace=True)

    # match the from the to nodes with the coordinates and create linestrings
    df_BuildPipe['from_geometry'] = df_BuildPipe['node_from'].map(dict_nodes)
    df_BuildPipe['to_geometry'] = df_BuildPipe['node_to'].map(dict_nodes)

    # create linestrings
    df_BuildPipe['geometry'] = df_BuildPipe.apply(lambda x: LineString([x['from_geometry'], x['to_geometry']]), axis=1)

    # create a geodataframe with the investment decisions
    gdf_BuildPipe = gpd.GeoDataFrame(df_BuildPipe, geometry=df_BuildPipe['geometry'])

    # load the pipe candidates
    gdf_pipe_candidates = model_input_data['heat_network'].copy()
    gdf_pipe_candidates.crs = 'EPSG:3857'
    # change crs
    gdf_pipe_candidates = gdf_pipe_candidates.to_crs('EPSG:4326')   


    ## plot the investment decisions
    fig, ax = plt.subplots(figsize=(11, 8))

    # add a title
    plt.title('Investment decisions', fontsize=20)

    # plot the nodes
    for node_id, centroid in dict_nodes.items():
        plt.plot(centroid.x, centroid.y, 'grey', marker='o', markersize=3, zorder=4)


    gdf_DHconnect[gdf_DHconnect['value'] == 1].plot(ax=ax, color='bisque', edgecolor='k', markersize=100, alpha=0.75, zorder=1)
    gdf_DHconnect[gdf_DHconnect['value'] == 0].plot(ax=ax, color='grey', edgecolor='k', markersize=100, alpha=0.2, zorder=1)

    #plot pipe candidates
    gdf_pipe_candidates.plot(ax=ax, color='grey', alpha=1, linewidth=1, zorder=2)

    linewidth_scale = 0.0050

    # plot the pipes
    gdf_BuildPipe.plot(ax=ax, color='crimson', linewidth=linewidth_scale*gdf_BuildPipe['mass_flow'], zorder=3)


    # add a textlable below the legend
    height = 0.70
    ax.text(0.85, height, '1000 m³/h', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.plot([0.91, 0.98], [height, height], 'r-', linewidth=1000*linewidth_scale, transform=ax.transAxes)


    # add a maker for the heat generation units
    gdf_heat_gen_units = model_input_data['heat_gen_units'].copy()
    gdf_heat_gen_units.crs = 'EPSG:3857'
    # change crs
    gdf_heat_gen_units = gdf_heat_gen_units.to_crs('EPSG:4326')

    #load the investment decisions
    df_heat_investment = model_output_data['vCentralHeatProdInv'].copy()
    df_heat_investment.rename(columns={'index_set_1': 'unit', 'value': 'power_investment'}, inplace=True)

    # merge the investment decisions to the heat generation units
    gdf_heat_gen_units = gdf_heat_gen_units.merge(df_heat_investment, left_on='unit', right_on='unit', how='left')

    # distinguish between the types of heat generation units
    gdf_heat_gen_units['alpha'] = np.where(gdf_heat_gen_units['power_investment'] > 0.0, 0.9, 0.3)


    gdf_heat_gen_units[gdf_heat_gen_units['isWH']==1].plot(ax=ax, color='darkgreen', marker='s', edgecolor='k', markersize=200, alpha=gdf_heat_gen_units.loc[gdf_heat_gen_units['isWH']==1,'alpha'], zorder=5)
    gdf_heat_gen_units[gdf_heat_gen_units['isBoiler']==1].plot(ax=ax, color='gold', marker='s', edgecolor='k', markersize=200, alpha=gdf_heat_gen_units.loc[gdf_heat_gen_units['isBoiler']==1,'alpha'], zorder=5)
    #gdf_heat_gen_units[gdf_heat_gen_units['isTES']==1].plot(ax=ax, color='slateblue', marker='s', edgecolor='k', markersize=200, alpha=gdf_heat_gen_units.loc[gdf_heat_gen_units['isTES']==1,'alpha'], zorder=5)
    
    # add a text to the heat generation units where investments are made
    for index, row in gdf_heat_gen_units.iterrows():
        if row['power_investment'] > 0.0:
            plt.text(row['geometry'].x + 0.0000, row['geometry'].y + 0.0015, f"{row['power_investment']/1e3:.0f} MW", fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))


    # add a legend with marker symbols
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='bisque', marker='o', markeredgecolor='k', markersize=8, linestyle='None'),
                    Line2D([0], [0], color='grey', marker='o', markeredgecolor='k', markersize=8, linestyle='None'),
                    Line2D([0], [0], color='darkgreen', marker='s', markeredgecolor='k', markersize=8, linestyle='None'),
                    Line2D([0], [0], color='gold', marker='s', markeredgecolor='k', markersize=8, linestyle='None'),
                    Line2D([0], [0], color='slateblue', marker='s', markeredgecolor='k', markersize=8, linestyle='None'),                    Line2D([0], [0], color='red', lw=4)]
    ax.legend(custom_lines, ['Connected', 'Not connected', 'Waste Heat Unit', 'Boiler', 'TES', 'Pipe'])
    
    
    # add a basemap
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Positron)

    # show the plot
    plt.show()

    # check if the figure path exists
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    # save the plot as pdf
    fig.savefig(os.path.join(figure_path, 'investment_decisions.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(figure_path, 'investment_decisions.jpg'), bbox_inches='tight')
    
    return gdf_BuildPipe


def merge_time_series(model_input_data, model_output_data):
    df_heat_demand = model_input_data['heat_demand'].copy()

    # calc the sum over the heat nodes (Columns) excluding the 'hour' column
    df_heat_demand['heat_demand'] = df_heat_demand.drop(columns=['hour']).sum(axis=1)

    df_local_heat_production = model_output_data['vLocalHeatProd'].copy()
    df_local_heat_production.rename(columns={'index_set_1': 'node', 'index_set_2': 'hour', 'value': 'local_heat_gen'}, inplace=True)
    # regroup by the same hour and sum over all heat nodes
    df_local_heat_production = df_local_heat_production.groupby('hour').sum()
    df_local_heat_production.drop(columns='node', inplace=True)

    df_central_heat_production = model_output_data['vCentralHeatProd'].copy()
    df_central_heat_production.rename(columns={'index_set_1': 'unit', 'index_set_2': 'hour', 'value': 'power'}, inplace=True)
    # reshape and make the units to columns
    df_central_heat_production = df_central_heat_production.pivot(index='hour', columns='unit', values='power')
    #rename all columns with a prefix 'gen_'
    df_central_heat_production.rename(columns={col: f'gen_{col}' for col in df_central_heat_production.columns}, inplace=True)

    # load the potential waste heat from the heat generation units
    df_heat_gen_units = model_input_data['waste_heat_prof'].copy()
    # recalc all columns to kW
    df_heat_gen_units = df_heat_gen_units * 1.16389 * (model_input_data['parameter_cost']['pTsupply'] -  model_input_data['parameter_cost']['pTreturn'])
    
    #rename all columns with a prefix 'wh_'
    df_heat_gen_units.rename(columns={col: f'wh_{col}' for col in df_heat_gen_units.columns}, inplace=True)

    # merge all dataframes togehther by hour
    df_energy_balance = df_local_heat_production.merge(df_heat_demand.loc[:,'heat_demand'], left_index=True, right_index=True)
    df_energy_balance = df_energy_balance.merge(df_central_heat_production, left_index=True, right_index=True)
    df_energy_balance = df_energy_balance.merge(df_heat_gen_units, left_index=True, right_index=True)
    df_energy_balance.drop(columns='wh_hour', inplace=True)

    # the index now iterates the hours, generate a time index out of it sarting with 2019-01-01 00:00:00
    df_energy_balance.index = pd.date_range(start='2019-01-01 00:00:00', periods=len(df_energy_balance), freq='h')

    return df_energy_balance

def plot_energy_balance(df_energy_balance, figure_path):

    df_yearly_sums = df_energy_balance.resample('YE').sum()

     # make two subplots
    fig, ax = plt.subplots(1, 2, figsize=(11, 8))
    # plot a pie-chart for the first subplot
    labels = ['Decentral Heat Production', 'Electric Boiler', 'Waste Heat Utilisation']
    values = df_yearly_sums[['local_heat_gen', 'gen_ElectricBoiler', 'gen_Electrolyser']].values.flatten()
    total = sum(values)
    explode = (0, 0.05, 0)
    colors = ['dimgrey', 'dodgerblue', 'firebrick']

    ax[0].pie(values, explode=explode, labels=None,
                autopct=lambda p: '{:.1f}'.format(p * total / 100 / 1e6) + ' GWh',
                shadow=False, startangle=90, colors=colors,
                textprops=dict(bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')))
    ax[0].set_title('Total Heat Production' + '\n' + 'Total Yealy Demand = ' + f'{df_yearly_sums["heat_demand"].values[0]/1e6:.0f} GWh')
    # add a legend
    ax[0].legend(labels, loc='center', bbox_to_anchor=(0, 0))


    # make the second subplot with the same chart but plot waste heat utilisation
    labels = ['Excess Waste Heat', 'Utilised Waste Heat']
    values = np.concatenate([(df_yearly_sums['wh_Electrolyser'] - df_yearly_sums['gen_Electrolyser']).values.flatten(), df_yearly_sums['gen_Electrolyser'].values.flatten()])
    total = sum(values)
    explode = (0, 0.05)
    colors = ['olivedrab', 'firebrick']

    ax[1].pie(values, explode=explode, labels=None,
                autopct=lambda p: '{:.1f}'.format(p * total / 100 / 1e6) + ' GWh',
                shadow=False, startangle=90, colors=colors,
                textprops=dict(bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')))
    
    ax[1].set_title('Total Waste Heat Utilisation' + '\n' + 'Total Yealy Waste Heat = ' + f'{df_yearly_sums["wh_Electrolyser"].sum()/1e6:.0f} GWh')
    # add a legend
    ax[1].legend(labels, loc='center', bbox_to_anchor=(0,0))

    # show the plot
    plt.show()

    # save the plot as pdf
    fig.savefig(os.path.join(figure_path, 'energy_balance.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(figure_path, 'energy_balance.jpg'), bbox_inches='tight')

    return None


def plot_time_resolved(df_energy_balance, figure_path, time_invervall='W'):
    # resmaple 
    loc_df_energy_balance = df_energy_balance.resample(time_invervall).sum()/1e3

    # make a plot
    fig, ax = plt.subplots(figsize=(11, 8))

    # plot the heat demand
    ax.plot(loc_df_energy_balance.index, loc_df_energy_balance['heat_demand'], color='black', linewidth=1, label='Heat Demand')
    
    #plot the available waste heat
    ax.plot(loc_df_energy_balance.index, loc_df_energy_balance['wh_Electrolyser'], color='olivedrab', linewidth=1, label='Waste Heat')

    # make a stacked plot for the heat generation
    ax.stackplot(loc_df_energy_balance.index, loc_df_energy_balance['local_heat_gen'], loc_df_energy_balance['gen_ElectricBoiler'], loc_df_energy_balance['gen_Electrolyser'], labels=['Decentral Heat Production', 'Electric Boiler', 'Waste Heat Utilisation'], colors=['dimgrey', 'dodgerblue', 'firebrick'], alpha=0.7)

    # color the excess waste heat
    ax.fill_between(loc_df_energy_balance.index, loc_df_energy_balance['wh_Electrolyser'], loc_df_energy_balance['heat_demand'], color='olivedrab', alpha=0.3, label='Excess Waste Heat')

    # add a legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # add a title
    plt.title('Time resolved energy balance' + '\n' + f'Time intervall = {time_invervall}', fontsize=12)

    # add x and y labels
    plt.xlabel('Time')
    plt.ylabel('Power in MWh/time invervall')

    # show the plot
    plt.show()

    # save the plot as pdf
    fig.savefig(os.path.join(figure_path, str(time_invervall) + '_time_resolved_energy_balance.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(figure_path, str(time_invervall) + '_time_resolved_energy_balance.jpg'), bbox_inches='tight')

    
    return None

# make a plot of the buildings
def plot_HD_interactive(gdf_buildings: gpd.GeoDataFrame):
    """Generates an interactive map of the buildings with their yearly heat demand.

    :param gdf_buildings: geodataframe with the buildings and their yearly heat demand
    :type gdf_buildings: gpd.GeoDataFrame
    :return: an interactive map of the buildings with their yearly heat demand
    :rtype: folium.Map
    """

    gdf_buildings['YearlyDemand_MWh'] = gdf_buildings['YearlyDemand'] / 1000
    m = gdf_buildings.explore(
    column='YearlyDemand_MWh',
    scheme='fisherjenks',
    k = 10,
    cmap='coolwarm',
    legend=True,
    name='Buildings',
    tiles='CartoDB positron',
    zoom_start=14,
    fit_bounds=True,
    style_kwds={'fillOpacity': 0.7, 'color': 'black', 'weight': 0.5},
    width=800, height=600,
    control_scale=True,
    legend_kwds={
        'caption': 'Yearly Heat Demand in MWh/building',
        'caption_font_size': '12px',
        'fmt': '{:.0f}'
    }
    )
    return m

if __name__ == '__main__':
    case_study_name = 'Puertollano_open_data'
    model_name = 'hotwater_01'

    config = load_config()
    
    figure_path = os.path.join(case_study_name, config['scenario_dir'], model_name, config['plot_dir'])

    model_output_data = read_output_from_disk(case_study_name, model_name, config)
    model_input_data = load_data_from_disk(case_study_name, model_name, config)

    dict_nodes = extract_node_centroids(model_input_data)
    dict_shapes = extract_node_shapes(model_input_data)

    print(dict_nodes)

    plot_investment_decisions(model_input_data, model_output_data, dict_nodes, dict_shapes, figure_path)
    plot_energy_balance(merge_time_series(model_input_data, model_output_data), figure_path)
    plot_time_resolved(merge_time_series(model_input_data, model_output_data), figure_path, time_invervall='W')
    

    #plot_node_properties(model_input_data)

    

