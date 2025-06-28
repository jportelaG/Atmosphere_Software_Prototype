
import geopandas as gpd
import pandas as pd
import os
import sys
import yaml
import json
import shutil




def load_data_from_disk(casestudy: str, scenario: str, config: dict):
    # Load data from disk based on the provided case study and scenario
    # This function assumes that the scenario is in the 'CaseStudies' directory
    
    CaseStudies_dir = config['CaseStudies_dir']
    scenario_dir = config['scenario_dir']
    data_dir = config['data_dir']
    heat_root_dir = config['model_data']['root_dir']

    data_path = os.path.join(CaseStudies_dir,casestudy, scenario_dir, scenario, data_dir, heat_root_dir)

    print("Loading data from: ", data_path)

    df_heat_demand = gdf_heat_gen_units = gdf_heat_network = gdf_heat_nodes = df_waste_heat_prof = None

    if os.path.exists(os.path.join(data_path, config['model_data']['heat_dem'])):
        df_heat_demand = pd.read_csv(os.path.join(data_path, config['model_data']['heat_dem']), sep=',')

    if os.path.exists(os.path.join(data_path, config['model_data']['heat_gen_units'])):
        gdf_heat_gen_units = gpd.read_file(os.path.join(data_path, config['model_data']['heat_gen_units']))

    if os.path.exists(os.path.join(data_path, config['model_data']['heat_network'])):
        gdf_heat_network = gpd.read_file(os.path.join(data_path, config['model_data']['heat_network']))

    if os.path.exists(os.path.join(data_path, config['model_data']['heat_nodes'])):
        gdf_heat_nodes = gpd.read_file(os.path.join(data_path, config['model_data']['heat_nodes']))

    if os.path.exists(os.path.join(data_path, config['model_data']['wh_profiles'])):
        df_waste_heat_prof = pd.read_csv(os.path.join(data_path, config['model_data']['wh_profiles']), sep=',')

    with open(os.path.join(data_path, config['model_data']['cost_parameter'])) as json_file:
        dict_parameter_cost = json.load(json_file)



    # merge them into a dictionary
    model_input_data = {
        'heat_demand': df_heat_demand,
        'heat_gen_units': gdf_heat_gen_units,
        'heat_network': gdf_heat_network,
        'heat_nodes': gdf_heat_nodes,
        'waste_heat_prof': df_waste_heat_prof,
        'parameter_cost': dict_parameter_cost
    }

    return model_input_data


def load_config():
    with open('_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_cost_parameter(casestudy: str, scenario: str, config: dict):
    CaseStudies_dir = config['CaseStudies_dir']
    scenario_dir = config['scenario_dir']
    data_dir = config['data_dir']
    heat_root_dir = config['model_data']['root_dir']

    data_path = os.path.join(CaseStudies_dir,casestudy, scenario_dir, scenario, data_dir, heat_root_dir)
    with open(os.path.join(data_path, config['model_data']['cost_parameter'])) as json_file:
        dict_parameter_cost = json.load(json_file)

    return dict_parameter_cost


# a test dataset can be found under: M:\Institut\04_PROJEKTE\Atmosphere\01_Daten\testdataset
# copy that and set the scenario_path to that directory
def load_model_data(scenario_path): ## not used anymore
    cprint("Start: loading data from: " + scenario_path)
    # Load the data if the coresponding dataframes are not in the workspace allready
    subfolder = 'model_input_data'

    assert os.path.exists(os.path.join(scenario_path,subfolder, 'Heat_Demand.csv')), 'Heat_Demand.csv not found!'
    df_heat_demand = pd.read_csv(os.path.join(scenario_path,subfolder, 'Heat_Demand.csv'))

    assert os.path.exists(os.path.join(scenario_path, subfolder, 'Heat_Generation_Units.geojson')), \
        'Heat_Generation_Units.geojson not found!'
    gdf_heat_gen_units = gpd.read_file(os.path.join(scenario_path,subfolder, 'Heat_Generation_Units.geojson'))

    assert os.path.exists(os.path.join(scenario_path, subfolder, 'Heat_Network.geojson')), \
        'Heat_Network.geojson not found!'
    gdf_heat_network = gpd.read_file(os.path.join(scenario_path,subfolder, 'Heat_Network.geojson'), sep=';')

    assert os.path.exists(os.path.join(scenario_path, subfolder, 'Heat_Nodes.csv')), \
        'Heat_Nodes.geojson not found!'
    gdf_heat_nodes = gpd.read_file(os.path.join(scenario_path,subfolder, 'Heat_Nodes.geojson'))

    if os.path.exists(os.path.join(scenario_path, subfolder, 'Heat_WH_Profiles.csv')):
        df_waste_heat_prof = pd.read_csv(os.path.join(scenario_path, subfolder, 'Heat_WH_Profiles.csv'), sep=';')

#add here for the other parameters and import them


    # merge them into a dictionary
    model_input_data = {
        'heat_demand': df_heat_demand,
        'heat_gen_units': gdf_heat_gen_units,
        'heat_network': gdf_heat_network,
        'heat_nodes': gdf_heat_nodes,
        'waste_heat_prof': df_waste_heat_prof
        }

    print("Done: loading model input data", 'green')
    return model_input_data


def load_temp_data(scenario: str, config: dict) -> pd.DataFrame:
    """Loads the outside temperature data for the scenario from disk.

    :param scenario: name of the scenario
    :type scenario: str
    :param config: configuration dictionary
    :type config: dict
    :return: dataframe containing the outside temperature data and the time index
    :rtype: pd.DataFrame
    """

    path_temp = os.path.join(config['CaseStudies_dir'],scenario, config["parameter_dir"], config["MCMC_dir"], config['case_study_data']['outside_temp'])

    df_temperature = pd.read_excel(path_temp) #, skiprows=[0,1,2])
    df_temperature['time'] = pd.to_datetime(df_temperature['time'])
    return df_temperature

def load_solar_gain_data(scenario: str, config: dict) -> pd.DataFrame:
    """Loads the solar gain data for the scenario from disk.

    Solar gain data here referes to the solar irradiation in W/m2 data that is used to calculate the solar gain for the buildings.

    :param scenario: name of the scenario
    :type scenario: str
    :param config: configuration dictionary
    :type config: dict
    :return: dataframe containing the solar gain data and the time index
    :rtype: pd.DataFrame
    """

    path_temp = os.path.join(config['CaseStudies_dir'],scenario, config["parameter_dir"], config["MCMC_dir"], config['case_study_data']['solar_gain'])

    df_solar_gain = pd.read_excel(path_temp) #, skiprows=[0,1,2])
    df_solar_gain['time'] = pd.to_datetime(df_solar_gain['time'])
    return df_solar_gain


def generate_new_scenario(casestudy: str, scenario_dir: str, config: dict):
    # create a new scenario folder
    # copy the data from the template folder
    # create a new scenario
    
    CaseStudies_dir = config['CaseStudies_dir']
    scenario_path = os.path.join(CaseStudies_dir,casestudy, config['scenario_dir'], scenario_dir)

    # check if the scenario folder exists
    if os.path.exists(scenario_path):
        print("Scenario folder exists already")
    else:
        print("Creating new scenario folder with default values")
        os.makedirs(scenario_path)
        # copy the data from the defailt folder
        default_path = os.path.join(config['default_data'])
        input_path = os.path.join(scenario_path,config['input_dir'])
        os.makedirs(input_path)
        # copy the data
        shutil.copyfile(os.path.join(default_path,config['heat_source_data']['heat_profiles']), os.path.join(input_path,config['heat_source_data']['heat_profiles']))
        shutil.copyfile(os.path.join(default_path,config['heat_source_data']['heat_sources']), os.path.join(input_path,config['heat_source_data']['heat_sources']))
        shutil.copyfile(os.path.join(default_path,config['parameter_costs']['input_file']), os.path.join(input_path,config['parameter_costs']['input_file']))
        


def generate_new_case_study(casestudy: str, config: dict):
    # create a new case study folder
    # copy the data from the template folder
    # create a new scenario
    CaseStudies_dir = config['CaseStudies_dir']
    casestudy_path = os.path.join(CaseStudies_dir,casestudy)

    # check if the scenario folder exists
    if os.path.exists(casestudy_path):
        print("Case study folder exists already")
    else:
        default_path = os.path.join(config['default_data'])
        buidling_data_path = os.path.join(casestudy_path,config['parameter_dir'],config['building_data_dir'])
        MCMC_data_path = os.path.join(casestudy_path,config['parameter_dir'],config['MCMC_dir'])
        scenario_path = os.path.join(CaseStudies_dir,casestudy, config['scenario_dir'])
        print("Creating new case study folder with default values")
        os.makedirs(buidling_data_path)
        os.makedirs(MCMC_data_path)
        os.makedirs(scenario_path)
        # copy the data from the defailt folder
        # copy the data
        shutil.copyfile(os.path.join(default_path,config['case_study_data']['building_typology']), os.path.join(buidling_data_path,config['case_study_data']['building_typology']))
        shutil.copyfile(os.path.join(default_path,config['case_study_data']['outside_temp']), os.path.join(MCMC_data_path,config['case_study_data']['outside_temp']))
        shutil.copyfile(os.path.join(default_path,config['case_study_data']['solar_gain']), os.path.join(MCMC_data_path,config['case_study_data']['solar_gain']))
        shutil.copyfile(os.path.join(default_path,config['case_study_data']['transition_matrix_WD']), os.path.join(MCMC_data_path,config['case_study_data']['transition_matrix_WD']))
        shutil.copyfile(os.path.join(default_path,config['case_study_data']['transition_matrix_WE']), os.path.join(MCMC_data_path,config['case_study_data']['transition_matrix_WE']))

    
def prepare_parameter_file(case_study_name: str, scenario_name: str, config: dict):
    CaseStudies_dir = config['CaseStudies_dir']
    #load the parameter xlsx
    path_read_parameter = os.path.join(CaseStudies_dir,case_study_name, config['scenario_dir'], scenario_name, config['input_dir'])
    path_write_parameter = os.path.join(CaseStudies_dir,case_study_name, config['scenario_dir'], scenario_name, config['data_dir'], config['model_data']['root_dir'])
    print("Start: load heat generation units from: " + path_read_parameter)
    df_parametercost = pd.read_excel(os.path.join(path_read_parameter,config['parameter_costs']['input_file']))

    # change the columns parameter and value to dictionary
    df_parametercost = df_parametercost[['Parameter','Value']]
    print(df_parametercost)
    dict_parametercost = df_parametercost.set_index('Parameter').T.to_dict('records')[0]

    # check if the directory exists
    if not os.path.exists(path_write_parameter):
        os.makedirs(path_write_parameter)
                    
    # save dictionary to json
    with open(os.path.join(path_write_parameter,config['model_data']['cost_parameter']), 'w') as fp:
        json.dump(dict_parametercost, fp)


def read_output_from_disk(case_study_name: str, model_name: str, config: dict):
    CaseStudies_dir = config['CaseStudies_dir']
    path_output = os.path.join(CaseStudies_dir,case_study_name, config['scenario_dir'], model_name, config['output_dir'])

    # get all the file names in the folder 
    files = os.listdir(path_output)
    
    model_output_data = {}
    
    for file in files:
        temp_filename = file.split('.')[0]
        if file.endswith('.csv'):
            model_output_data[temp_filename] = pd.read_csv(os.path.join(path_output, file), sep=',')  # change sperator if needed (and changed for the rest)
        elif file.endswith('.geojson'):
            model_output_data[file] = gpd.read_file(os.path.join(path_output, file))
        else:
            print("File format not supported: ", file)

    return model_output_data





# load the configuration file from scenario_parameters.py
# exec(open(scenario_folder + '/scenario_parameters.py').read()) # the parameters are now available as variables and can be directly accessed

# print("Scenario folder selected: ", scenario_folder)
#
# df_heat_demand = pd.read_csv(os.path.join(scenario_folder,'data', 'Heat_Demand.csv'), sep=';')
# df_heat_gen_units = pd.read_csv(os.path.join(scenario_folder,'data', 'Heat_Generation_Units.csv'), sep=';')
# df_heat_network = pd.read_csv(os.path.join(scenario_folder,'data', 'Heat_Network.csv'), sep=';')
# df_heat_nodes = pd.read_csv(os.path.join(scenario_folder,'data', 'Heat_Nodes.csv'), sep=';')
# df_waste_heat_prof = pd.read_csv(os.path.join(scenario_folder,'data', 'Heat_WH_Profiles.csv'), sep=';')
#
# print("Done: read data")