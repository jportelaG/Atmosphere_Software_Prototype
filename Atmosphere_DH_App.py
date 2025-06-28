import streamlit as st
# import general packages
import os

# import the Atmosphere libaries (these are custom modules inside the src folder)
# Ensure the src directory is in the Python path
import sys
sys.path.append(os.path.join(os.getcwd(), "src"))
# Import custom modules
from src import data, clustering, model, visualisation, prepare_geodata, hd_time_series_generator, constructive_elements


import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from pdf2image import convert_from_path

# Set wide layout
st.set_page_config(layout="wide", page_title="District Heating Network Designer", )

# These will be initialized in show_menu and must persist
case_study = st.session_state.get("case_study", "Puertollano")
scenario = st.session_state.get("scenario", "Scenario_demo2")
config = st.session_state.get("config", data.load_config())

# Sidebar Menu
# Add an image to the sidebar
sidebar_image = 'logoApp.png'  # replace with the path to your image
st.sidebar.image(sidebar_image, use_container_width =True)

st.sidebar.title("Navigation")
menu_option = st.sidebar.radio("Go to", [
    "Menu", 
    "1. Define Case Study and Scenarios",
    "2. Parameter Settings",
    "3. Obtain Heat Demand Profiles",
    "4. Clustered district heat demand",
    "5. District Heating Optimization",
    "6. Explore Optimization Results"
])


# Placeholder functions
def show_menu():
    st.title("District Heating Network Designer")
    st.markdown("""
    Welcome to the **District Heating Network Designer Tool**.  
    This tool allows you to design and optimize district heating networks using real-world data for different locations.
        
    It is designed to support heat energy planning at the community level using open-data sources. It provides a flexible framework for estimating heat demand, generating detailed heat demand time series, and evaluating the potential for waste heat utilization through district heating networks using linear optimization.
    
    **Main Features:**
    - Stochastic heat demand modeling: Generates building-level heat demand time series using probabilistic models.
    - District heating network proposals: Offers clustering and greenfield network planning based on spatial and energy criteria.
    - Waste heat utilization assessment: Evaluates the feasibility of integrating local waste heat sources.

    **General Framework**
    The tool is structured into four modular components:
    - Prepare geodata Accesses building geometry and attributes from OpenStreetMap (OS and estimates annual heating demands based on building typologies.
    - Generate heat demand time series Calculates dynamic heat demand profiles through thermal modeling combined with active occupancy simulations.
    - Cluster heat demand and propose networks Performs spatial clustering of buildings to suggest candidate district heating networks, considering both technical and spatial factors.
    - Optimize waste heat utilization Applies a linear optimization model to maximize the utilization of available waste heat sources, balancing investment and operational costs.
    
    Use the sidebar to navigate between different stages:
    - Define or import load profiles  
    - Optimize the network  
    - Visualize the results
    
    """)
    
    # Add a link to the GitHub repository
    st.markdown("""    For more information, visit the [GitHub repository](https://github.com/jportelaG/Atmosphere_Software_Prototype).
    """)
    
    # add funding information
    st.markdown("""This application has been developed for the ATMOSPHERE project, which was supported by funding from CDTI, with Grant Number MIG-20221006 associated with the ATMOSPHERE Project.""")
    # add image of the ATMOSPHERE project
    st.image('CDTI.png', use_container_width =False, width=400)
    # add to the side of the image, the logo of innomerics
    st.image('Logo_innomerics.png', use_container_width =False, width=300)
    # add logo of iit
    st.image('Logo IIT_Color.jpg', use_container_width =False, width=250)
    # add logo of iee
    st.image('Logo_IEE.png', use_container_width =False, width=300)
    




def define_case_study_and_scenario():
    # --- Selectors ---
    st.subheader("Select Case Study and Scenario")
    st.markdown("""
    This section allows you to select a case study and scenario for district heating network design.
    
    You can choose from existing case studies or create a new scenario based on the selected case study.
    If you create a new scenario, it will be initialized with default parameters and can be customized later.
    
    Please select a case study and scenario from the dropdown menus below.            
    """)

    # Function to get scenario options based on selected case study
    config = data.load_config()
    CaseStudies_dir = config['CaseStudies_dir']
    scenario_dir = config['scenario_dir']
    # Get the list of case studies from the CaseStudies directory
    case_study_options = [d for d in os.listdir(CaseStudies_dir) if os.path.isdir(os.path.join(CaseStudies_dir, d))]
    case_study = st.selectbox("Case Study", case_study_options)
    
    # Button for creating a new case study
    with st.popover("**New** Case Study"):
        new_case_study_name = st.text_input("Enter new case study name")
        if new_case_study_name:
            if st.button("Create New Case Study: " + new_case_study_name):
                data.generate_new_case_study(new_case_study_name, config)
                st.success(f"New case study '{new_case_study_name}' created successfully! Please, reload and generate a new scenario for it.")

    # if case study is not empty
    if case_study:
        # Get the scenarios for the selected case study
        scenario_options = [
            d for d in os.listdir(os.path.join(CaseStudies_dir, case_study, scenario_dir)) 
            if os.path.isdir(os.path.join(CaseStudies_dir, case_study, scenario_dir, d))
        ]
        scenario = st.selectbox("Scenario", scenario_options)

        # Button for creating a new scenario
        with st.popover("**New** Scenario"):
            new_Scenario_name = st.text_input("Enter new Scenario name")
            if new_Scenario_name:
                if st.button("Create New Scenario: " + new_Scenario_name):
                    scenario = new_Scenario_name
                    st.session_state["case_study"] = case_study
                    st.session_state["scenario"] = scenario
                    # create the new case study directory
                    data.generate_new_scenario(case_study, scenario, config)
                    st.success(f"Case study '{case_study}' with scenario '{scenario}' created and loaded successfully!")

        # If a scenario is selected, show Button to load the selected case study and scenario
        if scenario:
            if st.button("Load Case Study and Scenario" , type="primary"):
                with st.spinner("Loading configuration and generating scenario..."):
                    st.session_state["case_study"] = case_study
                    st.session_state["scenario"] = scenario
                    data.generate_new_scenario(case_study, scenario, config)
                    st.success(f"Case study '{case_study}' with scenario '{scenario}' loaded successfully!")

    # add division line to separate the section
    st.markdown("---")
    
           
    # Button to reset the session state
    if st.button("Reset Session State"):
        st.session_state.clear()
        st.success("Session state reset. Please reload the page to start fresh.")








def parameter_settings_update():
    st.title("Parameter Settings")
    st.markdown("Update the parameters for the district heating network design.")
    
    st.subheader("Case Study Settings")
    st.markdown("""
    For obtaining building-level heat demand profiles for the Case Study the following parameters are required:
    - **Building Typology**: This file defines the annual heating demand in kWh/m²/year for various building types. It distinguishes between building age (year of construction) and building use (e.g., residential, commercial). You should adjust these values for your specific region and climate zone.
    - **Outside Temperature**: This file contains the hourly outside temperature data for the selected case study.
    - **Solar Gain**: This file contains the hourly solar irradiation in $$W/m^2$$ data for the selected case study.
    - **Transition Matrices (WD; WE)**: These matrices are used to generate stochastic active occupancy profiles.
    
    You can open the parameter files in Excel to edit them. The files will be opened in excel if it is installed on your system.
        """)
    # Button to open the Building Typology file
    if st.button("Open Building Typology file"):
        file_path = os.path.join(config['CaseStudies_dir'],case_study,config['parameter_dir'],config['building_data_dir'],config['case_study_data']['building_typology'])
        if os.path.exists(file_path):
            os.startfile(file_path)
        else:
            st.error(f"File not found: {file_path}. Please check the path and try again.")
    # Button to open the Outside Temperature file
    if st.button("Open Outside Temperature file"):
        file_path = os.path.join(config['CaseStudies_dir'],case_study,config['parameter_dir'],config['MCMC_dir'],config['case_study_data']['outside_temp'])
        if os.path.exists(file_path):
            os.startfile(file_path)
        else:
            st.error(f"File not found: {file_path}. Please check the path and try again.")
    # Button to open the Solar Gain file
    if st.button("Open Solar Gain file"):
        file_path = os.path.join(config['CaseStudies_dir'],case_study,config['parameter_dir'],config['MCMC_dir'],config['case_study_data']['solar_gain'])
        if os.path.exists(file_path):
            os.startfile(file_path)
        else:
            st.error(f"File not found: {file_path}. Please check the path and try again.")
    # Button to open the Transition Matrix file (WD)
    if st.button("Open Transition Matrix (WD) file"):
        file_path = os.path.join(config['CaseStudies_dir'],case_study,config['parameter_dir'],config['MCMC_dir'],config['case_study_data']['transition_matrix_WD'])
        if os.path.exists(file_path):
            os.startfile(file_path)
        else:
            st.error(f"File not found: {file_path}. Please check the path and try again.")
    # Button to open the Transition Matrix file (WE)
    if st.button("Open Transition Matrix (WE) file"):
        file_path = os.path.join(config['CaseStudies_dir'],case_study,config['parameter_dir'],config['MCMC_dir'],config['case_study_data']['transition_matrix_WE'])
        if os.path.exists(file_path):
            os.startfile(file_path)
        else:
            st.error(f"File not found: {file_path}. Please check the path and try again.")
    
    
    # add division line to separate the section
    st.markdown("---")
    
    # Display current configuration, reading from 
    st.subheader("Scenario Settings: Parameter Costs")
    st.markdown("The table below shows the current parameter costs used in the district heating network design.")
    #load the parameter xlsx
    path_read_parameter = os.path.join(config['CaseStudies_dir'],case_study, config['scenario_dir'], scenario, config['input_dir'])
    path_write_parameter = os.path.join(config['CaseStudies_dir'],case_study, config['scenario_dir'], scenario, config['data_dir'], config['model_data']['root_dir'])
    print("Start: load parameters from: " + path_read_parameter)
    df_parametercost = pd.read_excel(os.path.join(path_read_parameter,config['parameter_costs']['input_file']))
    # Display the parameter costs in a dataframe
    df_parametercost = st.data_editor(df_parametercost, use_container_width=True, key="parameter_costs")
    st.markdown("You can edit the values in the table above and click 'Update Parameter' to save changes.")    

    if st.button("Update Parameter Costs"):
        with st.spinner("Updating parameter costs..."):
            # Save the updated parameter costs to the file
            df_parametercost.to_excel(os.path.join(path_write_parameter, config['parameter_costs']['input_file']), index=False)
            st.success("Parameter costs updated successfully!")
    
    # add division line to separate the section
    st.markdown("---")
    
    st.subheader("Scenario Settings: Heat Generation Units")
    st.markdown("This section loads the heat generation units from the disk for the selected **Case study and scenario**.")
    
    path_read_parameter = os.path.join(config['CaseStudies_dir'],case_study, config['scenario_dir'], scenario, config['input_dir'])
    path_write_parameter = os.path.join(config['CaseStudies_dir'],case_study, config['scenario_dir'], scenario, config['data_dir'], config['model_data']['root_dir'])
    print("Start: load heat generation units from: " + path_read_parameter)
    df_heatsourcet = pd.read_excel(os.path.join(path_read_parameter,config['heat_source_data']['heat_sources']))
    
    # Display the parameter costs in a table which can be edited
    st.markdown("You can edit the values in the table below and click 'Update Parameter' to save changes.")
    # Display the parameter costs in a dataframe
    df_heatsourcet = st.data_editor(df_heatsourcet, use_container_width=True, key="df_heatsourcet")
    st.markdown("You can edit the values in the table above and click 'Update Parameter' to save changes.")    
    if st.button("Update Heat Source Parameters"):
        with st.spinner("Updating parameter costs..."):
            # Save the updated parameter costs to the file
            df_heatsourcet.to_excel(os.path.join(path_write_parameter, config['heat_source_data']['heat_sources']), index=False)
            st.success("Parameter costs updated successfully!")
            
    # Add buttons to open the excel files of other parameters
    st.subheader("Scenario Settings: Electrolyser Waste Heat")
    st.markdown("You can open the parameter files in Excel to edit them. The files will be opened in excel if it is installed on your system.")
    # Button to open the parameter costs file
    if st.button("Open Waste Heat File"):
        file_path = os.path.join(path_read_parameter, config['heat_source_data']['heat_profiles'])
        if os.path.exists(file_path):
            os.startfile(file_path)
        else:
            st.error(f"File not found: {file_path}. Please check the path and try again.")
            
            


        
        
        
        
        
        
def create_building_load_profiles(): 
    st.title("Create Building-level Load Profiles")
    st.markdown("This section generates heat demand profiles for each building in the desired location for the case study.")
    
    # load the case study and scenario from session state
    if 'case_study' not in st.session_state or 'scenario' not in st.session_state:
        st.error("Please define a case study and scenario first in the 'Define Case Study and Scenarios' section.")
        return
    
    # get the case study and scenario from session state
    case_study = st.session_state['case_study']
    scenario = st.session_state['scenario']
    config = data.load_config()
    

    output_file = os.path.join(config['CaseStudies_dir'], case_study, 'heat_demand_map.html')
    # if the output file already exists, show the map
    if os.path.exists(output_file):
        st.subheader("Existing Heat Demand Map")
        st.markdown("The map below shows the locations of buildings with their heat demand profiles. You can interact with it to explore the data.")
        st.components.v1.html(open(output_file, 'r').read(), height=500)
        #if buildings_TS is in session state, show the buildings_TS dataframe
        if 'buildings_TS' in st.session_state:
            buildings_TS = st.session_state['buildings_TS'] 
        else:
            #load the buildings_TS from disk
            buildings = clustering.read_geo_data_from_disk(case_study, config)
            buildings_TS = clustering.read_building_TS_from_disk(case_study, config)
            st.session_state['buildings_TS'] = buildings_TS
            st.session_state['buildings'] = buildings
    else:
        st.markdown("No heat demand map found for the selected case study. Please generate load profiles first.")
            
    # Specify the location for obtaining building data
    with st.popover("**Regenerate Building Data**"):
        st.subheader("Select Location for Building Data")
        st.markdown(""" Choose a location to obtain building data for generating load profiles.  
        The selected location will be used to fetch building geometry and attributes from OpenStreetMap (OSM) and estimate annual heating demands based on building typologies.
        You can select from predefined locations or enter a custom location.    """)    
        # edit textbox to enter a custom location
        custom_location = st.text_input("Enter custom location (e.g., city name or coordinates)", value="Puertollano, Spain")
        # Button to generate load profiles
        if st.button("Generate Building-level load profiles"):
            with st.spinner("Generating Building-level profiles..."):
                # generate a complete geodataset for the case study
                gdf_buildings = prepare_geodata.generate_complete_geodataset(case_study, custom_location)
                
                # check the heat demand in the area
                fig = visualisation.plot_HD_interactive(gdf_buildings)
                fig.save(output_file)# Display the saved HTML map            
                st.markdown("The map below shows the locations of buildings with their heat demand profiles. You can interact with it to explore the data.")
                st.components.v1.html(open(output_file, 'r').read(), height=500)
                
                st.success("Load profiles generated and scenario data saved.")
        
        if st.button("Generate Building-level load profiles2"):     
            df_HD_time_series = hd_time_series_generator.fast_TS_generator(case_study, True)
    
    
    # add dropdown menu to select the building to plot
    if os.path.exists(output_file):
        ## Load the generated building-level load profiles from disk and plot
        st.subheader("Plot Building-level Heat Demand Profiles")
        building_id = st.selectbox("Select Building ID", buildings_TS.columns[1:].to_list())  # Exclude 'hour' column
        if building_id:
            st.markdown(f"Plotting load profile for Building ID: {building_id}")
            # filter the building time series for the selected building and create plotly chart for the selected building
            fig = px.line(buildings_TS, x='hour', y=building_id, title=f'Load Profile for Building ID: {building_id}')
            st.plotly_chart(fig, use_container_width=True)

            

            
def clustered_load_profiles():
    st.title("Clustered district heat demand")
    st.markdown("This section allows clustering the building-level heat demand and prepare data for network optimization.")

    # load the data from session state
    if 'buildings' not in st.session_state:
        st.error("Please generate load profiles first in the 'Create Load Profiles' section.")
        return
    
    # load the buildings and buildings_TS from session state
    buildings = st.session_state['buildings']
    buildings_TS = st.session_state['buildings_TS']
    
    # check if building_cluster is in session state 
    if 'building_cluster' in st.session_state:
        st.success("Building clustering already exists")
        model_input_data = data.load_data_from_disk(case_study, scenario, config)
        building_cluster = st.session_state['building_cluster']
        clustered_TS = st.session_state['clustered_TS']
        generation_units = st.session_state['generation_units']
        waste_heat_profiles = st.session_state['waste_heat_profiles']
        network = st.session_state['network']
        all_heat_nodes = st.session_state['all_heat_nodes']
        

    elif os.path.exists(os.path.join(config['CaseStudies_dir'],case_study, config['scenario_dir'], scenario, config['data_dir'], config['model_data']['root_dir'], config['model_data']['heat_network'])):
        st.success("Building clustering already exists in disk")
        model_input_data = data.load_data_from_disk(case_study, scenario, config)
        clustered_TS = model_input_data["heat_demand"]
        generation_units = model_input_data["heat_gen_units"]
        waste_heat_profiles = model_input_data["waste_heat_prof"]
        network = model_input_data["heat_network"]
        all_heat_nodes = model_input_data["heat_nodes"]
        #load to session state
        st.session_state['clustered_TS'] = clustered_TS
        st.session_state['generation_units'] = generation_units
        st.session_state['waste_heat_profiles'] = waste_heat_profiles
        st.session_state['network'] = network
        st.session_state['all_heat_nodes'] = all_heat_nodes
    
    
    if st.button("Generate clustered Load Profiles"):
        with st.spinner("Generating clustered load profiles..."):
            
            # 1. Prepare parameter file
            clustering.prepare_parameter_file(case_study, scenario, config)

            # 2. Load cost and parameter data
            cost_parameter = clustering.load_cost_parameter(case_study, scenario, config)

            # 3. Load input datasets
            #buildings = clustering.read_geo_data_from_disk(case_study, config)
            #buildings_TS = clustering.read_building_TS_from_disk(case_study, config)
            generation_units = clustering.read_heatGenUnits_from_disk(case_study, scenario, config)
            waste_heat_profiles = clustering.read_wasteHeatProfiles_from_disk(generation_units, case_study, scenario, config)

            # 4. Building clustering
            #buildings = clustering.read_geo_data_from_disk(case_study, config)  # required for visual map
            clustering.get_centroids(buildings)
            building_cluster = clustering.cluster_buildings(buildings, cost_parameter)

            # 5. Cluster time series
            clustered_TS = clustering.cluster_heat_demand(building_cluster, buildings_TS)
            clustered_TS = clustering.add_hot_water_demand(building_cluster, clustered_TS, cost_parameter)

            # 6. Merge clusters with generation units
            all_heat_nodes = clustering.merge_heatGenUnits(building_cluster, generation_units)

            # 7. Propose network
            network = clustering.propose_network(all_heat_nodes)

            # 8. Save scenario data
            clustering.save_data(all_heat_nodes, clustered_TS, network, generation_units, waste_heat_profiles, case_study, scenario, config)

            # save buildings, buildings_TS, clustered_TS, generation_units, to memory for later use
            #st.session_state['building_cluster'] = building_cluster
            st.session_state['clustered_TS'] = clustered_TS
            st.session_state['generation_units'] = generation_units
            st.session_state['waste_heat_profiles'] = waste_heat_profiles
            st.session_state['network'] = network
            st.session_state['all_heat_nodes'] = all_heat_nodes

            
            st.success("Clustered Load profiles generated and scenario data saved.")
    
    # --- Plots Section ---
    # plot different plots
    if 'all_heat_nodes' in st.session_state:
        st.subheader("Visualize Clustered Heat Demand")
        
        # dropdown menu to select the type of plot
        plot_type = st.selectbox("Select Plot Type", ["Heat Demand by cluster", "Electrolyser Waste Heat Profile", "Cluster-level Heat Demand Profiles"])

        # if selected plot type is Heat Demand Profiles
        if plot_type == "Heat Demand by cluster":
            st.subheader("Heat Demand by cluster")
            st.markdown("The map below shows the locations of clusters with their yearly heat demand. You can interact with it to explore the data.")           
            # save explore() into html file
            output_file = os.path.join(config['CaseStudies_dir'], case_study,config['scenario_dir'],scenario,config['plot_dir'],'explore_buildings.html')
            fig = all_heat_nodes.explore('YearlyDemand', width=800, height=400, cmap='coolwarm',marker_kwds={'radius': 10} )
            fig.save(output_file)# Display the saved HTML map
            st.components.v1.html(open(output_file, 'r').read(), height=500)

        # if selected plot type is Electrolyser Waste Heat Profile
        elif plot_type == "Electrolyser Waste Heat Profile":
            st.subheader("Electrolyser Waste Heat Profile")
            fig = px.line(waste_heat_profiles, x='hour', y='Electrolyser', title='Electrolyser Waste Heat Profile')
            st.plotly_chart(fig, use_container_width=True)

        # if selected plot type is Clustered Heat Demand Profile
        elif plot_type == "Cluster-level Heat Demand Profiles":        
            st.subheader("Cluster-level Heat Demand Profiles")
            st.markdown("The clustered heat demand profile shows the aggregated heat demand for each cluster of buildings.")
            cluster_id = st.selectbox("Select Building ID", clustered_TS.columns[1:].to_list())  # Exclude 'hour' column
            if cluster_id:
                st.markdown(f"Plotting load profile for Building ID: {cluster_id}")
                # filter the building time series for the selected building and create plotly chart for the selected building
                fig = px.line(clustered_TS, x='hour', y=cluster_id, title=f'Clustered Heat Demand Profile for cluster ID: {cluster_id}')
                st.plotly_chart(fig, use_container_width=True)
            
        
        
        
        
        
def optimize_network():
    st.title("District Heating Optimization")
    st.markdown("Run the optimization model and visualize key investment decisions.")

    # load the data from session state
    if 'network' not in st.session_state:
        st.error("Please generate clustered load profiles first in the 'Clustered District Heat Demand' section.")
        return

    figure_path = os.path.join(config['CaseStudies_dir'],case_study, config['scenario_dir'], scenario, config['plot_dir'])
    
    if st.button("Run Optimization & Visualize Decisions"):
        with st.spinner("Optimizing network design..."):
            # run the combined optimisation model
            # model.run_model(case_study, scenario, config)
            
            # 1. Read model output and input data
            model_output_data = data.read_output_from_disk(case_study, scenario, config)
            model_input_data = data.load_data_from_disk(case_study, scenario, config)
            # 2. Extract nodes and shapes
            dict_nodes = visualisation.extract_node_centroids(model_input_data)
            dict_shapes = visualisation.extract_node_shapes(model_input_data)
            # Obtain buildPipe and save it to session state
            gdf_BuildPipe = visualisation.plot_investment_decisions(model_input_data, model_output_data, dict_nodes, dict_shapes, figure_path)
            st.session_state['gdf_BuildPipe'] = gdf_BuildPipe
            st.session_state['model_output_data'] = model_output_data
            
            visualisation.plot_energy_balance(visualisation.merge_time_series(model_input_data, model_output_data), figure_path)
            # plot the time resolved energy balance; the time_inverall specifies the time resolution of the plot. 
            # For example, 'h' for hourly, 'd' for daily, 'W' for weekly, 'M' for monthly, 'Y' for yearly.
            visualisation.plot_time_resolved(visualisation.merge_time_series(model_input_data, model_output_data), figure_path, time_invervall='W')

            st.success("Optimization completed and investment decisions obtained.")

    # if former optimization results are in folder, load them
    if os.path.exists(os.path.join(figure_path, "investment_decisions.jpg")):
        st.subheader("Optimization Results")
        st.success("Optimization results already exists")
        gdf_BuildPipe = st.session_state['gdf_BuildPipe']
        # 5. Display results if figure(s) were saved
        st.subheader("🖼️ Investment Decision Plots")
        # 4. Plot investment decisions
        st.image(os.path.join(figure_path, "investment_decisions.jpg"), caption="Investment Decisions")
        # plot the energy balance
        st.subheader("📊 Energy Balance Plot")
        st.image(os.path.join(figure_path, "energy_balance.jpg"), caption="Energy Balance")
        # plot the time resolved energy balance
        st.subheader("📈 Time Resolved Energy Balance Plot")
        st.image(os.path.join(figure_path, "W_time_resolved_energy_balance.jpg"), caption="Time Resolved Energy Balance")

def show_results():
    st.title("Explore Optimization Results")
    st.markdown("View and interact with the optimization results.")
    
    if 'gdf_BuildPipe' not in st.session_state:
        st.error("Please run the optimization first in the 'District Heating Optimization' section.")
        return
    # load the gdf_BuildPipe from session state
    gdf_BuildPipe = st.session_state['gdf_BuildPipe']
    model_output_data = st.session_state['model_output_data']
    
    # Calculate the total length of built pipes
    gdf_built_pipes = constructive_elements.total_length_of_built_pipes(gdf_BuildPipe)
    gdf_built_pipes = constructive_elements.design_pump_and_pipe(gdf_built_pipes)
    st.subheader("Built Pipes Data")
    total_length_m = gdf_built_pipes['length_m'].sum()
    st.markdown(f"Total length of built pipes: {total_length_m:.2f} meters")
    st.markdown("The table below shows the built pipes with their properties such as node connections, mass flow, and length in meters.")
    # Display the built pipes data in a table
    #st.dataframe(gdf_built_pipes[['node_from', 'node_to', 'mass_flow', 'length_m']], use_container_width=True)
    st.dataframe(gdf_built_pipes[['node_from', 'node_to', 'mass_flow', 'length_m','D_mm', 'ΔP_bar']], use_container_width=True)
    

    # plot the built pipes on a map
    st.subheader("Built Pipes Map") 
    st.markdown("The map below shows the built pipes with their properties such as node connections, mass flow, and length in meters. You can interact with it to explore the data.")
    gdf_BuildPipe_wgs84 = gdf_built_pipes#.set_crs(epsg=4326)
    fig = gdf_BuildPipe_wgs84.explore(
        'mass_flow',  # Optional: color lines by 'mass_flow'
        cmap='turbo',     # Choose colormap
        legend=True,         # Add a legend
        #tooltip=['node_from', 'node_to', 'mass_flow'],  # Show info on hover
        style_kwds={'weight': 4},  # Line thickness
        height=500,
        width=800
        )
    output_file = os.path.join(config['CaseStudies_dir'], case_study,config['scenario_dir'],scenario,config['plot_dir'],'heat_demand_map.html')
    fig.save(output_file)# Display the saved HTML map
    st.components.v1.html(open(output_file, 'r').read(), height=500)
    
    # create rounded values for the diameter DN 100, 200,400 , 800, 1000
    def round_to_nearest_dn(diameter_mm):
        dn_values = [100, 200, 400, 800, 1000]
        return min(dn_values, key=lambda x: abs(x - diameter_mm))

    # Apply the rounding function to the diameter column
    gdf_built_pipes['DN'] = gdf_built_pipes['D_mm'].apply(round_to_nearest_dn)

    # group by DN, count pipes and sum the lengths
    grouped = gdf_built_pipes.groupby('DN').agg(
        pipe_count=('length_m', 'count'),
        total_length_m=('length_m', 'sum')
    ).reset_index()
    # Display the grouped data in a table
    st.subheader("Grouped Built Pipes Data")
    st.markdown("The table below shows the grouped built pipes data by diameter (DN) with the count of pipes and total length in meters.")
    st.markdown("Total number of connected districts: " + str(model_output_data['vDHconnect'].value.sum()))
    st.dataframe(grouped, use_container_width=True)
    

# Render selected section
if menu_option == "Menu":
    show_menu()
elif menu_option == "1. Define Case Study and Scenarios":
    define_case_study_and_scenario()
elif menu_option == "2. Parameter Settings":
    parameter_settings_update()
elif menu_option == "3. Obtain Heat Demand Profiles":
    create_building_load_profiles()
elif menu_option == "4. Clustered district heat demand":
    clustered_load_profiles()
elif menu_option == "5. District Heating Optimization":
    optimize_network()
elif menu_option == "6. Explore Optimization Results":
    show_results()
else:
    st.error("Please select a valid option from the sidebar menu.")
