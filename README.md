# Atmosphere_Software_Prototype
Atmosphere_Software_Prototype

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

This application has been developed for the ATMOSPHERE project, which was supported by funding from CDTI, with Grant Number MIG-20221006 associated with the ATMOSPHERE Project.