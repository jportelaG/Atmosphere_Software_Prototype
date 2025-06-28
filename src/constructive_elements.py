import numpy as np
import pandas as pd
import geopandas as gpd

def total_length_of_built_pipes(gdf_BuildPipe):
    """
    Calculate the total length of built pipes from a GeoDataFrame.
    
    Parameters:
    gdf_BuildPipe (GeoDataFrame): DataFrame containing pipe segments with 'invest' and 'geometry'.
    
    Returns:
    None: Prints the total length of built pipes in meters.
    """
        
    # Step 0: Set the CRS if not already set (assume WGS84, EPSG:4326)
    if gdf_BuildPipe.crs is None:
        gdf_BuildPipe = gdf_BuildPipe.set_crs(epsg=4326)

    # Step 1: Project the GeoDataFrame to a metric CRS (e.g., UTM zone 30N for Spain)
    # You can change the EPSG code depending on your location
    gdf_BuildPipe_projected = gdf_BuildPipe.to_crs(epsg=25830)  # EPSG:25830 = ETRS89 / UTM zone 30N

    # Step 2: Filter only pipes that were actually built (invest == 1)
    gdf_built_pipes = gdf_BuildPipe_projected[gdf_BuildPipe_projected['invest'] == 1]

    # Step 3: Calculate the length of each built pipe segment in meters
    gdf_built_pipes['length_m'] = gdf_built_pipes['geometry'].length

    return gdf_built_pipes

def design_pump_and_pipe(gdf_built_pipes):
    """
    Design pumps and pipes based on mass flow and length.
    
    Parameters:
    gdf_built_pipes (GeoDataFrame): DataFrame containing pipe segments with mass flow and length.
    
    Returns:
    GeoDataFrame: Updated DataFrame with calculated diameters, pressure drops, and pump power.
    """
    
    # Constants  
    rho = 971.8          # kg/m³ (water at ~80°C)
    mu = 0.000355        # Pa·s
    epsilon = 0.0001     # m (pipe roughness)
    v_design = 2.0       # m/s (chosen design velocity)
    eta_pump = 0.75      # pump efficiency (can vary from 0.6 to 0.85)

    
    # 1. Compute inner diameter based on mass flow and velocity
    def compute_diameter(m_dot, v=v_design):
        A = m_dot / (rho * v)
        D = np.sqrt(4 * A / np.pi)
        return D  # in meters

    # 2. Compute friction factor using Swamee-Jain equation
    def swamee_jain_friction(Re, D):
        if Re < 2000:
            return 64 / Re
        else:
            return 0.25 / (np.log10(epsilon / (3.7 * D) + 5.74 / Re**0.9))**2

    # 3. Compute pressure drop
    def compute_pressure_drop(D, m_dot, length):
        A = np.pi * D**2 / 4
        v = m_dot / (rho * A)
        Re = rho * v * D / mu
        f = swamee_jain_friction(Re, D)
        dp = f * (rho * v**2) / (2 * D) * length
        return dp, Re, f

    # 4. Compute pump power
    def compute_pump_power(m_dot, dp, eta=eta_pump):
        Q = m_dot / rho  # Volumetric flow [m³/s]
        power_hydraulic = Q * dp  # Watts
        power_electric = power_hydraulic / eta
        return power_hydraulic, power_electric

    # 5. Apply to each pipe segment
    def design_pump_and_pipe(row):
        m_dot = row['mass_flow']
        length = row['length_m']
        
        D = compute_diameter(m_dot)
        dp, Re, f = compute_pressure_drop(D, m_dot, length)
        P_hyd, P_el = compute_pump_power(m_dot, dp)
        
        return pd.Series({
            'D_m': D,
            'D_mm': D * 1000,
            'Re': Re,
            'f': f,
            'ΔP_Pa': dp,
            'ΔP_bar': dp / 1e5,
            'Pump_hydraulic_W': P_hyd,
            'Pump_electric_W': P_el
        })

    # Apply to GeoDataFrame
    results = gdf_built_pipes.apply(design_pump_and_pipe, axis=1)

    # Join with original dataframe
    gdf_built_pipes = gdf_built_pipes.join(results)

    return gdf_built_pipes
    