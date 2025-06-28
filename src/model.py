# import libraries to work with Pyomo models
from data import *
import logging
import os

from itertools import product
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
import pandas as pd
import geopandas as gpd
import utils

from termcolor import colored, cprint

import time


class HeatNetworkModel(ConcreteModel):

    def __init__(self, scenario_name:str, logger = None):
        super().__init__(name=scenario_name)
        self.data_dict = {}
        self.logging = False

        if logger is not None:
            self.logger = logger
            self.logging = True
    
    
    def fill_model_data(self, input_dict:dict):

        df_heat_demand = input_dict['heat_demand']
        df_heat_gen_units = input_dict['heat_gen_units']
        df_heat_network = input_dict['heat_network']
        df_heat_nodes = input_dict['heat_nodes']
        dict_parameter_cost = input_dict['parameter_cost']

        df_waste_heat_prof = None
        if input_dict['waste_heat_prof'] is not None:
            df_waste_heat_prof = input_dict['waste_heat_prof']
            # check that the waste heat has an associated heat generation unit
            for wh in df_waste_heat_prof.columns[1:]:
                if wh not in df_heat_gen_units.source.unique():
                    self.logger.error(f"Waste heat profile {wh} has no associated heat generation unit")
                    assert False, f"Waste heat profile {wh} has no associated heat generation unit"

        hours = df_heat_demand.hour.values.astype(int)
        hn = df_heat_nodes.node.unique()

        if self.logging:
            self.logger.info("######################")
            self.logger.info("Start: fill model data")
            self.logger.info("######################")
        
        # Define sets
        self.n = hn # Thermal nodes
        self.h = hours # Hourly index

        # General parameters
        self.pCostHNS = Param(initialize=dict_parameter_cost['pCostHNS'])
        self.pCostPumping = Param(initialize=dict_parameter_cost['pCostPumping']) 
        self.pTsupply = Param(initialize=dict_parameter_cost['pTsupply'])
        self.pTreturn = Param(initialize=dict_parameter_cost['pTreturn'])

        self.pPipeCostIni = Param(initialize=dict_parameter_cost['pPipeCostIni'])
        self.pPipeCostSlope = Param(initialize=dict_parameter_cost['pPipeCostsSlope'])
        self.pMassFlowIni = Param(initialize=dict_parameter_cost['pMassFlowIni'])
        self.pAllowDoubleHeating = Param(initialize=dict_parameter_cost['allow_double_heating'])


        # will be replaced by a parameter of heat gen units later
        self.pTESlosses = Param(initialize=0.0)
        
        # Physical parameters, in kWh/m³ water at standard conditions
        self.pCWater = Param(initialize=1.16389)


        start = time.time()

        # dictionary to create the heat network
        dict_heat_network = {}
        for idx, row in df_heat_network.iterrows():
            dict_heat_network[(row.iat[0], row.iat[1])] = row['distance'] 


        if self.logging:
            self.logger.info("Created heat network dictionary - {:.2f}".format(time.time() - start))

        print("Done: create heat network dictionary")
        print("Time: ", time.time() - start)
        start = time.time()

        # use the network dictionary  to create the connections index
        self.pc = Set(initialize=[(n1, n2) for n1, n2 in dict_heat_network.keys()]) # Pipe candidate connections
        self.hg = Set(initialize=df_heat_gen_units.source.unique())  # Heat generation units
        self.hgn = Set(initialize=[(row.source, row.heat_unit_id) for idx, row in df_heat_gen_units.iterrows()])  # Assignment of heat gen units to nodes
        self.tes = Set(within=self.hg, initialize=df_heat_gen_units[df_heat_gen_units.isTES == 1].source.unique())  # Thermal energy storage
        self.hb = Set(within=self.hg, initialize=df_heat_gen_units[df_heat_gen_units.isBoiler == 1].source.unique())  # Heat boilers

        if df_waste_heat_prof is not None:
            self.wh = Set(within=self.hg, initialize=df_waste_heat_prof.columns[1:])  # Waste heat generation units
        else:
            self.wh = Set(within=self.hg)  # Waste heat generation units



        if self.logging:
            self.logger.info("Created heat network dictionary - {:.2f}".format(time.time() - start))

        print("Done: add parameters")
        print("Time: ", time.time() - start)
        start = time.time()

        # add the heat demand data to the model
        dict_cost_dh_connect = {}
        dict_heat_demand = {}
        dict_max_dh_power = {}
        dict_local_heat_prod_cost = {}
        for n in self.n:
            # create connection cost
            dict_cost_dh_connect[n] = df_heat_nodes[df_heat_nodes.node == n].CostsDHconnect.values[0]
            # create max dh power
            dict_max_dh_power[n] = df_heat_nodes[df_heat_nodes.node == n].maxDHPower.values[0]
            # create local heat production cost
            dict_local_heat_prod_cost[n] = df_heat_nodes[df_heat_nodes.node == n].LocalHeatProdCost.values[0]
            # create heat demand
            # code before using demand as rows
            # aux = df_heat_demand.T
            # aux.columns = aux.iloc[0]
            # aux.drop(aux.index[0], inplace=True)
            # aux.reset_index(inplace=True)
            # aux['index'] = aux.iloc[:,0].astype('int16')
            # aux.set_index('index', inplace=True)

            aux_df = df_heat_demand.loc[:, ['hour', n]]

            # for h in self.h:
            #     dict_heat_demand[(n, h)] = df_heat_demand[df_heat_demand.node == n][str(h)].values[0]

            # this could be improved to use itertuples, but it is not a bottleneck right now
            for idx, row in aux_df.iterrows():
                dict_heat_demand[(n, int(row['hour']))] = row[n]


        if self.logging:
            self.logger.info("Created heat demand dictionary - {:.2f}".format(time.time() - start))


        print("Done: created heat demand dictionary")
        print("Time: ", time.time() - start)
        start = time.time()

        self.pHeatDemand = Param(self.n, self.h, initialize=dict_heat_demand, within=NonNegativeReals)
        self.pCostDHConnect = Param(self.n, initialize=dict_cost_dh_connect, within=NonNegativeReals)
        self.pMaxDHPower = Param(self.n, initialize=dict_max_dh_power, within=NonNegativeReals)
        self.pCostLocalHeatProd = Param(self.n, initialize=dict_local_heat_prod_cost,within=NonNegativeReals)
        self.pPipeLength = Param(self.pc, initialize=dict_heat_network, within=NonNegativeReals)


        if self.logging:
            self.logger.info("Added heat demand data - {:.2f}".format(time.time() - start))

        print("Done: add heat demand data")
        print("Time: ", time.time() - start)
        start = time.time()


        dict_waste_heat_prof = {}
        dict_cost_central_heat_prod = {}
        dict_cost_central_heat_prod_inv = {}
        dict_cost_tes_inv = {}
        for hg in self.hg:
            if not(np.isnan(df_heat_gen_units[df_heat_gen_units.source == hg].OMVarCost.values[0])):
                dict_cost_central_heat_prod[hg] = df_heat_gen_units[df_heat_gen_units.source == hg].OMVarCost.values[0]
            if not (np.isnan(df_heat_gen_units[df_heat_gen_units.source == hg].PowerInvCost.values[0])):
                dict_cost_central_heat_prod_inv[hg] = df_heat_gen_units[df_heat_gen_units.source == hg].PowerInvCost.values[0]
            if (df_heat_gen_units[df_heat_gen_units.source == hg].isTES.values[0] == 1 and
                    not (np.isnan(df_heat_gen_units[df_heat_gen_units.source == hg].StorageInvCost.values[0]))):
                dict_cost_tes_inv[hg] = df_heat_gen_units[df_heat_gen_units.source == hg].StorageInvCost.values[0]

            if df_waste_heat_prof is not None:
                if hg in df_waste_heat_prof.columns:
                    for idx, row in df_waste_heat_prof.iterrows():
                        dict_waste_heat_prof[(hg, row['hour'])]  = row[hg]

        if self.logging:
            self.logger.info("Created heat generation dictionaries - {:.2f}".format(time.time() - start))

        print("Done: add heat generation dictionaries")
        print("Time: ", time.time() - start)
        start = time.time()

        self.pMaxWHMF = Param(self.hg, self.h, initialize=dict_waste_heat_prof, within=NonNegativeReals)
        self.pCostCentrHeatProd = Param(self.hg, initialize=dict_cost_central_heat_prod, within=NonNegativeReals)
        self.pCostCentralHeatProdInv = Param(self.hg, initialize=dict_cost_central_heat_prod_inv, within=NonNegativeReals)
        self.pCostTESInv = Param(self.hg, initialize=dict_cost_tes_inv, within=NonNegativeReals)

        if self.logging:
            self.logger.info("Added heat generation data - {:.2f}".format(time.time() - start))

        print("Done: add heat generation parameters")
        print("Time: ", time.time() - start)

        if self.logging:
            self.logger.info("######################")
            self.logger.info("End: fill model data")
            self.logger.info("######################")


    
    def initialize_variables(self):

        if self.logging:
            self.logger.info("###########################")
            self.logger.info("Start: initialize variables")
            self.logger.info("###########################")

        start = time.time()

        # Continuous variables
        #self.vTotalVCost = Var(domain=NonNegativeReals)
        self.vMF = Var(self.pc, self.h, domain=NonNegativeReals)
        self.vHNS = Var(self.n, self.h, domain=NonNegativeReals)
        self.vLocalHeatProd = Var(self.n, self.h, domain=NonNegativeReals)
        self.vMFConsumption = Var(self.n, self.h, domain=NonNegativeReals)
        self.vMFInjection = Var(self.n, self.h, domain=NonNegativeReals)
        self.vCentralHeatProd = Var(self.hg, self.h, domain=NonNegativeReals)
        self.vCentralHeatProdInv = Var(self.hg, domain=NonNegativeReals)
        self.vTESLevel = Var(self.hg, self.h, domain=NonNegativeReals)
        self.vTESCharge = Var(self.hg, self.h, domain=NonNegativeReals)
        self.vTESDischarge = Var(self.hg, self.h, domain=NonNegativeReals)
        self.vTESCapacitivInv = Var(self.hg, domain=NonNegativeReals)
        self.vPipeMassFlowInv = Var(self.pc, domain=NonNegativeReals)

        # Binary variables
        self.vDHconnect = Var(self.n, domain=Binary, initialize=0)
        self.vBinBuildPipe = Var(self.pc, domain=Binary)


        if self.logging:
            self.logger.info("Added variables - {:.2f}".format(time.time() - start))

        print("Done: add variables")
        print("Time: ", time.time() - start)

        if self.logging:
            self.logger.info("###########################")
            self.logger.info("End: initialize variables")
            self.logger.info("###########################")

    
    def initialize_constraints(self):

        if self.logging:
            self.logger.info("#############################")
            self.logger.info("Start: initialize constraints")
            self.logger.info("#############################")


        start = time.time()

        def hourly_cost_function(model, h):
            return (
                sum(model.pCostHNS * model.vHNS[n, h] for n in model.n) +
                sum(model.pCostLocalHeatProd[n] * model.vLocalHeatProd[n, h] for n in model.n) +
                sum(model.pCostCentrHeatProd[hg] * model.vCentralHeatProd[hg, h] for hg in model.hg) +
                sum(model.pCostPumping * model.pPipeLength[n,m] * model.vMF[n, m, h] for (n, m) in model.pc)
            )

        self.hourly_cost = Expression(self.h, rule=hourly_cost_function)

        # Objective function
        def objective_function(model):
            return (
                sum(model.pCostDHConnect[n] * model.vDHconnect[n] for n in model.n) +
                sum(model.pCostCentralHeatProdInv[hg] * model.vCentralHeatProdInv[hg] for hg in model.hg) +
                sum(model.pCostTESInv[tes] * model.vTESCapacitivInv[tes] for tes in model.tes) +
                sum(model.pPipeCostIni * model.pPipeLength[n, m] * model.vBinBuildPipe[n, m] for (n, m) in model.pPipeLength) +
                sum(model.pPipeCostSlope * model.pPipeLength[n, m] * model.vPipeMassFlowInv[n, m] for (n, m) in model.pPipeLength) +  # tbd add variable slope price
                sum(model.hourly_cost[h] for h in model.h)
            )
        self.obj = Objective(rule=objective_function, sense=minimize)

        if self.logging:
            self.logger.info("Added objective function - {:.2f}".format(time.time() - start))

        print("Done: add objective function")
        print("Time: ", time.time() - start)

        start = time.time()
        # Energy balance in kW for ever heat node and hour
        def energy_balance_rule(model, n, h):
            return (
                model.vMFConsumption[n, h] == (model.pHeatDemand[n, h] - model.vLocalHeatProd[n, h]) / ((model.pTsupply - model.pTreturn) * model.pCWater)
            )
        self.energy_balance = Constraint(self.n, self.h, rule=energy_balance_rule)

        if self.logging:
            self.logger.info("Added energy balance - {:.2f}".format(time.time() - start))

        print("Done: add energy balance")
        print("Time: ", time.time() - start)

        start = time.time()
        # Limits use of distrct heating only to connected nodes and to a maximum power (as a specific bigM)
        def max_dh_power_rule(model, n, h):
            return (
                model.vMFConsumption[n, h] <= model.vDHconnect[n] * model.pMaxDHPower[n] / ((model.pTsupply - model.pTreturn) * model.pCWater)
            )
        self.max_dh_power = Constraint(self.n, self.h, rule=max_dh_power_rule)

        if self.logging:
            self.logger.info("Added max. district heating consumption - {:.2f}".format(time.time() - start))


        print("Done: add max dh power")
        print("Time: ", time.time() - start)

        start = time.time()
        # Mass flow balacne between all nodes
        def mf_balance_rule(model, n, h):
            aux_pc_out = set((i, j) for i, j in model.pc if i == n)
            aux_pc_inf = set((j, i) for j, i in model.pc if i == n)
            return (
                model.vMFConsumption[n, h] +
                sum(model.vTESCharge[tes, h] for tes in model.tes if (tes, n) in model.hgn) -
                model.vMFInjection[n, h] ==
                # sum(model.vMF[m, n, h] for m in model.n if (m, n) in aux_pc) -
                sum(model.vMF[m, n, h] for m, _ in aux_pc_inf) -
                # sum(model.vMF[n, m, h] for m in model.n if (n, m) in aux_pc)
                sum(model.vMF[n, m, h] for _, m in aux_pc_out)
            )
        self.mf_balance = Constraint(self.n, self.h, rule=mf_balance_rule)

        if self.logging:
            self.logger.info("Added mass flow balance system - {:.2f}".format(time.time() - start))

        print("Done: add mf balance")
        print("Time: ", time.time() - start)

        start = time.time()
        # Mass flow injection into the network
        def mf_injection_rule(model, n, h):
            return (
                model.vMFInjection[n,h] ==
                sum(model.vCentralHeatProd[wh, h] for wh in model.wh if (wh, n) in model.hgn) / ((model.pTsupply - model.pTreturn) * model.pCWater) +
                sum(model.vCentralHeatProd[hb, h] for hb in model.hb if (hb, n) in model.hgn) / ((model.pTsupply - model.pTreturn) * model.pCWater) +
                sum(model.vCentralHeatProd[tes, h] for tes in model.tes if (tes, n) in model.hgn) / ((model.pTsupply - model.pTreturn) * model.pCWater)
            )
        self.mf_injection = Constraint(self.n, self.h, rule=mf_injection_rule)

        if self.logging:
            self.logger.info("Added mass flow balance node - {:.2f}".format(time.time() - start))

        print("Done: add mf injection")
        print("Time: ", time.time() - start)


        if self.pAllowDoubleHeating == 0:
            start = time.time()
            # Exclude decentral heating for nodes that are connected to the network
            def decentral_condition_rule(model, n, h):
                return (
                    model.vLocalHeatProd[n, h] <= (1 - model.vDHconnect[n]) * model.pMaxDHPower[n]
                )
            self.decentral_condition = Constraint(self.n, self.h, rule=decentral_condition_rule)

            if self.logging:
                self.logger.info("Added local heating - {:.2f}".format(time.time() - start))

            print('Info: No decentral heating allowed when building connected to the network')
            print("Done: add decentral condition")
            print("Time: ", time.time() - start)
        else:
            print('Info: Decentral heating allowed when building connected to the network!')


        start = time.time()
        # Limit the maximum power of the waste heat generation units
        def max_wh_power_rule(model, wh, h):
            return (
                model.vCentralHeatProd[wh, h] <= model.pMaxWHMF[wh, h] * (model.pTsupply - model.pTreturn) * model.pCWater
            )
        self.max_wh_power = Constraint(self.wh, self.h, rule=max_wh_power_rule)

        if self.logging:
            self.logger.info("Added max. waste heat - {:.2f}".format(time.time() - start))

        print("Done: add max wh power")
        print("Time: ", time.time() - start)

        start = time.time()
        # calculate the heat production of the TES
        def tes_production_rule(model, tes, h):
            return (
                model.vCentralHeatProd[tes, h] == model.vTESDischarge[tes, h] * (model.pTsupply - model.pTreturn) * model.pCWater
            )
        self.tes_production = Constraint(self.tes, self.h, rule=tes_production_rule)

        if self.logging:
            self.logger.info("Added thermal storage production - {:.2f}".format(time.time() - start))

        print("Done: add tes production")
        print("Time: ", time.time() - start)

        start = time.time()
        # Limit the maximum power to investments
        def max_power_invest_rule(model, hg, h):
            return (
                model.vCentralHeatProd[hg, h] <= model.vCentralHeatProdInv[hg]
            )
        self.max_power_invest = Constraint(self.hg, self.h, rule=max_power_invest_rule)

        if self.logging:
            self.logger.info("Added max. power invest - {:.2f}".format(time.time() - start))

        print("Done: add max power invest")
        print("Time: ", time.time() - start)

        start = time.time()
        # Limit the size of the storage to its investments
        def tes_storage_invest_rule(model, tes, h):
            return (
                model.vTESLevel[tes,h] <= model.vTESCapacitivInv[tes]
            )
        self.tes_storage_invest = Constraint(self.tes, self.h, rule=tes_storage_invest_rule)

        if self.logging:
            self.logger.info("Added max. thermal storage invest - {:.2f}".format(time.time() - start))

        print("Done: add tes storage invest")
        print("Time: ", time.time() - start)

        start = time.time()
        # Balance equation for the TES
        def tes_balance_rule(model, tes, h):
            if h == 1: return model.vTESLevel[tes, h] == model.vTESLevel[tes, len(model.h)] + model.vTESCharge[tes, h] - model.vTESDischarge[tes, h]
            return (
                model.vTESLevel[tes, h] == model.vTESLevel[tes, h-1] * (1 - model.pTESlosses) + model.vTESCharge[tes, h] - model.vTESDischarge[tes, h]
            )
        self.tes_balance = Constraint(self.tes, self.h, rule=tes_balance_rule)

        if self.logging:
            self.logger.info("Added thermal storage balance - {:.2f}".format(time.time() - start))

        print("Done: add tes balance")
        print("Time: ", time.time() - start)

        start = time.time()
        # Limit mass flow through the pipes
        def max_mass_flow_rule(model, n, m, h):
            return (
                model.vMF[n, m, h] <= model.vBinBuildPipe[n, m] * model.pMassFlowIni + model.vPipeMassFlowInv[n, m]
            )
        self.max_mass_flow = Constraint(self.pc, self.h, rule=max_mass_flow_rule)

        def logic_mass_flow_rule(model, n, m):
            return (
                model.vPipeMassFlowInv[n, m] <= model.vBinBuildPipe[n, m] * 1e5
            )
        self.logic_mass_flow = Constraint(self.pc, rule=logic_mass_flow_rule)

        if self.logging:
            self.logger.info("Added mass flow pipe - {:.2f}".format(time.time() - start))

        print("Done: build max mass flow")
        print("Time: ", time.time() - start)

        if self.logging:
            self.logger.info("#############################")
            self.logger.info("End: initialize constraints")
            self.logger.info("#############################")

        
    def model_run(self):

        if self.logging:
            self.logger.info("#############################")
            self.logger.info("Start: model solve")
            self.logger.info("#############################")

        start = time.time()
        print("Start: solve model")
        # Solve the model
        #opt = SolverFactory('glpk', executable = 'C:\\GLPK\\winglpk-4.65\\glpk-4.65\\w64\\glpsol')  # Replace with your preferred solver
        #opt.options['mipgap'] = 0.01
        opt = SolverFactory('gurobi')  # Replace with your preferred solver
        opt.gurobi_options = "mipgap=0.01"  # Set the mipgap to 1.0% to speed up the calculation
        opt.solve(self, tee=True)
        print("Done: solve model")
        print("Time: ", time.time() - start)

        if self.logging:
            self.logger.info("#############################")
            self.logger.info("End: model solve")
            self.logger.info("#############################")


    def export_results(self, case_study_name, model_name, config):
        print("Start: export results")
        # convert the DH connection to a dataframe
        df_dh_connection = pd.DataFrame()
        df_dh_connection['Node'] = self.n
        df_dh_connection['DH Connection'] = [self.vDHconnect[n].value for n in self.n]

        # convert the pipe connections to a dataframe
        df_pipe_connections = pd.DataFrame()
        df_pipe_connections['from'] = [n for (n, m) in self.pc]
        df_pipe_connections['to'] = [m for (n, m) in self.pc]
        df_pipe_connections['Build Pipe'] = [self.vBinBuildPipe[n, m].value for (n, m) in self.pc]
        df_pipe_connections['Max Mass Flow'] = [max(self.vMF[n, m, h].value for h in self.h) for (n, m) in self.pc]
        df_pipe_connections['Mass Flow Investment'] = [self.vBinBuildPipe[n, m].value * self.pMassFlowIni + self.vPipeMassFlowInv[n, m].value for (n, m) in self.pc]

        # convert the heat generation time series to a dataframe
        data = []
        for hg in self.hg:
            for h in self.h:
                data.append([hg, h, self.vCentralHeatProd[hg, h].value])

        df_heat_gen = pd.DataFrame(data, columns=['Generation Unit', 'Hour', 'Heat Production'])
        df_heat_gen = df_heat_gen.pivot(index='Generation Unit', columns='Hour', values='Heat Production').reset_index()

        df_heat_gen.loc['Local Heat Production', 'Generation Unit'] = 'Dezentral Heat Production'
        for h in self.h:
            df_heat_gen.loc['Local Heat Production', h] = sum(self.vLocalHeatProd[n, h].value for n in self.n)

     
        # write economic results to a dataframe
        df_economic_results = pd.DataFrame({
            'Costs in k€': [
                'Total Costs',
                'DH Connection Cost',
                'HNS Cost',
                'Local Heat Production Cost',
                'Central Heat Production Cost',
                'Central Heat Production Investment Cost',
                'TES Investment Cost',
                'Pumping Cost',
                'Pipe Base Investment Cost',
                'Pipe Slope Investment Cost'
            ],
            'Value': [
                round(value(self.obj)/1e3,1),
                round(sum(self.pCostDHConnect[n] * self.vDHconnect[n].value for n in self.n if self.vDHconnect[n].value is not None)/1e3,1),
                round(sum(self.pCostHNS * self.vHNS[n, h].value for n in self.n for h in self.h if self.vHNS[n, h].value is not None)/1e3,1),
                round(sum(self.pCostLocalHeatProd[n] * self.vLocalHeatProd[n, h].value for n in self.n for h in self.h if self.vLocalHeatProd[n, h].value is not None)/1e3,1),
                round(sum(self.pCostCentrHeatProd[hg] * self.vCentralHeatProd[hg, h].value for hg in self.hg for h in self.h if   self.vCentralHeatProd[hg, h].value is not None)/1e3,1),
                round(sum(self.pCostCentralHeatProdInv[hg] * self.vCentralHeatProdInv[hg].value for hg in self.hg if self.vCentralHeatProdInv[hg].value is not None)/1e3,1),
                round(sum(self.pCostTESInv[tes] * self.vTESCapacitivInv[tes].value for tes in self.tes if self.vTESCapacitivInv[tes].value is not None)/1e3,1),
                round(sum(self.pCostPumping * self.pPipeLength[n,m] * self.vMF[n, m, h].value for (n, m) in self.pc for h in self.h if self.vMF[n, m, h].value is not None)/1e3,1),
                round(sum(self.pPipeCostIni * self.pPipeLength[n, m] * self.vBinBuildPipe[n, m].value for (n, m) in self.pc if self.vBinBuildPipe[n, m].value is not None)/1e3,1),
                round(sum(self.pPipeCostSlope * self.pPipeLength[n, m] * self.vPipeMassFlowInv[n, m].value for (n, m) in self.pc if self.vPipeMassFlowInv[n, m].value is not None)/1e3,1)
            ]
        })

        # write the investment decisions per technology to a dataframe
        df_investment_decisions = pd.DataFrame(columns=['Generation Unit', 'Type', 'Capacity Investments / kW', 'Storage Investments / m3'])
        # add if the generation unit is a boiler, waste heat or TES
        for hg in (self.hg):
            df_investment_decisions.loc[hg,'Generation Unit'] = hg
            df_investment_decisions.loc[hg,'Type'] = 'Boiler' if hg in self.hb else 'Waste Heat' if hg in self.wh else 'TES' if hg in self.tes else 'Central Heat'
            df_investment_decisions.loc[hg,'Capacity Investments / kW'] = self.vCentralHeatProdInv[hg].value
            df_investment_decisions.loc[hg,'Storage Investments / m3'] = self.vTESCapacitivInv[hg].value if hg in self.tes else 0

        CaseStudies_dir = config['case_studies_dir']
        output_folder = os.path.join(CaseStudies_dir,case_study_name, config['scenario_dir'], model_name, config['expost_dir'])
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


        # write the economic results to a csv file
        df_economic_results.to_csv(os.path.join(output_folder, 'Economic_Results.csv'), sep=';', index=False)
        df_investment_decisions.to_csv(os.path.join(output_folder, 'Investment_Decisions.csv'), sep=';', index=False)
        # write the results to a csv file
        df_dh_connection.to_csv(os.path.join(output_folder,'DH_Connection.csv'), sep=';', index=False)
        df_pipe_connections.to_csv(os.path.join(output_folder,'Pipe_Connections.csv'), sep=';', index=False)
        df_heat_gen.to_csv(os.path.join(output_folder,'Heat_Generation_TS.csv'), sep=';', index=False)

        print("Done: export results")


    def export_model_variables(self):
        """"
        This function exports the variables from a pyomo model to a dictionary of dataframes. Does not depend
        on the model's structure, all the data is extracted from the model's components
        """
        # a dictionary of dataframes to store all the variables data, includingg their indices and values.
        # The keys are the variables' names
        d_vars = {}
        # get all the variables from the model
        variables = self.component_objects(Var)
        for v in variables:
            # get the variable's representation inside the model
            var_object = getattr(self, str(v))
            # create a dictionary to store the variable's data: its index set and its value
            aux_d = {}
            aux_d['value'] = []

            # get the indices of the variable
            idx_elements = list(var_object.index_set())
            aux_data = v.get_values()
            aux_data_df = pd.DataFrame({'indices': list(aux_data.keys()), 'value': list(aux_data.values())})
            # if it is a tuple, the variable has more than one index set
            if isinstance(idx_elements[0], tuple):
                # count the number of index sets
                len_test_elem = len(idx_elements[0])
                aux_cols_idx = ['index_set_' + str(i+1) for i in range(0, len_test_elem)]
                aux_data_df[aux_cols_idx] = pd.DataFrame(aux_data_df['indices'].tolist(), index=aux_data_df.index)
                aux_data_df.drop(columns=['indices'], inplace=True)

            # if it is not a tuple, the variable one index set and we can the values directly
            else:
                aux_d['index_set_1'] = []
                aux_cols_idx = ['index_set_1']
                aux_data_df[aux_cols_idx] = pd.DataFrame(aux_data_df['indices'].tolist(), index=aux_data_df.index)
                aux_data_df.drop(columns=['indices'], inplace=True)

            # store the dataframe in the dictionary with all the other variables
            d_vars[str(v)] = aux_data_df

        return d_vars

    def export_model_parameters(self):
        # a dictionary of dataframes to store all the variables data, including their indices and values.
        # The keys are the variables' names
        d_params = {}
        # get all the variables from the model
        parameters = self.component_objects(Param)
        for p in parameters:
            param_object = getattr(self, str(p))
            # create a dictionary to store the variable's data: its index set and its value
            aux_d = {}
            aux_d['value'] = []

            # get the indices of the variable
            idx_elements = list(param_object.index_set())
            aux_data = p.values()
            aux_data_df = pd.DataFrame({'indices': idx_elements, 'value': list(aux_data)})
            # if it is a tuple, the variable has more than one index set
            if isinstance(idx_elements[0], tuple):
                # count the number of index sets
                len_test_elem = len(idx_elements[0])
                aux_cols_idx = ['index_set_' + str(i+1) for i in range(0, len_test_elem)]
                aux_data_df[aux_cols_idx] = pd.DataFrame(aux_data_df['indices'].tolist(), index=aux_data_df.index)
                aux_data_df.drop(columns=['indices'], inplace=True)

            # if it is not a tuple, the variable one index set and we can the values directly
            else:
                aux_d['index_set_1'] = []
                aux_cols_idx = ['index_set_1']
                aux_data_df[aux_cols_idx] = pd.DataFrame(aux_data_df['indices'].tolist(), index=aux_data_df.index)
                aux_data_df.drop(columns=['indices'], inplace=True)

            # store the dataframe in the dictionary with all the other variables
            d_params[str(v)] = aux_data_df

        return d_params
    

def run_model(case_study_name, model_name, config):
    logger = utils.create_logger(case_study_name + "_" + model_name, config['log_dir'])
    dh_model = HeatNetworkModel(case_study_name, logger)
    input_dict = load_data_from_disk(case_study_name,model_name, config)


    error = False
    if input_dict['heat_nodes'] is None:
        logger.error("Heat nodes data is missing")
        error = True
    if input_dict['heat_gen_units'] is None:
        logger.error("Heat generation units data is missing")
        error = True
        assert False, "Heat generation units data is missing"
    if input_dict['heat_demand'] is None:
        logger.error("Heat demand data is missing")
        error = True
        assert False, "Heat demand data is missing"
    if input_dict['waste_heat_prof'] is None:
        logger.warning("Waste heat profile data is missing")

    if error:
        assert False, "Missing input data (e.g., heat nodes, heat generation units, heat demand)"

    input_dict['heat_nodes'].rename(columns={'cluster_id': 'node'}, inplace=True)
    input_dict['heat_nodes'].fillna(0, inplace=True)
    input_dict['heat_gen_units'].rename(columns={'unit': 'source'}, inplace=True)
    input_dict['heat_gen_units'].rename(columns={'O&M and Fuel Costs': 'OMVarCost'}, inplace=True)
    input_dict['heat_gen_units'].rename(columns={'Power Investment Costs': 'PowerInvCost'}, inplace=True)
    input_dict['heat_gen_units'].rename(columns={'Storage Investment Costs': 'StorageInvCost'}, inplace=True)

    if input_dict['waste_heat_prof'] is not None:
        input_dict['waste_heat_prof'].rename(columns={'unit': 'source'}, inplace=True)

    # Even zero demand is needed to create the balance at each node
    heat_gen_units = input_dict['heat_gen_units'].source.values

    for hg in heat_gen_units:
        input_dict['heat_demand']['heat_unit_'+str(hg)] = 0

    dh_model.fill_model_data(input_dict)
    dh_model.initialize_variables()
    dh_model.initialize_constraints()
    dh_model.model_run()

    res_dict = dh_model.export_model_variables()
    CaseStudies_dir = config['CaseStudies_dir']
    output_folder = os.path.join(CaseStudies_dir, case_study_name, config['scenario_dir'], model_name, config['output_dir'])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for k, v in res_dict.items():
        v.to_csv(os.path.join(output_folder, k + '.csv'), sep=',', index=False)
        print("Exported: ", k)

    dh_model.export_results(case_study_name, model_name, config)

    logging.shutdown()
    
    return None



if __name__ == "__main__":
    # dh_model = model_run()
    # vars = export_model_variables(dh_model)
    case_study_name = 'Puertollano_open_data'
    model_name = 'realistic_costs'
    config = load_config()
    logger = utils.create_logger(case_study_name + "_" + model_name, config['log_dir'])
    dh_model = HeatNetworkModel(case_study_name, logger)
    input_dict = load_data_from_disk(case_study_name,model_name, config)


    error = False
    if input_dict['heat_nodes'] is None:
        logger.error("Heat nodes data is missing")
        error = True
    if input_dict['heat_gen_units'] is None:
        logger.error("Heat generation units data is missing")
        error = True
        assert False, "Heat generation units data is missing"
    if input_dict['heat_demand'] is None:
        logger.error("Heat demand data is missing")
        error = True
        assert False, "Heat demand data is missing"
    if input_dict['waste_heat_prof'] is None:
        logger.warning("Waste heat profile data is missing")

    if error:
        assert False, "Missing input data (e.g., heat nodes, heat generation units, heat demand)"

    input_dict['heat_nodes'].rename(columns={'cluster_id': 'node'}, inplace=True)
    input_dict['heat_nodes'].fillna(0, inplace=True)
    input_dict['heat_gen_units'].rename(columns={'unit': 'source'}, inplace=True)
    input_dict['heat_gen_units'].rename(columns={'O&M and Fuel Costs': 'OMVarCost'}, inplace=True)
    input_dict['heat_gen_units'].rename(columns={'Power Investment Costs': 'PowerInvCost'}, inplace=True)
    input_dict['heat_gen_units'].rename(columns={'Storage Investment Costs': 'StorageInvCost'}, inplace=True)

    if input_dict['waste_heat_prof'] is not None:
        input_dict['waste_heat_prof'].rename(columns={'unit': 'source'}, inplace=True)

    # Even zero demand is needed to create the balance at each node
    heat_gen_units = input_dict['heat_gen_units'].source.values

    for hg in heat_gen_units:
        input_dict['heat_demand']['heat_unit_'+str(hg)] = 0
    # input_dict['heat_demand']['heat_unit_Electrolyser'] = 0
    # input_dict['heat_demand']['heat_unit_ElectricBoiler'] = 0
    # input_dict['heat_demand']['heat_unit_ThermalEnergyStorage'] = 0
    # input_dict['heat_demand']['heat_unit_ThermalEnergyStorage_2'] = 0

    dh_model.fill_model_data(input_dict)
    dh_model.initialize_variables()
    dh_model.initialize_constraints()
    #
    #
    dh_model.model_run()

    res_dict = dh_model.export_model_variables()
    CaseStudies_dir = config['CaseStudies_dir']
    output_folder = os.path.join(CaseStudies_dir,case_study_name, config['scenario_dir'], model_name, config['output_dir'])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for k, v in res_dict.items():
        v.to_csv(os.path.join(output_folder, k + '.csv'), sep=',', index=False)
        print("Exported: ", k)


    dh_model.export_results()

    logging.shutdown()
    ### Comand line output for debugging

    # for item in dh_self.vMF:
    #     if dh_self.vMF[item]() > 0:
    #         print(item, dh_self.vMF[item]())

    # for item in dh_self.vLocalHeatProd:
    #     if dh_self.vLocalHeatProd[item]() > 0:
    #         print(item, dh_self.vLocalHeatProd[item]())

    # for item in dh_self.vMFConsumption:
    #     if dh_self.vMFConsumption[item]() > 0:
    #         print(item, dh_self.vMFConsumption[item]())

    # # Print the results
    # print("DH Connection: ", dh_self.vDHconnect.get_values())
    # print("Investments: ", dh_self.vCentralHeatProdInv.get_values())
    # print("TES Invest: ",dh_self.vTESCapacitivInv.get_values())

    # # convert results into a dataframe
    # df_results = pd.DataFrame()
    # df_results['CentralHeatProd'] = [dh_self.vCentralHeatProd[hg, h].value for hg in dh_self.hg for h in dh_self.h]

    # print("Total central production: ", df_results['CentralHeatProd'].sum())

    # print("TES Level: ", dh_self.vTESLevel.get_values())

    # print("Piping costs: ", dh_self.pCostPipeInv.extract_values())

    # # Calculate the cost contribion of the different components
    # print("Cost DH Connection: ", sum(dh_self.pCostDHConnect[n] * dh_self.vDHconnect[n].value for n in dh_self.n if dh_self.vDHconnect[n].value is not None))
    # print("Cost HNS: ", sum(dh_self.pCostHNS * dh_self.vHNS[n, h].value for n in dh_self.n for h in dh_self.h if dh_self.vHNS[n, h].value is not None))
    # print("Cost Local Heat Production: ", sum(dh_self.pCostLocalHeatProd[n] * dh_self.vLocalHeatProd[n, h].value for n in dh_self.n for h in dh_self.h if dh_self.vLocalHeatProd[n, h].value is not None))
    # print("Cost Central Heat Production: ", sum(dh_self.pCostCentrHeatProd[hg] * dh_self.vCentralHeatProd[hg, h].value for hg in dh_self.hg for h in dh_self.h if dh_self.vCentralHeatProd[hg, h].value is not None))
    # print("Cost Central Heat Production Investment: ", sum(dh_self.pCostCentralHeatProdInv[hg] * dh_self.vCentralHeatProdInv[hg].value for hg in dh_self.hg if dh_self.vCentralHeatProdInv[hg].value is not None))
    # print("Cost TES Investment: ", sum(dh_self.pCostTESInv[tes] * dh_self.vTESCapacitivInv[tes].value for tes in dh_self.tes if dh_self.vTESCapacitivInv[tes].value is not None))
    # print("Cost Pumping: ", sum(dh_self.pCostPumping * dh_self.vMF[n, m, h].value for (n, m) in dh_self.pc for h in dh_self.h if dh_self.vMF[n, m, h].value is not None))
    # #print("Cost Pipe Investment: ", sum(dh_self.pCostPipeInv[n, m] * dh_self.vBinBuildPipe[n, m].value for (n, m) in dh_self.pc if dh_self.vBinBuildPipe[n, m].value is not None))
    # print(dh_self.pc.data())
    # print(dh_self.vBinBuildPipe.get_values())