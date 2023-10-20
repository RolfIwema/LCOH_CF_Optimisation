# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import pulp
import matplotlib as mp
import pandas as pd
import time
import os


# Record running time for performance analysis

start_time = time.time()
t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)


#%% Function definitions

def Component_costs(CAPEX,OPEX,Lifetime,country): 
    """This function calculates the Net Present Costs of a component"""
    Salvage =  (1-General_data['Project Economics'].loc['Salvage Value',country])
    Lifetime = Lifetime-1                                                       # Account for start value 0
    C_annual = []
    C_cap = []
    C_var = []
    for j in range(int(Project.loc['Project lifetime',country])):
          if j == 0:                                                            # First installation in project cycle
              C_ID_i = CAPEX     
              C_OM_i = (CAPEX * OPEX)/((1+i_r)**j )
          elif j % (Lifetime+1) == 0 and j != 0:                                # Installation new component
              C_ID_i = (Salvage*CAPEX)/((1+i_r)**j )
              C_OM_i = (CAPEX * OPEX)/((1+i_r)**j )
              year_rep = j
          elif j == Project.loc['Project lifetime',country] -1:                 # End of life
              if Lifetime+1 >= Project.loc['Project lifetime',country]:
                  C_ID_i = (((Project.loc['Project lifetime',country]-(Lifetime+1))/(Lifetime+1))*CAPEX)/((1+i_r)**j )    
              else:
                  C_ID_i = ((((j-year_rep)-(Lifetime+1))/(Lifetime+1))*CAPEX)/((1+i_r)**j ) 
          else:                                                                 # Standard year
              C_OM_i = (CAPEX * OPEX)/((1+i_r)**j )
              C_ID_i = 0

          C_j = C_ID_i+C_OM_i
          C_annual.append(C_j)
          C_cap.append(C_ID_i)
          C_var.append(C_OM_i)
          
    Total_C = sum(C_annual) 
    Total_cap = sum(C_cap)
    Total_var = sum(C_var)
    return [Total_C, Total_cap, Total_var]
         
def Compressibility_Factor(Pressure,Temperature):   # 
   """This function calculates the compressibility factor of Hydrogen"""
   Temp =np.array([-150, -125, -100, -75, -50, -25, 0, 25, 50, 75, 100, 125])          #Celcius
   Press =np.array(           [0.001, 0.1 ,   1,    5,     10,    30,    50,  100])    #MPa
   CompFact_table = np.array([[1,1.0003,1.0036,1.0259,1.0726,1.3711,1.7167, 0],        #-150 C
                             [1,1.0006,1.0058,1.0335,1.0782,1.3231,1.6017,2.2856],     #-125 C
                             [1,1.0007,1.0066,1.0356,1.0778,1.2880,1.5216,2.1006],     #-100 C
                             [1,1.0007,1.0068,1.0355,1.0751,1.2604,1.4620,1.9634],     #-75 C
                             [1,1.0007,1.0067,1.0344,1.0714,1.2377,1.4153,1.8572],     #-50 C
                             [1,1.0006,1.0065,1.0329,1.0675,1.2186,1.3776,1.7725],     #-25 C
                             [1,1.0006,1.0062,1.0313,1.0637,1.2022,1.3462,1.7032],     #0 C
                             [1,1.0006,1.0059,1.0297,1.0601,1.1879,1.3197,1.6454],     #25 C
                             [1,1.0006,1.0056,1.0281,1.0567,1.1755,1.2969,1.5964],     #50 C
                             [1,1.0005,1.0053,1.0266,1.0536,1.1644,1.2770,1.5542],     #75 C
                             [1,1.0005,1.0050,1.0252,1.0507,1.1546,1.2596,1.5175],     #100 C
                             [1,1.0005,1.0048,1.0240,1.0481,1.1458,1.2441,1.4852]] )   #125 C
   F_c1 = interpolate.RectBivariateSpline(Press,Temp,CompFact_table.T)
   F_c = F_c1(Pressure,Temperature-273.15)
   if F_c <= 0:
       print('Compressibility Factor Fail')
       1/F_c
   else: 
       return F_c[0]

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

#%% Basic Parameters
#print("Define parameters")

year = 365.25;      # days
day = 24;           # hours

Kelvin = 273.15     # K
T_amb = 30+Kelvin   # K Ambient temperature
T_norm = 0+Kelvin   # Kelvin normal conditions

Ndens = 0.08375     # kg/Nm^3 density of Hydrogen at normal conditions
R = 8.314/1000      # MPa*m^3/(kg*mol*K)
G = 0.0696          # Specific Gravity
m = 2.016           # g/ml

P_norm = 0.101352   # Mpa normal conditions    


#%% Set-up model


# Get file path
current_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = "Model Input.xlsx"
absolute_path = os.path.join(current_directory, relative_path)

Model_General = pd.read_excel(absolute_path, sheet_name="Model_general")
Solar_data = pd.read_excel(absolute_path, sheet_name="Country inputs - Solar")
Wind_data = pd.read_excel(absolute_path, sheet_name="Country inputs - Wind")
Time = Wind_data['Country:'].values[3:]

# delete empty columns
cols_to_drop = Model_General.columns[Model_General.apply(lambda col: all(pd.isna(x) or x == 0 for x in col))]
Model_General = Model_General.drop(columns=cols_to_drop)

## Set up solar data
Solar_data.drop(index=[0,1],columns=['Unnamed: 0', 'Unnamed: 1','Country:'], inplace=True)
I_mean = Solar_data.loc[[2]]
Solar_data = Solar_data.iloc[1:].div(1000)
Solar_data.reset_index(drop=True, inplace=True)
I_mean.reset_index(drop=True, inplace=True)

## Set up Wind data
Wind_data.drop(columns=['Unnamed: 0', 'Unnamed: 1','Country:'], inplace=True)
Wind_option = Wind_data.loc[:1].copy()
Wind_option=Wind_option.loc[[0]]
Mean_windspeed = Wind_data.loc[[2]].copy()
Mean_windspeed.reset_index(drop=True, inplace=True)
Capacity_Turbine = Wind_data.loc[[1]]
Wind_data[3:] /= 1000*Capacity_Turbine.iloc[0]
Wind_data = Wind_data[3:]
Wind_data.reset_index(drop=True, inplace=True)

# Separate tables from single excel sheet

nan_check = Model_General.columns[1]
nan_mask = pd.isnull(Model_General[nan_check])
row_indices_with_nan = Model_General.index[nan_mask].tolist()

Project = Model_General.iloc[:row_indices_with_nan[0]]
Project.set_index('General', inplace=True)
Model_General['dataframe_name'] = np.nan
Model_General.loc[row_indices_with_nan, 'dataframe_name'] = Model_General.loc[row_indices_with_nan, 'General']
Model_General['dataframe_name'].fillna(method='ffill', inplace=True)
General_data = {name: group for name, group in Model_General.groupby('dataframe_name')}
for df_name, df in General_data.items():
    General_data[df_name] = df.drop(columns=['dataframe_name'])
for df_name, df in General_data.items():
    General_data[df_name] = df.dropna()
for df_name, df in General_data.items():
    df.set_index('General', inplace=True)
    df.index.name = df_name


PV_systems = {}
Wind_systems = {}
Electrolysers = {}   
for df_name, df in General_data.items():
    
    if df_name.startswith('Solar_'):
        PV_systems[df_name[len('Solar_'):]] = df  
    elif df_name.startswith('Wind_'):
        Wind_systems[df_name[len('Wind_'):]] = df
    elif df_name.startswith('Electrolyser_'):
        Electrolysers[df_name[len('Electrolyser_'):]] = df
        
for df_name in list(General_data.keys()):
    if df_name.startswith('Solar_') or df_name.startswith('Wind_') or df_name.startswith('Electrolyser_'):
        General_data.pop(df_name)


Countries = Wind_data.columns.tolist()
Electrolyser_types = list(Electrolysers.keys())

# Set-up PV data

PV_types = list(PV_systems.keys())
efficiency_cdte = PV_systems['CdTe'].loc['Efficiency']  # Solar power data is calculated based on CDTe panels, output should therefore be scaled relative to CDTe.

for pv in PV_types:
    if pv != 'CdTe':
        PV_systems[pv].loc['Efficiency'] /= efficiency_cdte[0]
PV_systems['CdTe'].loc['Efficiency'] /= efficiency_cdte[0]

for pv in PV_types:
    for country in Countries:
        CarbonFootprint_PVtype = 10**(64.3 - 0.03088*Project.loc['Project Start',country]+PV_systems[pv].loc['Carbon Footprint: beta',country]+PV_systems[pv].loc['Carbon Footprint: beta_Y',country]*Project.loc['Project Start',country]-0.000232*I_mean.loc[0,country])
        PV_systems[pv].loc['Carbon Footprint total (kg CO2/MWh)',country] = CarbonFootprint_PVtype
        PV_systems[pv].loc['Yield_solar',country] = PV_systems[pv].loc['Efficiency',country]*Solar_data[country].sum()/3    

# Set-up Wind turbine choice
Wind_choice = pd.DataFrame()
for country in Countries:
    choice = Wind_option.at[0,country]
    Wind_choice[country]=Wind_systems[choice][country]
    CarbonFootprint_Windtype = 10**(2.41-0.20*Mean_windspeed.loc[0,country]-5.9*10**-4*Wind_choice.loc['Rated turbine power',country]-8.3*10**-5*Wind_choice.loc['Hub height',country]+5.3*10**-3*Wind_choice.loc['# Turbines',country]+0.33*Wind_choice.loc['Offshore?',country])
    Wind_choice.loc['Carbon Footprint total (kg CO2/MWh)',country] = CarbonFootprint_Windtype
    Wind_choice.loc['Yield_wind',country] = Wind_data[country].sum()/3 

# Project data
       
Project.loc['Delivery pressure'].div(10)   # [MPa] Delivery pressure
General_data['Centrifugal'].loc['H2 leakage']= General_data['Centrifugal'].loc['H2 leakage'].div(100) # [-] Hydrogen leakage
Project.loc['Hydrogen_required'] = round(1000*Project.loc['Annual production']/(1-General_data['Centrifugal'].loc['H2 leakage']))*3 # 3 years of data


#Economics
General_data['Project Economics']/=100


# Set-up variables for different scenarios
Scenarios = {}
Sensitivity = ['N','CS+','CW+','CE+','CS-','CW-','CE-','CFS+','CFW+','CFE+','CFS-','CFW-','CFE-','PC+','PC-','r-','r+']


#%% Scenario set up
for country in Countries:
    Scenarios[country]={}
    for electrolyser in Electrolyser_types:
        Scenarios[country][electrolyser]={}
        
        Electrolysers[electrolyser].loc['Stack investment cost',country]/=100
        Electrolysers[electrolyser].loc['Energy required'] = round(Project.loc['Hydrogen_required']*Electrolysers[electrolyser].loc['Power consumption'])
        Electrolysers[electrolyser].loc['Max. pressure',country] = Electrolysers[electrolyser].loc['Max. pressure',country]/10       # [MPa] Operating Pressure
        
        if Electrolysers[electrolyser].loc['Max. pressure',country] >= Project.loc['Delivery pressure',country]:
            Electrolysers[electrolyser].loc['Compressor Constant',country]=0
        else:        
            # Set-up Compressor 
            T_operation = Electrolysers[electrolyser].loc['Operating temperature',country]+Kelvin   # [K] Temperature in operation
            T_delivery = T_amb                                                                      # [K] Temperature at delivery
            efficiency_isen = General_data['Centrifugal'].loc['Isentropic efficiency',country]/100  # [-] isentropic efficiency 
            efficiency_driver = General_data['Centrifugal'].loc['Driver efficiency',country]/100    # [-] efficiency driver
            k = 1.41                                                                                # [-] Cp/Cv = 1.41 for Hydrogen
            x_comp = 3                                                                              # [-] Compression ratio between 2.1 and 4
            N = np.ceil(np.log(Project.loc['Delivery pressure',country]/Electrolysers[electrolyser].loc['Max. pressure',country])/np.log(x_comp))                                               # [-] Number of compression stages needed     
            Z_avg = (Compressibility_Factor(Project.loc['Delivery pressure',country], T_delivery)+Compressibility_Factor(Electrolysers[electrolyser].loc['Max. pressure',country], T_amb))/2    # [-] Average compressibility factor
            Z_avg = Z_avg[0]            
            q_m = 10**6/(3600*m*Electrolysers[electrolyser].loc['Power consumption',country])       # [mol/s/MWh] Molar flow
            Electrolysers[electrolyser].loc['Compressor constant',country] = (N*(k/(k-1))*(Z_avg/efficiency_isen)*T_delivery*q_m*R*((Project.loc['Delivery pressure',country]/Electrolysers[electrolyser].loc['Max. pressure',country])**(k-1/k)-1)/10**6)/efficiency_driver #[-] Linearised compressor constant
        
        for pv in PV_types:
            Scenarios[country][electrolyser][pv]={} 
            for sens in Sensitivity:
                Scenarios[country][electrolyser][pv][sens]={}                   # Library set-up accounting for every possible scenario
                
                # Sensitivity analysis variable variation
                
                PV_module_costs = PV_systems[pv].loc['Module costs',country]
                Wind_investment = Wind_choice.loc['Investment costs',country]
                Electrolyser_investment = Electrolysers[electrolyser].loc['Investment costs',country]
                Power_consumption_elec = Electrolysers[electrolyser].loc['Power consumption',country]
                i_r = General_data['Project Economics'].loc['Nominal interest rate',country] 
                
                wscf = 1    # Weight Solar Carbon Footprint (100%)
                wwcf = 1    # Weight Wind Carbon Footprint (100%)
                wecf = 1    # Weight Electrolyser Carbon Footprint (100%)
                
                if sens == 'CS+':
                    PV_module_costs = General_data['Sensitivities'].loc[pv+' +',country]         
                elif sens == 'CS-':              
                    PV_module_costs = General_data['Sensitivities'].loc[pv+' -',country]     
               
                elif sens == 'CW+':
                    Wind_investment = General_data['Sensitivities'].loc[Wind_option.at[0,country]+' +',country]  
                elif sens == 'CW-':
                    Wind_investment = General_data['Sensitivities'].loc[Wind_option.at[0,country]+' -',country]  
                
                elif sens == 'CE+':
                    Electrolyser_investment = General_data['Sensitivities'].loc[electrolyser+' +',country] 
                elif sens == 'CE-':
                    Electrolyser_investment = General_data['Sensitivities'].loc[electrolyser+' -',country] 
                
                elif sens == 'CFS+':
                    wscf = 1 + (General_data['Sensitivities'].loc['CF PV',country]/100) 
                elif sens == 'CFS-':
                    wscf = 1 - (General_data['Sensitivities'].loc['CF PV',country]/100) 
                
                elif sens == 'CFW+':
                    wwcf = 1 + (General_data['Sensitivities'].loc['CF Wind',country]/100) 
                elif sens == 'CFW-':
                    wwcf = 1 - (General_data['Sensitivities'].loc['CF Wind',country]/100) 
                
                elif sens == 'CFE+':
                    wecf = 1 + (General_data['Sensitivities'].loc['CF Electrolyser',country]/100) 
                elif sens == 'CFE-':
                    wecf = 1  - (General_data['Sensitivities'].loc['CF Electrolyser',country]/100) 
                
                elif sens == 'PC+':
                    Power_consumption_elec = General_data['Sensitivities'].loc['PC '+electrolyser+' +',country] 
                elif sens == 'PC-':
                    Power_consumption_elec = General_data['Sensitivities'].loc['PC '+electrolyser+' -',country] 
                elif sens == 'r+':
                    i_r = General_data['Project Economics'].loc['Nominal interest rate',country] * (1+General_data['Sensitivities'].loc['interest variation',country]/100 )
                elif sens == 'r-':
                    i_r = General_data['Project Economics'].loc['Nominal interest rate',country] * (1-General_data['Sensitivities'].loc['interest variation',country]/100 )
                
                # Energy required for annual production target
                
                Energy_required = round(Project.loc['Hydrogen_required',country]*Power_consumption_elec ) 
                
                # Scenario specific characteristics 
                
                # Solar
                CAPEX_solar =1000*(PV_module_costs/(PV_systems[pv].loc['Module % of CAPEX',country]/100))   # [$/MW] 
                OPEX_solar = PV_systems[pv].loc['O&M',country]/100                                          # [% of Initial investment/yr]
                Solar_per_MW = Component_costs(CAPEX_solar,                                                 # Cost function
                                               OPEX_solar,
                                               PV_systems[pv].loc['Lifetime',country],
                                               country)
                Solar_invest_fract = Solar_per_MW[1]/Solar_per_MW[0]            # Fraction of total cost CAPEX

                Solar_CF = wscf*Project.loc['Project lifetime',country]*PV_systems[pv].loc['Yield_solar',country] * PV_systems[pv].loc['Carbon Footprint total (kg CO2/MWh)',country] # Carbon Footprint over lifetime
                # Wind 
                CAPEX_wind = 1000* Wind_investment                              # [$/MW] 
                OPEX_wind = Wind_choice.loc['O&M',country]/100                  # [$/MW/yr]
                Wind_per_MW = Component_costs(CAPEX_wind,                       # Cost function
                                               OPEX_wind,
                                               Wind_choice.loc['Lifetime',country],
                                               country)
                Wind_invest_fract = Wind_per_MW[1]/Wind_per_MW[0]               # Fraction of total cost CAPEX
         
                Wind_CF = wwcf*Project.loc['Project lifetime',country]*Wind_choice.loc['Yield_wind',country]*Wind_choice.loc['Carbon Footprint total (kg CO2/MWh)',country] # Carbon Footprint over lifetime
                
                # Electrolyser
                CAPEX_Electrolyser = 1000*Electrolyser_investment                       # [$/MW] 
                OPEX_Electrolyser = Electrolysers[electrolyser].loc['O&M',country]/100  # [$/MW/yr]
                Electrolyser_per_MW = Component_costs(CAPEX_Electrolyser,               # Cost function
                                               OPEX_Electrolyser,
                                               Electrolysers[electrolyser].loc['Lifetime system',country],
                                               country)
                Electrolyser_invest_fract = Electrolyser_per_MW[1]/Electrolyser_per_MW[0] # Fraction of total cost CAPEX
                    # stack costs
                Stack_costs = 0
                stack_yr = (Energy_required/Electrolysers[electrolyser].loc['Stack lifetime',country])*Electrolysers[electrolyser].loc['Stack investment cost',country]*(1000*Electrolyser_investment)
                for j in range(int(Project.loc['Project lifetime',country])):    
                    Stack_costs = Stack_costs + stack_yr/((1+i_r)**j)
                Elec_CF = wecf*Electrolysers[electrolyser].loc['Carbon Footprint',country] # Carbon Footprint 
               
                # Compressor
                CAPEX_Compressor = 1000*General_data['Centrifugal'].loc['Investment costs',country]*Electrolysers[electrolyser].loc['Compressor constant',country]  # $/MW electrolyser capacity
                OPEX_Compressor = General_data['Centrifugal'].loc['O&M',country]/100    # [$/MW/yr]
                Compressor_per_MW = Component_costs(CAPEX_Compressor,                   # Cost function
                                               OPEX_Compressor,
                                               General_data['Centrifugal'].loc['Lifetime',country],
                                               country)
                Compressor_invest_fract = Compressor_per_MW[1]/Compressor_per_MW[0]     # Fraction of total cost CAPEX
                    
                    # CF compressor included in electrolyser
                
                # Feedstock
                
                Feedstock_C = General_data['Feedstock'].loc['Water use',country]*General_data['Feedstock'].loc['Price desalinated water',country]/1000
                Cooling_C = General_data['Feedstock'].loc['Cooling water',country]*General_data['Feedstock'].loc['Price desalinated water',country]/1000
                
                Feedstock_CF = General_data['Feedstock'].loc['Water use',country]*General_data['Feedstock'].loc['Carbon Footprint water',country]/1000
                Cooling_CF = General_data['Feedstock'].loc['Cooling water',country]*General_data['Feedstock'].loc['Carbon Footprint water',country]/1000
            
                if electrolyser == 'Alkaline':
                    Feedstock_CF = Feedstock_CF + General_data['Feedstock'].loc['Carbon Footprint KOH',country]*Electrolysers[electrolyser].loc['KOH use',country]
                
                # Set-up scenarios dataframe
                Optimization_constants = pd.DataFrame([Solar_per_MW[0],
                                                       Wind_per_MW[0],
                                                       Electrolyser_per_MW[0],
                                                       Stack_costs,
                                                       Compressor_per_MW[0],
                                                       Feedstock_C,
                                                       Cooling_C,
                                                       Solar_CF,
                                                       Wind_CF,
                                                       Elec_CF,
                                                       Feedstock_CF,
                                                       Cooling_CF,
                                                       Solar_invest_fract,
                                                       Wind_invest_fract,
                                                       Electrolyser_invest_fract,
                                                       Compressor_invest_fract,
                                                       Energy_required])
                indices = ['Solar ($/MW)',
                           'Wind ($/MW)',
                           'Electrolyser ($/MW)',
                           'Stacks annual ($/yr)',
                           'Compressor ($/MW)',
                           'Costs Feedstock ($/kg H2)',
                           'Costs Cooling ($/kg H2)',
                           'Carbon Footprint Solar (kg/MW)',
                           'Carbon Footprint Wind (kg/MW)',
                           'Carbon Footprint Electrolyser (kg/MW)',
                           'Carbon Footprint Feedstock (kg/kg H2)',
                           'Carbon Footprint Cooling (kg/kg H2)',
                           'Investment fraction Solar (%)',
                           'Investment fraction Wind (%)',
                           'Investment fraction Electrolyser (%)',
                           'Investment fraction Compressor (%)',
                           "Energy required"]
                Optimization_constants.index = indices
                
                
                Scenarios[country][electrolyser][pv][sens]=Optimization_constants
            
#%% Optimization Definition


## Define run experiment 
 
                             
Countries_run =  [Countries[0]]                        # 'Countries' for all countries, [Countries[0]],[Countries[:1]] etc for selection of countries
Electrolyser_types_run = [Electrolyser_types[0]]        # 'Electrolyser_types' for all electrolysers, [Electrolyser_types[0]] for Alkaline ,[Electrolyser_types[1]] for PEM
PV_types_run = [PV_types[2]]                           # PV_types for all PV types, [PV_types[0]],[PV_types[:1]] etc for selection of PV types

Carbon_price = 0                                        # 0: for weight factor method, 1: for carbon price method 
Carbon_price_max = 500                                  # $/ton : Maximum carbon price in optimisation
lump = 1                                                # 0 for complete LCOH breakdown, 1 for lumped breakdown (Electrolyser, PV, Wind)                                    
steps = 50                                              # Resolution results (amount of calculation steps, minimum for graphs 10 steps)                     
  
# Sensitivity selection
Sens_no         = ['N']                                                         # No sensitivity analysis
Sens_all        = ['N','CS+','CW+','CE+','CS-','CW-','CE-','CFS+','CFW+','CFE+','CFS-','CFW-','CFE-','PC+','PC-','r-','r+'] # Complete sensitivity analysis
Sens_cost       = ['N','CS+','CW+','CE+','CS-','CW-','CE-']                     # Cost sensitivity analysis
Sens_Carb       = ['N','CFS+','CFW+','CFE+','CFS-','CFW-','CFE-']               # Carbon sensitivity analysis
Sens_Pow_con    = ['N','PC+','PC-']                                             # Power consumption sensitivity analysis
Sens_interest   = ['N','r-','r+']                                               # Interest rate sensitivity
                  
Sens_run = Sens_no  # Choice of sensitivity

for country in Countries_run:
    for electrolyser in Electrolyser_types_run:
        for pv in PV_types_run:                    
            for sens in Sens_run:              
                 
                # Set-up dictionaries Objective & Results
                 variables = {}
                 Power_levels_dict = {}
                 Sizes = {}
                 CF = {}
                 Costs = {} 
                 CF_fraction = {}
                 Cost_fraction = {}
                 
                 if Carbon_price == 0:
                     stepsize = int(100/steps)
                     opt_range = 101
                 else:    
                     stepsize = int(Carbon_price_max/steps)
                     opt_range = Carbon_price_max+1
                      
                     
                 for Weight_factor in range(0,opt_range,stepsize):
                        
                        if Carbon_price == 0:
                           Weight_factor = Weight_factor/100
                        else:    
                           Weight_factor = Weight_factor/1000
 
                        CarbonFootprint_solar = Scenarios[country][electrolyser][pv][sens].loc['Carbon Footprint Solar (kg/MW)']  
                        CarbonFootprint_wind =  Scenarios[country][electrolyser][pv][sens].loc['Carbon Footprint Wind (kg/MW)']     
                        CarbonFootprint_electrolyser =  Scenarios[country][electrolyser][pv][sens].loc['Carbon Footprint Electrolyser (kg/MW)']
                                   
                        Solar_costs = Scenarios[country][electrolyser][pv][sens].loc['Solar ($/MW)']
                        Wind_costs =  Scenarios[country][electrolyser][pv][sens].loc['Wind ($/MW)']
                        Electrolyser_costs = Scenarios[country][electrolyser][pv][sens].loc['Electrolyser ($/MW)'] + Scenarios[country][electrolyser][pv][sens].loc['Compressor ($/MW)']
                        
                        if Carbon_price == 0:
                            CarbonFootprint = np.array([CarbonFootprint_solar,CarbonFootprint_wind,CarbonFootprint_electrolyser])
                            CF_normalized = Weight_factor*CarbonFootprint/max(CarbonFootprint)
                            Cost_components = np.array([Solar_costs,Wind_costs,Electrolyser_costs])
                            Costs_normalized = (1-Weight_factor)*Cost_components/max(Cost_components) 
                        else:  # No normalisation when working with carbon footprint  
                            CarbonFootprint = np.array([CarbonFootprint_solar,CarbonFootprint_wind,CarbonFootprint_electrolyser])
                            CF_normalized = Weight_factor*CarbonFootprint 
                            Cost_components = np.array([Solar_costs,Wind_costs,Electrolyser_costs])
                            Costs_normalized = Cost_components 
                        
                        # Optimisation constants Carbon Footprint & Costs of components
                        CF_C = CF_normalized + Costs_normalized
                        
                        Components = [
                            "Solar",
                            "Wind",
                            "Electrolyser" ]
                        
                        CFandCost = {
                            "Solar":CF_C[0],
                            "Wind":CF_C[1],
                            "Electrolyser":CF_C[2]
                            }
                        
                        Hours = list(range(len(Wind_data)))
                        Switch = list(range(len(Wind_data)))
                        
                        ###### Create optimization problem ############################################################
                       
                        # Variables
                        Size_vars = pulp.LpVariable.dicts("Size",Components,0,7000)                 # (name,List,min,max) Maximum 7 GW capacity selected: is certainly not reached
                        Power_vars = pulp.LpVariable.dicts("P", Hours, 0,7000)                      # (name,List,min,max) Maximum 7 GW capacity selected: is certainly not reached
                        prob = pulp.LpProblem("Carbon_and_Cost_minimization", pulp.LpMinimize)      # Problem creation
                           
                        # objective function

                        prob += (
                            pulp.lpSum([Size_vars[c]*CFandCost[c] for c in Components]),            
                            "Minimize_CF_C",
                            ) 
                        
                        # Constraints
                       
                        prob += (
                            pulp.lpSum([Power_vars[h] for h in Hours]) == Scenarios[country][electrolyser][pv][sens].loc["Energy required"],
                            "ProductionRequirement",
                            )
                        
                        for h in Hours:
                            
                            prob += PV_systems[pv].loc['Efficiency',country]*Solar_data.loc[h,country]*Size_vars["Solar"] + Wind_data.loc[h,country]*Size_vars["Wind"] - Power_vars[h] >= 0, "HourlyPowerBalance"+str(h)
                            prob += Size_vars["Electrolyser"] - Power_vars[h] >= 0, "ElectrolyserSizing"+str(h)
                        
                        prob.writeLP("GreenHydrogenSizing.lp")
                        
                        prob.solve(pulp.PULP_CBC_CMD(msg=0))                    # Selected solver: CBC
        
                        ### Process the results
                        
                        for v in prob.variables():
                            variables[str(Weight_factor)+v.name] = v.varValue
                            
                        Power_levels = []
                        for h in Hours:
                            Power_levels.append(variables[str(Weight_factor)+'P_'+str(h)])

                        Power_levels_dict[str(Weight_factor)] = Power_levels      
                        Project_total_production = (Project.loc['Project lifetime',country]*Project.loc['Annual production',country]*10**6)
                        
                        Sizes["W: "+str(Weight_factor)+' Size_Solar']=variables[str(Weight_factor)+'Size_Solar']
                        Sizes["W: "+str(Weight_factor)+' Size_Wind']=variables[str(Weight_factor)+'Size_Wind']
                        Sizes["W: "+str(Weight_factor)+' Size_Electrolyser']=variables[str(Weight_factor)+'Size_Electrolyser']
        
                        # Post-optimization processing of electrolyser limits
                        
                        PowerGeneration = PV_systems[pv].loc['Efficiency',country]*Solar_data[country]*variables[str(Weight_factor)+'Size_Solar'] + Wind_data[country]*variables[str(Weight_factor)+'Size_Wind']
                        H2prod = Power_levels.copy()        # Power levels of electrolyser
                        missed_prod = 0                     # Definition missed production      
                        
                        production_limit = Electrolysers[electrolyser].loc['Lower load limit',country]/100
                        
                        for i in range(len(PowerGeneration)):
                            if PowerGeneration[i] < production_limit*variables[str(Weight_factor)+'Size_Electrolyser']: 
                                H2prodi = 0
                                missed_prod = missed_prod+PowerGeneration[i]
                            elif PowerGeneration[i] > variables[str(Weight_factor)+'Size_Electrolyser']: 
                                H2prodi = variables[str(Weight_factor)+'Size_Electrolyser']
                            else:
                                H2prodi = PowerGeneration[i]
                            H2prod.append(H2prodi) 
                            
                        H2prod = np.array(H2prod)
                        Energy_required = Scenarios[country][electrolyser][pv][sens].loc["Energy required",0]
                        scale_up = 1+missed_prod/Energy_required
                        
                        PowerGeneration_scaled = PowerGeneration*scale_up
                        ELectrolyser_scaled = variables[str(Weight_factor)+'Size_Electrolyser']*scale_up
                        H2prod_scaled = []
                        for i in range(len(PowerGeneration)):
                            if PowerGeneration[i] < 0.1*ELectrolyser_scaled:    # This scenario does not occur
                                H2prodi = 0
                            elif PowerGeneration_scaled[i] > ELectrolyser_scaled:
                                H2prodi = ELectrolyser_scaled
                            else: 
                                H2prodi = PowerGeneration_scaled[i]
                            H2prod_scaled.append(H2prodi) 
                        H2prod_scaled = np.array( H2prod_scaled)
                        Curtailment = -1*( PowerGeneration_scaled-H2prod_scaled)
                        
                    
                        Sizes[str(Weight_factor)]={'Size_Solar':variables[str(Weight_factor)+'Size_Solar']*scale_up,
                                                   'Size_Wind':variables[str(Weight_factor)+'Size_Wind']*scale_up,
                                                   'Size_Electrolyser':variables[str(Weight_factor)+'Size_Electrolyser']*scale_up}
                        
                        # Calculate resulting CF & Costs
                      
                        CF[str(Weight_factor)]=(Sizes[str(Weight_factor)]['Size_Solar']*CarbonFootprint_solar+ Sizes[str(Weight_factor)]['Size_Wind']*CarbonFootprint_wind+ Sizes[str(Weight_factor)]['Size_Electrolyser']*CarbonFootprint_electrolyser )/Project_total_production + Scenarios[country][electrolyser][pv][sens].loc['Carbon Footprint Feedstock (kg/kg H2)'] + Scenarios[country][electrolyser][pv][sens].loc['Carbon Footprint Cooling (kg/kg H2)']
                        CF[str(Weight_factor)] = CF[str(Weight_factor)].iloc[0]
                        
                        if Carbon_price == 1:
                           Costs[str(Weight_factor)]=CF[str(Weight_factor)]*Weight_factor + (Sizes[str(Weight_factor)]['Size_Solar']*Solar_costs+ Sizes[str(Weight_factor)]['Size_Wind']*Wind_costs+ Sizes[str(Weight_factor)]['Size_Electrolyser']*Electrolyser_costs +Scenarios[country][electrolyser][pv][sens].loc['Stacks annual ($/yr)'])/Project_total_production + Scenarios[country][electrolyser][pv][sens].loc['Costs Feedstock ($/kg H2)'] + Scenarios[country][electrolyser][pv][sens].loc['Costs Cooling ($/kg H2)']
                           Costs[str(Weight_factor)] = Costs[str(Weight_factor)].iloc[0]
                        else:
                            Costs[str(Weight_factor)]= (Sizes[str(Weight_factor)]['Size_Solar']*Solar_costs+ Sizes[str(Weight_factor)]['Size_Wind']*Wind_costs+ Sizes[str(Weight_factor)]['Size_Electrolyser']*Electrolyser_costs + Scenarios[country][electrolyser][pv][sens].loc['Stacks annual ($/yr)'])/Project_total_production + Scenarios[country][electrolyser][pv][sens].loc['Costs Feedstock ($/kg H2)'] + Scenarios[country][electrolyser][pv][sens].loc['Costs Cooling ($/kg H2)']
                            Costs[str(Weight_factor)] = Costs[str(Weight_factor)].iloc[0]
                         
                            
                        # Calculate Carbon Footprint contribution different components
                        
                        CF_fract_sol =(Sizes[str(Weight_factor)]['Size_Solar']*CarbonFootprint_solar/Project_total_production)/CF[str(Weight_factor)]
                        CF_fract_wind = (Sizes[str(Weight_factor)]['Size_Wind']*CarbonFootprint_wind/Project_total_production)/CF[str(Weight_factor)]
                        CF_fract_elec = (Sizes[str(Weight_factor)]['Size_Electrolyser']*CarbonFootprint_electrolyser/Project_total_production)/CF[str(Weight_factor)]
                        CF_fract_Feed = Scenarios[country][electrolyser][pv][sens].loc['Carbon Footprint Feedstock (kg/kg H2)']/CF[str(Weight_factor)]  
                        CF_fract_Cool = Scenarios[country][electrolyser][pv][sens].loc['Carbon Footprint Cooling (kg/kg H2)']/CF[str(Weight_factor)]
                        
                        CF_fraction[str(Weight_factor)] = {'Solar':float(CF_fract_sol.iloc[0]),
                                                           'Wind':float(CF_fract_wind.iloc[0]),
                                                           'Electrolyser':float(CF_fract_elec.iloc[0]),  
                                                           'Feedstock':float(CF_fract_Feed.iloc[0]),
                                                           'Cooling':float(CF_fract_Cool.iloc[0])
                                                           }
                        
                        # Calculate Costs contribution different components
                        
                        Cost_fract_I_sol = (Scenarios[country][electrolyser][pv][sens].loc['Investment fraction Solar (%)'])*(Sizes[str(Weight_factor)]['Size_Solar']*Solar_costs/Project_total_production)/Costs[str(Weight_factor)]
                        Cost_fract_V_sol = (1-Scenarios[country][electrolyser][pv][sens].loc['Investment fraction Solar (%)'])*(Sizes[str(Weight_factor)]['Size_Solar']*Solar_costs/Project_total_production)/Costs[str(Weight_factor)]
                        
                        Cost_fract_I_wind = (Scenarios[country][electrolyser][pv][sens].loc['Investment fraction Wind (%)'])* (Sizes[str(Weight_factor)]['Size_Wind']*Wind_costs/Project_total_production)/Costs[str(Weight_factor)]
                        Cost_fract_V_wind = (1-Scenarios[country][electrolyser][pv][sens].loc['Investment fraction Wind (%)'])* (Sizes[str(Weight_factor)]['Size_Wind']*Wind_costs/Project_total_production)/Costs[str(Weight_factor)]
                        
                        Cost_fract_I_elect = (Scenarios[country][electrolyser][pv][sens].loc['Investment fraction Electrolyser (%)'])* (Sizes[str(Weight_factor)]['Size_Electrolyser']*(Scenarios[country][electrolyser][pv][sens].loc['Electrolyser ($/MW)'])/Project_total_production)/Costs[str(Weight_factor)]
                        Cost_fract_V_elect = (1-Scenarios[country][electrolyser][pv][sens].loc['Investment fraction Electrolyser (%)'])* (Sizes[str(Weight_factor)]['Size_Electrolyser']*Scenarios[country][electrolyser][pv][sens].loc['Electrolyser ($/MW)']/Project_total_production)/Costs[str(Weight_factor)]
                        
                        Cost_fract_stack = (Scenarios[country][electrolyser][pv][sens].loc['Stacks annual ($/yr)']/Project_total_production)/Costs[str(Weight_factor)]
                        
                        Cost_fract_I_comp = (Scenarios[country][electrolyser][pv][sens].loc['Investment fraction Compressor (%)'])* (Sizes[str(Weight_factor)]['Size_Electrolyser']*Scenarios[country][electrolyser][pv][sens].loc['Compressor ($/MW)']/Project_total_production)/Costs[str(Weight_factor)]
                        Cost_fract_V_comp = (1-Scenarios[country][electrolyser][pv][sens].loc['Investment fraction Compressor (%)'])* (Sizes[str(Weight_factor)]['Size_Electrolyser']*Scenarios[country][electrolyser][pv][sens].loc['Compressor ($/MW)']/Project_total_production)/Costs[str(Weight_factor)]
                        
                        Cost_fract_Feed = Scenarios[country][electrolyser][pv][sens].loc['Costs Feedstock ($/kg H2)']/Costs[str(Weight_factor)]
                        Cost_fract_Cool = Scenarios[country][electrolyser][pv][sens].loc['Costs Cooling ($/kg H2)']/Costs[str(Weight_factor)]
                        
                        if Carbon_price == 1:
                            Cost_fract_carbon = (CF[str(Weight_factor)]*Weight_factor)/Costs[str(Weight_factor)]
                            
                            
                            if lump == 1:
                                Cost_fraction[str(Weight_factor)] = {'Solar PV':float(Cost_fract_I_sol.iloc[0])+float(Cost_fract_V_sol.iloc[0]),
                                      'Wind':float(Cost_fract_I_wind.iloc[0])+float(Cost_fract_V_wind.iloc[0]),
                                      'Electrolyser':float(Cost_fract_I_elect.iloc[0])+float(Cost_fract_V_elect.iloc[0])+float(Cost_fract_stack.iloc[0])+float(Cost_fract_I_comp.iloc[0])+float(Cost_fract_V_comp.iloc[0])+float(Cost_fract_Feed.iloc[0])+float(Cost_fract_Cool.iloc[0]),
                                      'Carbon price':float(Cost_fract_carbon)
                                      }
                            else:
                                Cost_fraction[str(Weight_factor)] = {'Solar CAPEX':float(Cost_fract_I_sol.iloc[0]),
                                                                  'Solar O&M': float(Cost_fract_V_sol.iloc[0]),
                                                                  'Wind Investment':float(Cost_fract_I_wind.iloc[0]),
                                                                  'Wind O&M':float(Cost_fract_V_wind.iloc[0]),
                                                                  'Production facility CAPEX':float(Cost_fract_I_elect.iloc[0]),
                                                                  'Production facility O&M':float(Cost_fract_V_elect.iloc[0]),
                                                                  'Electrolyser stack':float(Cost_fract_stack.iloc[0]),
                                                                  'Compressor CAPEX':float(Cost_fract_I_comp.iloc[0]),
                                                                  'Compressor O&M':float(Cost_fract_V_comp.iloc[0]),
                                                                  'Feedstock':float(Cost_fract_Feed.iloc[0]),
                                                                  'Cooling':float(Cost_fract_Cool.iloc[0]),
                                                                  'Carbon price':float(Cost_fract_carbon)
                                                                  }                            
                           
                        else:
                            if lump == 1:
                                Cost_fraction[str(Weight_factor)] = {'Solar PV':float(Cost_fract_I_sol.iloc[0])+float(Cost_fract_V_sol.iloc[0]),
                                      'Wind':float(Cost_fract_I_wind.iloc[0])+float(Cost_fract_V_wind.iloc[0]),
                                      'Electrolyser':float(Cost_fract_I_elect.iloc[0])+float(Cost_fract_V_elect.iloc[0])+float(Cost_fract_stack.iloc[0])+float(Cost_fract_I_comp.iloc[0])+float(Cost_fract_V_comp.iloc[0])+float(Cost_fract_Feed.iloc[0])+float(Cost_fract_Cool.iloc[0]),
                                      }
                            else: 
                                Cost_fraction[str(Weight_factor)] = {'Solar CAPEX':float(Cost_fract_I_sol.iloc[0]),
                                                                  'Solar O&M': float(Cost_fract_V_sol.iloc[0]),
                                                                  'Wind Investment':float(Cost_fract_I_wind.iloc[0]),
                                                                  'Wind O&M':float(Cost_fract_V_wind.iloc[0]),
                                                                  'Production facility CAPEX':float(Cost_fract_I_elect.iloc[0]),
                                                                  'Production facility O&M':float(Cost_fract_V_elect.iloc[0]),
                                                                  'Electrolyser stack':float(Cost_fract_stack.iloc[0]),
                                                                  'Compressor CAPEX':float(Cost_fract_I_comp.iloc[0]),
                                                                  'Compressor O&M':float(Cost_fract_V_comp.iloc[0]),
                                                                  'Feedstock':float(Cost_fract_Feed.iloc[0]),
                                                                  'Cooling':float(Cost_fract_Cool.iloc[0])
                                                                  }   
                        
                        # LCOE calculation
                        
                        LCOE_actual = (Sizes[str(Weight_factor)]['Size_Solar']*Solar_costs +  Sizes[str(Weight_factor)]['Size_Wind']*Wind_costs)/(20*(Sizes[str(Weight_factor)]['Size_Solar']*PV_systems[pv].loc['Yield_solar',country] +  Sizes[str(Weight_factor)]['Size_Wind']*Wind_choice.loc['Yield_wind',country]))
                        LCOE_useful = (Sizes[str(Weight_factor)]['Size_Solar']*Solar_costs +  Sizes[str(Weight_factor)]['Size_Wind']*Wind_costs)/(20*Energy_required)
                        LCOE = [LCOE_actual,LCOE_useful]
                        
                        ################## POWER PROFILE GRAPH ##########################################################
                        ### Comment this section out for the creation of power profile charts ###########################
                        #################################################################################################
                        # if Weight_factor == 0.0 or Weight_factor == 0.4 or Weight_factor == 1.0:
                            
                        #         print(f"{country} Weighting: {Weight_factor} CF and {1-Weight_factor} Costs, {pv}&{electrolyser} with {sens}")
                        #         print(f"Solar : {Sizes[str(Weight_factor)]['Size_Solar']} MW")
                        #         print(f"Wind : {Sizes[str(Weight_factor)]['Size_Wind']} MW")
                        #         print(f"Electrolyser : {Sizes[str(Weight_factor)]['Size_Electrolyser']} MW")
                        #         print(f"Off-time correction: {round(100*(scale_up-1),2)}%")
                        #         print(f"LCOH: {round(Costs[str(Weight_factor)],3)} [$/kg H\u2082] -/- CF: {round(CF[str(Weight_factor)],3)} [kg CO\u2082/kg H\u2082]")
                        #         print("============================================================================================")
                            
                            
                        #         ### Choose style
                        #         ## Choose style
                        #         style =  'fivethirtyeight'
                        #         mp.style.use(style)
                             
                             
        
                        #         A = PowerGeneration_scaled.reset_index(drop=True)
                        #         B = pd.Series(H2prod_scaled)
                        #         C = Curtailment.reset_index(drop=True)
                        #         print(sum(C)/Energy_required)
                        #         #[1416:3623]
                        #         xtick_values = [1417, 2160, 2880, 3624]          #[0,     744,  1417, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016,8759 ]# 
                        #         xtick_labels = ['Mar','Apr','May','Jun']          #['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan']# 
                            
                            
                        #         # Plot the generation, electrolyser and curtailment data
                        #         if Weight_factor == 0.0:
                        #             fig, ax = plt.subplots(nrows=3,squeeze=True,figsize=(15, 15)) #[1416:3623]
                        #             ax[0].plot(A[1416:3623], label='Generation',linewidth=2)
                        #             ax[0].plot(B[1416:3623], label='Electrolyser',linewidth=2)
                        #             ax[0].plot(C[1416:3623], label='Curtailment',linewidth=2)
                        #             # Labels, title and legend
                        #             ax[0].set_xlabel('Time (hrs)')
                        #             ax[0].set_ylabel('Power (MW)')
                        #             ax[0].set_title(f'Power Profile Duqm,{electrolyser}, Mono c-Si  w={Weight_factor}')    
                        #             ax[0].set_xticks(xtick_values)
                        #             ax[0].set_xticklabels(xtick_labels)
                        #         elif Weight_factor == 0.4:
                        #             ax[1].plot(A[1416:3623], label='Generation',linewidth=2)
                        #             ax[1].plot(B[1416:3623], label='Electrolyser',linewidth=2)
                        #             ax[1].plot(C[1416:3623], label='Curtailment',linewidth=2)
                        #             # Labels, title and legend
                        #             ax[1].set_xlabel('Time (hrs)')
                        #             ax[1].set_ylabel('Power (MW)')
                        #             ax[1].set_title(f'Power Profile w={Weight_factor}')
                        #             ax[1].set_xticks(xtick_values)
                        #             ax[1].set_xticklabels(xtick_labels)
                        #             ax[1].legend()
                        #             titel_weight = Weight_factor
                        #         elif Weight_factor == 1.0:
                        #             ax[2].plot(A[1416:3623], label='Generation',linewidth=2)
                        #             ax[2].plot(B[1416:3623], label='Electrolyser',linewidth=2)
                        #             ax[2].plot(C[1416:3623], label='Curtailment',linewidth=2)
                        #             # Labels, title and legend
                        #             ax[2].set_xlabel('Time (hrs)')
                        #             ax[2].set_ylabel('Power (MW)')
                        #             ax[2].set_title(f'Power Profile w={Weight_factor}')
                        #             ax[2].set_xticks(xtick_values)
                        #             ax[2].set_xticklabels(xtick_labels)
                        #             plt.subplots_adjust(hspace=0.3)
                        #             plt.savefig('Plots\ProdProfile_'+str(titel_weight)+country[:3]+'_'+pv[:3]+'_'+electrolyser[:3]+sens+'.png', format='png',bbox_inches='tight')
                        #             plt.savefig('Plots\ProdProfile_'+str(titel_weight)+country[:3]+'_'+pv[:3]+'_'+electrolyser[:3]+sens+'.eps', format='eps',bbox_inches='tight')
                                 
                        #             plt.show()
                               
                 Optimization_constants =  Scenarios[country][electrolyser][pv][sens]   
                        
                 Scenarios[country][electrolyser][pv][sens]={
                        "Optimization constants":Optimization_constants,
                        #"Decision variable values":variables,
                        "Power levels":Power_levels_dict,
                        "Component sizes":Sizes,
                        "Carbon Footprint":CF,
                        "Costs":Costs,
                        "CF fraction":CF_fraction,
                        "Cost fraction":Cost_fraction,
                        "LCOE":LCOE}
                    
                 cf_values = list(CF.values())
                 cost_values = list(Costs.values())
                    
                 if sens == 'N':
                     senslabel = ''
                 else:
                     senslabel = f'Sensitivity: {sens}'   
    
                 ## Scatter plot =========================================================================
                 print(f'Carbon footprint {pv} = {CarbonFootprint_solar}')
                 style =  'fivethirtyeight'
                 mp.style.use(style)
                
                 plt.figure(figsize=(8, 8))
                 plt.scatter(cf_values, cost_values, label='Opimal Values', color='b')
               
                # Set labels and title
               
                 if country == Countries[0]:
                      location = 'Duqm'
                 elif country == Countries[1]:
                      location = 'Groningen'
                 elif country == Countries[2]:
                      location = 'Dakhla'
                 else:
                      location = 'Unknown'
               
                 plt.xlabel('Carbon Footprint (kg CO\u2082/kg H\u2082)')
                 plt.ylabel('LCOH ($/kg H\u2082)')
                 plt.title(f"Pareto front: {location}, {electrolyser}, {pv} {senslabel}") # - Sensitivity: {SC},{SCF}
               
               
                # Display & save plot

                 plt.tight_layout()
                 plt.savefig('Plots\Pareto_'+country[:3]+'_'+pv[:3]+'_'+electrolyser[:3]+sens+'.png', format='png',bbox_inches='tight')
                 plt.savefig('Plots\Pareto_'+country[:3]+'_'+pv[:3]+'_'+electrolyser[:3]+sens+'.eps', format='eps',bbox_inches='tight')
                 plt.show()
                                
                #  ## Stacked bar chart =========    LCOH   ============================================================
                 style =  'fivethirtyeight'
                 mp.style.use(style)
     
                 Scenarios[country][electrolyser][pv][sens]['Cost fraction $'] ={}
                 x_labels = list(Scenarios[country][electrolyser][pv][sens]['Cost fraction'].keys())
                  # Extracting the y-axis keys
                 y_labels =list(Scenarios[country][electrolyser][pv][sens]['Cost fraction']["0.0"].keys())
                 ycap_labels = list(Scenarios[country][electrolyser][pv][sens]['Component sizes']["0.0"].keys())
                 SizeSol = []
                 SizeWind = []
                 SizeElec = []
                 for x_label in x_labels:
                    Scenarios[country][electrolyser][pv][sens]['Cost fraction $'][x_label] ={}
   
                    SizeSol.append(Scenarios[country][electrolyser][pv][sens]['Component sizes'][x_label][ycap_labels[0]])
                    SizeWind.append(Scenarios[country][electrolyser][pv][sens]['Component sizes'][x_label][ycap_labels[1]])
                    SizeElec.append(Scenarios[country][electrolyser][pv][sens]['Component sizes'][x_label][ycap_labels[2]])
                   
                    for y_label in y_labels:
                        Scenarios[country][electrolyser][pv][sens]['Cost fraction $'][x_label][y_label] = Scenarios[country][electrolyser][pv][sens]['Cost fraction'][x_label][y_label]*Scenarios[country][electrolyser][pv][sens]["Costs"][x_label]
                       
                 Sizes_chart = {"PV":SizeSol,
                              "Wind":SizeWind,
                              "Electrolyser":SizeElec}
               
                # Extracting the values for each x_label and y_label
                 fractions = {
                    y_label: [Scenarios[country][electrolyser][pv][sens]['Cost fraction $'][x_label][y_label] for x_label in x_labels]
                    for y_label in y_labels}
     
                #   # Plotting the stacked bar chart
                 fig, ax = plt.subplots(figsize=(12, 5))
                
                 
                # colors:         Sol CAP    Sol OP     Wind CAP    Wind OP    Prod Cap   Prod OP    Stack     Comp CAP    Comp OP    Feed     Cooling
                 
                 if Carbon_price == 0:
                  if lump == 0: 
                      colors = ['#e5ae38', '#FFE56F','#30a2da', '#77D9FF', '#fc4f30', '#FF8961', '#6d904f','#1f77b4' ,  '#2ca02c', '#810f7c', '#8b8b8b']
                  else:
                      colors = ['#e5ae38','#30a2da', '#fc4f30']
                 else:
                  if lump == 0: 
                      colors = ['#e5ae38', '#FFE56F','#30a2da', '#77D9FF', '#fc4f30', '#FF8961', '#6d904f','#1f77b4' ,  '#2ca02c', '#810f7c', '#8b8b8b','#2ca02c']
                  else:
                      colors = ['#e5ae38','#30a2da', '#fc4f30', '#8b8b8b']
                  
              # Creating the stacked bars
                 for i, y_label in enumerate(y_labels):
                    ax.bar(x_labels, fractions[y_label], label=y_label, bottom=np.sum(list(fractions.values())[:i], axis=0),color = colors[i]) #
     
                # Set labels and title
                 if Carbon_price == 0:
                     ax.set_xlabel("Weight factor") 
                 else:
                     ax.set_xlabel("Carbon price [$/kg CO\u2082]")
                
                 ax.set_ylabel(u"LCOH [$/kg H\u2082]")
                 ax.set_title(f"Cost distribution: {location}, {electrolyser}, {pv} {senslabel}", fontsize=18,loc='center')
                 xdisp = int(steps/10)
             
                 ax2 = ax.twinx()
                 ax2.plot(x_labels, SizeSol, color = 'olivedrab',marker='o', markevery=xdisp, label='PV')
                 ax2.plot(x_labels, SizeWind,color = 'mediumturquoise', marker='o',markevery=xdisp, label='Wind')
                 ax2.plot(x_labels, SizeElec,color = 'gold', marker='o', markevery=xdisp, label='Electrolyser')
                 ax2.set_ylabel('Installed Capacity (MW)')
                 ax2.grid(False)  
         
                 stacked_legend, labels1 = ax.get_legend_handles_labels()
                 line_legend, labels2 = ax2.get_legend_handles_labels()
                 
                 combined_legend = stacked_legend+line_legend
                 labels = labels1+labels2
                 
                 ax.legend(combined_legend,labels,title="Legend", bbox_to_anchor=(1.18, 1), loc='upper left',fontsize='smaller')
    
                 x_labels_display = x_labels[::xdisp]
                 plt.xticks(x_labels_display,rotation=45)
                 align_yaxis(ax, 0, ax2, 0)
              # Show the plot
                 plt.tight_layout()
             
                 plt.savefig('Plots\LCOH_'+country[:3]+'_'+pv[:3]+'_'+electrolyser[:3]+sens+'.png', format='png',bbox_inches='tight')
                 plt.savefig('Plots\LCOH_'+country[:3]+'_'+pv[:3]+'_'+electrolyser[:3]+sens+'.eps', format='eps',bbox_inches='tight')
            
                 plt.show()
             
            #   ## Stacked bar chart =========   Carbon footprint   ============================================================                       
                 style =  'fivethirtyeight'
                 mp.style.use(style)
             
                 Scenarios[country][electrolyser][pv][sens]['Carbon fraction tot'] ={}
              # Extracting the y-axis keys
                 y_labels =list(Scenarios[country][electrolyser][pv][sens]['CF fraction']["0.0"].keys())
             
                 for x_label in x_labels:
                  Scenarios[country][electrolyser][pv][sens]['Carbon fraction tot'][x_label] ={}
                  for y_label in y_labels:
                     
                      Scenarios[country][electrolyser][pv][sens]['Carbon fraction tot'][x_label][y_label] = Scenarios[country][electrolyser][pv][sens]['CF fraction'][x_label][y_label]*Scenarios[country][electrolyser][pv][sens]["Carbon Footprint"][x_label]
             
              # Extracting the values for each x_label and y_label
                 fractions = {
                  y_label: [Scenarios[country][electrolyser][pv][sens]['Carbon fraction tot'][x_label][y_label] for x_label in x_labels]
                  for y_label in y_labels
              }
 
                 fig, ax = plt.subplots(figsize=(12, 5))
             
              #              Solar     Wind      Stack      Feedstock  Cooling
                 colors_CF = ['#e5ae38','#30a2da', '#6d904f', '#8b8b8b', '#1f77b4']                    

                 for i, y_label in enumerate(y_labels):
                  ax.bar(x_labels, fractions[y_label], label=y_label, bottom=np.sum(list(fractions.values())[:i], axis=0),color=colors_CF[i])
             
             
              # Set labels and title
                 if Carbon_price == 0:
                     ax.set_xlabel("Weight factor") 
                 else:
                     ax.set_xlabel("Carbon price [$/kg CO\u2082]")
              
                 ax.set_ylabel(u"CF [kg CO\u2082/kg H\u2082]")
             
             
                 ax.set_title(f"Carbon distribution: {location}, {electrolyser}, {pv} {senslabel}", fontsize=18,loc='center')
             

             
                 ax2 = ax.twinx()
                 ax2.plot(x_labels, SizeSol, color = 'olivedrab',marker='o',markevery=xdisp, label='PV')
                 ax2.plot(x_labels, SizeWind,color = 'mediumturquoise', marker='o',markevery=xdisp, label='Wind')
                 ax2.plot(x_labels, SizeElec,color = 'gold', marker='o',markevery=xdisp, label='Electrolyser')
        
                 ax2.set_ylabel('Installed Capacity (MW)')
     
                 ax2.grid(False)  
     
                 stacked_legend, labels1 = ax.get_legend_handles_labels()
                 line_legend, labels2 = ax2.get_legend_handles_labels()
             
                 combined_legend = stacked_legend+line_legend
                 labels = labels1+labels2
             
                 ax.legend(combined_legend,labels,title="Legend", bbox_to_anchor=(1.18, 1), loc='upper left',fontsize='smaller')
             
                 x_labels_display = x_labels[::xdisp]
                 plt.xticks(x_labels_display,rotation=45)
             
                 align_yaxis(ax, 0, ax2, 0)
              # Show the plot
                 plt.tight_layout()
                 plt.savefig('Plots\CF_'+country[:3]+'_'+pv[:3]+'_'+electrolyser[:3]+sens+'.png', format='png',bbox_inches='tight')
                 plt.savefig('Plots\CF_'+country[:3]+'_'+pv[:3]+'_'+electrolyser[:3]+sens+'.eps', format='eps',bbox_inches='tight')
                 plt.show()
             
             
             #
                 t = time.localtime()
                 current_time = time.strftime("%H:%M:%S", t)
                 print("Optimisation finished: "+current_time)
             

#%% Plots - Normal
## Select what plots to make by commenting (select section --> ctrl+1)
## Pareto - regions  -------------------------------------------------------------------------------------------------------------------

# style =  'fivethirtyeight'
# mp.style.use(style)

# sens = 'N'
# pv = PV_types[2]
# plt.figure(figsize=(8, 8))

# cf_Oman = list(Scenarios[Countries[0]][Electrolyser_types[0]][PV_types[2]][sens]["Carbon Footprint"].values())
# cf_NL = list(Scenarios[Countries[1]][Electrolyser_types[0]][PV_types[2]][sens]["Carbon Footprint"].values())
# cf_Mor = list(Scenarios[Countries[2]][Electrolyser_types[0]][PV_types[2]][sens]["Carbon Footprint"].values())

# cost_Oman = list(Scenarios[Countries[0]][Electrolyser_types[0]][PV_types[2]][sens]["Costs"].values())
# cost_NL = list(Scenarios[Countries[1]][Electrolyser_types[0]][PV_types[2]][sens]["Costs"].values())
# cost_Mor = list(Scenarios[Countries[2]][Electrolyser_types[0]][PV_types[2]][sens]["Costs"].values())

# plt.scatter(cf_Oman, cost_Oman, label='Duqm')#, color='#008fd5')
# plt.scatter(cf_NL, cost_NL, label='Groningen')#, color='#fc4f30')
# plt.scatter(cf_Mor, cost_Mor, label='Dakhla')#, color='#e5ae38')
# # Set labels and title

# #plt.xlim(0,2)
# #plt.ylim(1.5,3)
# plt.xlabel('Carbon Footprint (kg CO\u2082/kg H\u2082)')
# plt.ylabel('LCOH ($/kg H\u2082)')
# plt.title(f"Pareto: {Electrolyser_types[0]}, Mono c-Si") # - Sensitivity: {SC},{SCF}


# # Show legend
# plt.legend(title="Legend", fontsize='smaller') # loc='upper left',
# plt.tight_layout()
# plt.savefig('Plots\Pareto_regions'+pv[:3]+'.png', format='png',bbox_inches='tight')
# plt.savefig('Plots\Pareto_regions'+pv[:3]+'.eps', format='eps',bbox_inches='tight')
# # #Show the plot
# plt.show()

# # Pareto - electrolysers -------------------------------------------------------------------------------------------------------------------
    

# style =  'fivethirtyeight'
# mp.style.use(style)

# plt.figure(figsize=(8, 8))


# cf_Oman_Alk = list(Scenarios[Countries[0]][Electrolyser_types[0]][PV_types[2]][sens]["Carbon Footprint"].values())
# cf_NL_Alk = list(Scenarios[Countries[1]][Electrolyser_types[0]][PV_types[2]][sens]["Carbon Footprint"].values())
# cf_Mor_Alk = list(Scenarios[Countries[2]][Electrolyser_types[0]][PV_types[2]][sens]["Carbon Footprint"].values())
# cf_Oman_PEM = list(Scenarios[Countries[0]][Electrolyser_types[1]][PV_types[2]][sens]["Carbon Footprint"].values())
# cf_NL_PEM = list(Scenarios[Countries[1]][Electrolyser_types[1]][PV_types[2]][sens]["Carbon Footprint"].values())
# cf_Mor_PEM = list(Scenarios[Countries[2]][Electrolyser_types[1]][PV_types[2]][sens]["Carbon Footprint"].values())


# cost_Oman_Alk = list(Scenarios[Countries[0]][Electrolyser_types[0]][PV_types[2]][sens]["Costs"].values())
# cost_NL_Alk = list(Scenarios[Countries[1]][Electrolyser_types[0]][PV_types[2]][sens]["Costs"].values())
# cost_Mor_Alk = list(Scenarios[Countries[2]][Electrolyser_types[0]][PV_types[2]][sens]["Costs"].values())
# cost_Oman_PEM = list(Scenarios[Countries[0]][Electrolyser_types[1]][PV_types[2]][sens]["Costs"].values())
# cost_NL_PEM = list(Scenarios[Countries[1]][Electrolyser_types[1]][PV_types[2]][sens]["Costs"].values())
# cost_Mor_PEM = list(Scenarios[Countries[2]][Electrolyser_types[1]][PV_types[2]][sens]["Costs"].values())

# plt.scatter(cf_Oman_Alk, cost_Oman_Alk, label='Duqm - AWE', color='#008fd5')
# plt.scatter(cf_Oman_PEM, cost_Oman_PEM, label='Duqm - PEM', color='#77D9FF')
# plt.scatter(cf_NL_Alk, cost_NL_Alk, label='Groningen - AWE', color='#fc4f30')
# plt.scatter(cf_NL_PEM, cost_NL_PEM, label='Groningen - AWE', color='#FF8961')
# plt.scatter(cf_Mor_Alk, cost_Mor_Alk, label='Dakhla - AWE', color='#e5ae38')
# plt.scatter(cf_Mor_PEM, cost_Mor_PEM, label='Dakhla - PEM', color='#FFE56F')
# # Set labels and title

# #plt.xlim(0,2.5)
# #plt.ylim(1.5,4)
# plt.xlabel('Carbon Footprint (kg CO\u2082/kg H\u2082)')
# plt.ylabel('LCOH ($/kg H\u2082)')
# plt.title("Pareto front - Mono c-Si") # - Sensitivity: {SC},{SCF}


# # Show legend
# plt.legend(title="Legend", fontsize='smaller')
# plt.tight_layout()
# plt.savefig('Plots\Pareto_regi_elec_'+pv[:3]+'_'+electrolyser[:3]+'.png', format='png',bbox_inches='tight')
# plt.savefig('Plots\Pareto_regi_elec_'+pv[:3]+'_'+electrolyser[:3]+'.eps', format='eps',bbox_inches='tight')
# #Show the plot
# plt.show()

# # Pareto - pv types ---------------------------------------------------------------------------------------------------------------------------

# style =  'fivethirtyeight'
# mp.style.use(style)

# plt.figure(figsize=(8, 8))

# pv_color =       ['#e5ae38', '#fc4f30','#008fd5' , '#6d904f', '#8b8b8b', '#810f7c']
# pv_color_light = ['#FFE56F', '#FF8961','#65C4FF' , '#A2C681', '#C0C0C0', '#B94EB1']
# pv_color_dark =  ['#A97A00', '#BB0000','#005D9E' , '#3B5D20', '#595959', '#4B004A']
# for i,pv in enumerate(PV_types_run):
    
    
    
#     cf_Oman = list(Scenarios[Countries[0]][Electrolyser_types[0]][pv][sens]["Carbon Footprint"].values())
#     cf_NL = list(Scenarios[Countries[1]][Electrolyser_types[0]][pv][sens]["Carbon Footprint"].values())
#     cf_Mor = list(Scenarios[Countries[2]][Electrolyser_types[0]][pv][sens]["Carbon Footprint"].values())
    
#     cost_Oman = list(Scenarios[Countries[0]][Electrolyser_types[0]][pv][sens]["Costs"].values())
#     cost_NL = list(Scenarios[Countries[1]][Electrolyser_types[0]][pv][sens]["Costs"].values())
#     cost_Mor = list(Scenarios[Countries[2]][Electrolyser_types[0]][pv][sens]["Costs"].values())
    
    
    
#     plt.scatter(cf_Oman, cost_Oman, label='Duqm:'+pv,color = pv_color[i])#, color='#008fd5')
#     plt.scatter(cf_NL, cost_NL, label='Groningen:'+pv,color=pv_color_light[i])#, color='#fc4f30')
#     plt.scatter(cf_Mor, cost_Mor, label='Dakhla:'+pv,color=pv_color_dark[i])#, color='#e5ae38')
    
# # Set labels and title

# #plt.xlim(0.4,2.2)
# #plt.ylim(1.6,3)

# plt.xlabel('Carbon Footprint (kg CO\u2082/kg H\u2082)')
# plt.ylabel('LCOH ($/kg H\u2082)')
# plt.title(f"Pareto front - {Electrolyser_types[0]} with carbon price") # - Sensitivity: {SC},{SCF}
# handles, labels = plt.gca().get_legend_handles_labels()
# order=[0,3,6,1,4,7,2,5,8]
# # Show legend

# plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],title="Legend", fontsize='smaller') #loc='upper left',
# plt.tight_layout()
# plt.savefig('Plots\Pareto_pvtypes_allregions_'+sens+'.png', format='png',bbox_inches='tight')
# plt.savefig('Plots\Pareto_pvtypes_allregions_'+sens+'.eps', format='eps',bbox_inches='tight')
# # #Show the plot
# plt.show()

# Pareto - pv types tax credit ---------------------------------------------------------------------------------------------------------------------------

# style =  'fivethirtyeight'
# mp.style.use(style)

# plt.figure(figsize=(8, 8))

# pv_color =       ['#e5ae38', '#fc4f30','#008fd5' , '#6d904f', '#8b8b8b', '#810f7c']
# pv_color_light = ['#FFE56F', '#FF8961','#65C4FF' , '#A2C681', '#C0C0C0', '#B94EB1']
# pv_color_dark =  ['#A97A00', '#BB0000','#005D9E' , '#3B5D20', '#595959', '#4B004A']
# for i,pv in enumerate(PV_types_run):
    
    
    
#     cf_Oman2 = list(Scenarios[Countries[0]][Electrolyser_types[0]][pv][sens]["Carbon Footprint"].values())
#     cf_NL2 = list(Scenarios[Countries[1]][Electrolyser_types[0]][pv][sens]["Carbon Footprint"].values())
#     cf_Mor2 = list(Scenarios[Countries[2]][Electrolyser_types[0]][pv][sens]["Carbon Footprint"].values())
    
#     cost_Oman2 = list(Scenarios[Countries[0]][Electrolyser_types[0]][pv][sens]["Costs"].values())
#     cost_NL2 = list(Scenarios[Countries[1]][Electrolyser_types[0]][pv][sens]["Costs"].values())
#     cost_Mor2 = list(Scenarios[Countries[2]][Electrolyser_types[0]][pv][sens]["Costs"].values())
    
#     tax_credit = 3
#     for j,cf in enumerate(cf_Oman2):
#         if cf < 0.45:
#             cost_Oman2[j]=cost_Oman2[j]-tax_credit
#         elif cf >= 0.45 and cf <= 1.5: 
#             cost_Oman2[j]=cost_Oman2[j]-0.334*tax_credit
#         elif cf > 1.5 and cf <= 2.5:   
#             cost_Oman2[j]=cost_Oman2[j]-0.25*tax_credit
#         elif cf > 2.5 and cf <= 4:  
#             cost_Oman2[j]=cost_Oman2[j]-0.2*tax_credit
            
#     for j,cf in enumerate(cf_NL2):
#         if cf < 0.45:
#             cost_NL2[j]=cost_NL2[j]-tax_credit
#         elif cf >= 0.45 and cf <= 1.5: 
#             cost_NL2[j]=cost_NL2[j]-0.334*tax_credit
#         elif cf > 1.5 and cf <= 2.5:   
#             cost_NL2[j]=cost_NL2[j]-0.25*tax_credit
#         elif cf > 2.5 and cf <= 4:  
#             cost_NL2[j]=cost_NL2[j]-0.2*tax_credit    
    
#     for j,cf in enumerate(cf_Mor2):
#         if cf < 0.45:
#             cost_Mor2[j]=cost_Mor2[j]-tax_credit
#         elif cf >= 0.45 and cf <= 1.5: 
#             cost_Mor2[j]=cost_Mor2[j]-0.334*tax_credit
#         elif cf > 1.5 and cf <= 2.5:   
#             cost_Mor2[j]=cost_Mor2[j]-0.25*tax_credit
#         elif cf > 2.5 and cf <= 4:  
#             cost_Mor2[j]=cost_Mor2[j]-0.2*tax_credit        
    
#     plt.plot(cf_Oman2, cost_Oman2, label='Duqm:'+pv,color = pv_color[i])#, color='#008fd5')
#     plt.plot(cf_NL2, cost_NL2, label='Groningen:'+pv,color=pv_color_light[i])#, color='#fc4f30')
#     plt.plot(cf_Mor2, cost_Mor2, label='Dakhla:'+pv,color=pv_color_dark[i])#, color='#e5ae38')
#     # plt.scatter(cf_Oman2, cost_Oman2, label='Duqm:'+pv,color = pv_color[i])#, color='#008fd5')
#     # plt.scatter(cf_NL2, cost_NL2, label='Groningen:'+pv,color=pv_color_light[i])#, color='#fc4f30')
#     # plt.scatter(cf_Mor2, cost_Mor2, label='Dakhla:'+pv,color=pv_color_dark[i])#, color='#e5ae38')
    
# # Set labels and title

# #plt.xlim(0.4,2.2)
# #plt.ylim(1.6,3)

# plt.xlabel('Carbon Footprint (kg CO\u2082/kg H\u2082)')
# plt.ylabel('LCOH ($/kg H\u2082)')
# plt.title(f"Pareto front - {Electrolyser_types[0]} with IRA tax credit") # - Sensitivity: {SC},{SCF}
# handles, labels = plt.gca().get_legend_handles_labels()
# order=[0,3,6,1,4,7,2,5,8]
# # Show legend

# plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],title="Legend", fontsize='smaller') #loc='upper left',
# plt.tight_layout()
# plt.savefig('Plots\Pareto_pvtypes_allregions_taxcredit_'+sens+'.png', format='png',bbox_inches='tight')
# plt.savefig('Plots\Pareto_pvtypes_allregions_taxcredit_'+sens+'.eps', format='eps',bbox_inches='tight')
# # #Show the plot
# plt.show()

#%% Plots - Sensitivities

# Sens_no         = ['N']
# Sens_all        = ['N','CS+','CW+','CE+','CS-','CW-','CE-','CFS+','CFW+','CFE+','CFS-','CFW-','CFE-','PC+','PC-','r-','r+']
# Sens_cost       = ['N','CS+','CW+','CE+','CS-','CW-','CE-']
# Sens_Carb       = ['N','CFS+','CFW+','CFE+','CFS-','CFW-','CFE-']
# Sens_Pow_con    = ['N','PC+','PC-']
# Sens_interest   = ['N','r-','r+']


# Pareto - Sensitivities  -Base case- Cost variation -------------------------------------------------------------------------------------------------------

# for country in Countries_run:
#     for pv in PV_types_run:
        
#         style =  'fivethirtyeight'
#         mp.style.use(style)
#         plt.figure(figsize=(8, 8))
        
#         color_sens = ['#e5ae38', '#fc4f30','#008fd5' , '#FFE56F', '#FF8961','#65C4FF' ,'#A97A00', '#BB0000','#005D9E']
#         Sens_plot = Sens_cost
        
#         for i,sens in enumerate(Sens_plot[1:]):
#             cf_sens = list(Scenarios[country][Electrolyser_types[0]][pv][sens]["Carbon Footprint"].values())
#             cost_sens = list(Scenarios[country][Electrolyser_types[0]][pv][sens]["Costs"].values())
#             plt.plot(cf_sens, cost_sens, label=sens,color = color_sens[i],marker='o',linestyle= ':')
        
        
#         cf_Oman = list(Scenarios[country][Electrolyser_types[0]][pv]['N']["Carbon Footprint"].values())
#         cost_Oman = list(Scenarios[country][Electrolyser_types[0]][pv]['N']["Costs"].values())
#         plt.plot(cf_Oman, cost_Oman, label='Base-case',color = '#810f7c',marker='o',linestyle= ':' )
#         # Set labels and title
       
        
#         # plt.xlim(xlimmin,xlimmax )
#         # plt.ylim(ylimmin,ylimmax)
        
#         plt.xlabel('Carbon Footprint (kg CO\u2082/kg H\u2082)')
#         plt.ylabel('LCOH ($/kg H\u2082)')
#         plt.title(f"{country}:{electrolyser}&{pv} sensitivity analysis: Costs") # - Sensitivity: {SC},{SCF}
        
#         # Show legend
#         plt.legend(title="Legend",fontsize='smaller')
#         plt.tight_layout()
#         plt.savefig('Plots\Pareto_Sens_C_'+country[:3]+pv[:3]+'_'+electrolyser[:3]+'.png', format='png',bbox_inches='tight')
#         plt.savefig('Plots\Pareto_Sens_C_'+country[:3]+pv[:3]+'_'+electrolyser[:3]+'.eps', format='eps',bbox_inches='tight')
#         # #Show the plot
#         plt.show()
        
        
#         # Pareto - Sensitivities  -Base case- Carbon variation -------------------------------------------------------------------------------------------------------
        
        # style =  'fivethirtyeight'
        # mp.style.use(style)
        # plt.figure(figsize=(8, 8))
        # color_sens = ['#e5ae38', '#fc4f30','#008fd5' , '#FFE56F', '#FF8961','#65C4FF' ,'#A97A00', '#BB0000','#005D9E']
        
        # Sens_plot = Sens_Carb
        # for i,sens in enumerate(Sens_plot[1:]):
        #     cf_sens = list(Scenarios[country][Electrolyser_types[0]][pv][sens]["Carbon Footprint"].values())
        #     cost_sens = list(Scenarios[country][Electrolyser_types[0]][pv][sens]["Costs"].values())
        #     plt.plot(cf_sens, cost_sens, label=sens,color = color_sens[i],marker='o',linestyle= ':')

            
        # cf_Oman = list(Scenarios[country][Electrolyser_types[0]][pv]['N']["Carbon Footprint"].values())
        # cost_Oman = list(Scenarios[country][Electrolyser_types[0]][pv]['N']["Costs"].values())
        # plt.plot(cf_Oman, cost_Oman, label='Base-case',color = '#810f7c',marker='o',linestyle= ':' )
        # # Set labels and title
            
        
        # # plt.xlim(xlimmin,xlimmax )
        # # plt.ylim(ylimmin,ylimmax)
        
        # plt.xlabel('Carbon Footprint (kg CO\u2082/kg H\u2082)')
        # plt.ylabel('LCOH ($/kg H\u2082)')
        # plt.title(f"{country}:{electrolyser}&{pv} sensitivity analysis: CF") # - Sensitivity: {SC},{SCF}
        
        # # Show legend
        # plt.legend(title="Legend",fontsize='smaller')
        # plt.tight_layout()
        # plt.savefig('Plots\Pareto_Sens_CF_'+country[:3]+pv[:3]+'_'+electrolyser[:3]+'.png', format='png',bbox_inches='tight')
        # plt.savefig('Plots\Pareto_Sens_CF_'+country[:3]+pv[:3]+'_'+electrolyser[:3]+'.eps', format='eps',bbox_inches='tight')
        # # #Show the plot
        # plt.show()
        
#         # #Pareto - Sensitivities  -Base case- Power Consumption (Only Oman)-------------------------------------------------------------------------------------------------------
       
        # style =  'fivethirtyeight'
        # mp.style.use(style)
        # plt.figure(figsize=(8, 8))
        # color_sens = ['#e5ae38', '#fc4f30','#008fd5' , '#FFE56F', '#FF8961','#65C4FF' ,'#A97A00', '#BB0000','#005D9E']
        
        # Sens_plot = Sens_Pow_con
        # for i,sens in enumerate(Sens_plot[1:]):
        #     cf_sens = list(Scenarios[country][Electrolyser_types[0]][pv][sens]["Carbon Footprint"].values())
        #     cost_sens = list(Scenarios[country][Electrolyser_types[0]][pv][sens]["Costs"].values())
        #     plt.plot(cf_sens, cost_sens, label=sens,color = color_sens[i],marker='o',linestyle= ':')

            
        # cf_Oman = list(Scenarios[country][Electrolyser_types[0]][pv]['N']["Carbon Footprint"].values())
        # cost_Oman = list(Scenarios[country][Electrolyser_types[0]][pv]['N']["Costs"].values())
        # plt.plot(cf_Oman, cost_Oman, label='Base-case',color = '#810f7c',marker='o',linestyle= ':' )
        # # Set labels and title
            
        
        # # plt.xlim(xlimmin,xlimmax )
        # # plt.ylim(ylimmin,ylimmax)
        
        # plt.xlabel('Carbon Footprint (kg CO\u2082/kg H\u2082)')
        # plt.ylabel('LCOH ($/kg H\u2082)')
        # plt.title(f"{country}:{electrolyser}&{pv} sensitivity analysis: PC") # - Sensitivity: {SC},{SCF}
        
        # # Show legend
        # plt.legend(title="Legend",fontsize='smaller')
        # plt.tight_layout()
        # plt.savefig('Plots\Pareto_Sens_PC_'+country[:3]+pv[:3]+'_'+electrolyser[:3]+'.png', format='png',bbox_inches='tight')
        # plt.savefig('Plots\Pareto_Sens_PC_'+country[:3]+pv[:3]+'_'+electrolyser[:3]+'.eps', format='eps',bbox_inches='tight')
        # # #Show the plot
        # plt.show()
        
        
        # # Pareto - Sensitivities  -Base case- Interest variation -------------------------------------------------------------------------------------------------------
       
        # style =  'fivethirtyeight'
        # mp.style.use(style)
        # plt.figure(figsize=(8, 8))
        # color_sens = ['#e5ae38', '#fc4f30','#008fd5' , '#FFE56F', '#FF8961','#65C4FF' ,'#A97A00', '#BB0000','#005D9E']
        
        # Sens_plot = Sens_interest
        # for i,sens in enumerate(Sens_plot[1:]):
        #     cf_sens = list(Scenarios[country][Electrolyser_types[0]][pv][sens]["Carbon Footprint"].values())
        #     cost_sens = list(Scenarios[country][Electrolyser_types[0]][pv][sens]["Costs"].values())
        #     plt.plot(cf_sens, cost_sens, label=sens,color = color_sens[i],marker='o',linestyle= ':')

            
        # cf_Oman = list(Scenarios[country][Electrolyser_types[0]][pv]['N']["Carbon Footprint"].values())
        # cost_Oman = list(Scenarios[country][Electrolyser_types[0]][pv]['N']["Costs"].values())
        # plt.plot(cf_Oman, cost_Oman, label='Base-case',color = '#810f7c',marker='o',linestyle= ':' )
        # # Set labels and title
            
        
        # # plt.xlim(xlimmin,xlimmax )
        # # plt.ylim(ylimmin,ylimmax)
        
        # plt.xlabel('Carbon Footprint (kg CO\u2082/kg H\u2082)')
        # plt.ylabel('LCOH ($/kg H\u2082)')
        # plt.title(f"{country}:{electrolyser}&{pv} sensitivity analysis: r") # - Sensitivity: {SC},{SCF}
        
        # # Show legend
        # plt.legend(title="Legend",fontsize='smaller')
        # plt.tight_layout()
        # plt.savefig('Plots\Pareto_Sens_r_'+country[:3]+pv[:3]+'_'+electrolyser[:3]+'.png', format='png',bbox_inches='tight')
        # plt.savefig('Plots\Pareto_Sens_r_'+country[:3]+pv[:3]+'_'+electrolyser[:3]+'.eps', format='eps',bbox_inches='tight')
        # # #Show the plot
        # plt.show()

# Pareto - Sensitivities  -Interest rate -------------------------------------------------------------------------------------------------------
# for country in Countries_run:
#     style =  'fivethirtyeight'
#     mp.style.use(style)
#     plt.figure(figsize=(8, 8))
    
#     Sens_plot = Sens_interest
    
#     if country == Countries[0]:
#             location = 'Duqm'
#     elif country == Countries[1]:
#             location = 'Groningen'
#     elif country == Countries[2]:
#             location = 'Dakhla'
     
#     pv_color =       ['#e5ae38', '#fc4f30','#008fd5' , '#6d904f', '#8b8b8b', '#810f7c']
#     pv_color_light = ['#FFE56F', '#FF8961','#65C4FF' , '#A2C681', '#C0C0C0', '#B94EB1']
#     pv_color_dark =  ['#A97A00', '#BB0000','#005D9E' , '#3B5D20', '#595959', '#4B004A']
#     for i,pv in enumerate(PV_types_run):
        
#         cf_base = list(Scenarios[country][Electrolyser_types[0]][pv][Sens_plot[0]]["Carbon Footprint"].values())
#         cf_plus = list(Scenarios[country][Electrolyser_types[0]][pv][Sens_plot[1]]["Carbon Footprint"].values())
#         cf_min  = list(Scenarios[country][Electrolyser_types[0]][pv][Sens_plot[2]]["Carbon Footprint"].values())
        
#         cost_base = list(Scenarios[country][Electrolyser_types[0]][pv][Sens_plot[0]]["Costs"].values())
#         cost_plus = list(Scenarios[country][Electrolyser_types[0]][pv][Sens_plot[1]]["Costs"].values())
#         cost_min  = list(Scenarios[country][Electrolyser_types[0]][pv][Sens_plot[2]]["Costs"].values())

#         plt.scatter(cf_base, cost_base, label=f'{pv} {Sens_plot[0]}' ,color = pv_color[i])#, color='#008fd5')
#         plt.scatter(cf_plus, cost_plus, label=f'{pv} {Sens_plot[1]}' ,color=pv_color_light[i])#, color='#fc4f30')
#         plt.scatter(cf_min , cost_min , label=f'{pv} {Sens_plot[2]}' ,color=pv_color_dark[i])#, color='#e5ae38')
        
#     # Set labels and title

#     #plt.xlim(0.4,2.2)
#     #plt.ylim(1.6,3)
#     plt.xlabel('Carbon Footprint (kg CO\u2082/kg H\u2082)')
#     plt.ylabel('LCOH ($/kg H\u2082)')
#     plt.title(f"Pareto front - {location},{Electrolyser_types[0]} Sensitivity: {sens}") # - Sensitivity: {SC},{SCF}
#     handles, labels = plt.gca().get_legend_handles_labels()
     
#     # Show legend
#     plt.legend(title="Legend",fontsize='smaller')
#     plt.tight_layout()
#     plt.savefig('Plots\Pareto_Sens_r_'+country[:3]+pv[:3]+'_'+electrolyser[:3]+'.png', format='png',bbox_inches='tight')
#     plt.savefig('Plots\Pareto_Sens_r_'+country[:3]+pv[:3]+'_'+electrolyser[:3]+'.eps', format='eps',bbox_inches='tight')
#     # #Show the plot
#     plt.show()

# # Pareto - Sensitivities  -Power consumption -------------------------------------------------------------------------------------------------------

# style =  'fivethirtyeight'
# mp.style.use(style)
# plt.figure(figsize=(8, 8))
# pv = PV_types[2]
# pv_color =       ['#e5ae38', '#fc4f30','#008fd5' , '#6d904f', '#8b8b8b', '#810f7c']
# pv_color_light = ['#FFE56F', '#FF8961','#65C4FF' , '#A2C681', '#C0C0C0', '#B94EB1']
# pv_color_dark =  ['#A97A00', '#BB0000','#005D9E' , '#3B5D20', '#595959', '#4B004A']

# Sens_plot = Sens_Pow_con
# for i,sens in enumerate(Sens_plot[1:]):
#     colorspc = pv_color_light
#     if i == 0:
#         colorspc = pv_color_dark

#     cf_sens = list(Scenarios[country][Electrolyser_types[0]][pv][sens]["Carbon Footprint"].values())
#     cost_sens = list(Scenarios[country][Electrolyser_types[0]][pv][sens]["Costs"].values())
#     plt.plot(cf_sens, cost_sens, label=Electrolyser_types[0]+' '+sens,color = colorspc[1],marker='o',linestyle= ':')
#     cf_sens = list(Scenarios[country][Electrolyser_types[1]][pv][sens]["Carbon Footprint"].values())
#     cost_sens = list(Scenarios[country][Electrolyser_types[1]][pv][sens]["Costs"].values())
#     plt.plot(cf_sens, cost_sens, label=Electrolyser_types[1]+' '+sens,color = colorspc[2],marker='o',linestyle= ':')    

# cf_Oman = list(Scenarios[country][Electrolyser_types[0]][pv]['N']["Carbon Footprint"].values())
# cost_Oman = list(Scenarios[country][Electrolyser_types[0]][pv]['N']["Costs"].values())
# plt.plot(cf_Oman, cost_Oman, label='Alkaline N',color = pv_color[1],marker='o',linestyle= ':' )
# cf_PEM = list(Scenarios[country][Electrolyser_types[1]][pv]['N']["Carbon Footprint"].values())
# cost_PEM = list(Scenarios[country][Electrolyser_types[1]][pv]['N']["Costs"].values())
# plt.plot(cf_PEM, cost_PEM, label='PEM N',color = pv_color[2],marker='o',linestyle= ':' )
# # Set labels and title


# # plt.xlim(xlimmin,xlimmax )
# # plt.ylim(ylimmin,ylimmax)

# plt.xlabel('Carbon Footprint (kg CO\u2082/kg H\u2082)')
# plt.ylabel('LCOH ($/kg H\u2082)')
# plt.title(f"{country}:{electrolyser}&{pv} sensitivity analysis: PC") # - Sensitivity: {SC},{SCF}

# # Show legend
# plt.legend(title="Legend",fontsize='smaller')
# plt.tight_layout()
# plt.savefig('Plots\Pareto_Sens_pc_'+country[:3]+pv[:3]+'_'+electrolyser[:3]+'.png', format='png',bbox_inches='tight')
# plt.savefig('Plots\Pareto_Sens_pc_'+country[:3]+pv[:3]+'_'+electrolyser[:3]+'.eps', format='eps',bbox_inches='tight')
# # #Show the plot
# plt.show()

# Pareto - pv types - Power consumption ---------------------------------------------------------------------------------------------------------------------------

# style =  'fivethirtyeight'
# mp.style.use(style)

# plt.figure(figsize=(8, 8))

# pv_color =       ['#e5ae38', '#fc4f30','#008fd5' , '#6d904f', '#8b8b8b', '#810f7c']
# pv_color_light = ['#FFE56F', '#FF8961','#65C4FF' , '#A2C681', '#C0C0C0', '#B94EB1']
# pv_color_dark =  ['#A97A00', '#BB0000','#005D9E' , '#3B5D20', '#595959', '#4B004A']
# for i,pv in enumerate(PV_types_run):
    
#     cf_Oman = list(Scenarios[Countries[0]][Electrolyser_types[0]][pv]['N']["Carbon Footprint"].values())
#     cf_NL = list(Scenarios[Countries[0]][Electrolyser_types[0]][pv]['PC-']["Carbon Footprint"].values())
#     cf_Mor = list(Scenarios[Countries[0]][Electrolyser_types[0]][pv]['PC+']["Carbon Footprint"].values())

#     cost_Oman = list(Scenarios[Countries[0]][Electrolyser_types[0]][pv]['N']["Costs"].values())
#     cost_NL = list(Scenarios[Countries[0]][Electrolyser_types[0]][pv]['PC-']["Costs"].values())
#     cost_Mor = list(Scenarios[Countries[0]][Electrolyser_types[0]][pv]['PC+']["Costs"].values())

#     plt.scatter(cf_Oman, cost_Oman, label=pv+' N',color = pv_color[i])#, color='#008fd5')
#     plt.scatter(cf_NL, cost_NL, label=pv+' PC-',color=pv_color_light[i])#, color='#fc4f30')
#     plt.scatter(cf_Mor, cost_Mor, label=pv+' PC+',color=pv_color_dark[i])#, color='#e5ae38')
    
# # Set labels and title

# #plt.xlim(0.4,2.2)
# #plt.ylim(1.6,3)
# plt.xlabel('Carbon Footprint (kg CO\u2082/kg H\u2082)')
# plt.ylabel('LCOH ($/kg H\u2082)')
# plt.title("Pareto front - PV types Sensitivity: PC") # - Sensitivity: {SC},{SCF}
# handles, labels = plt.gca().get_legend_handles_labels()
# order=[0,3,6,1,4,7,2,5,8]
# # Show legend

# plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],title="Legend", fontsize='smaller') #loc='upper left',
# plt.tight_layout()
# plt.savefig('Plots\Pareto_pvtypes_PCsens_'+sens+'.png', format='png',bbox_inches='tight')
# plt.savefig('Plots\Pareto_pvtypes_PCsens_'+sens+'.eps', format='eps',bbox_inches='tight')
# # #Show the plot
# plt.show()
 

#%% Get allround best performer

# import math

# All_distances={}

# for pv in PV_types_run:
#         cf_Oman = list(Scenarios[Countries[0]][Electrolyser_types[0]][pv][sens]["Carbon Footprint"].values())
#         cf_NL = list(Scenarios[Countries[1]][Electrolyser_types[0]][pv][sens]["Carbon Footprint"].values())
#         cf_Mor = list(Scenarios[Countries[2]][Electrolyser_types[0]][pv][sens]["Carbon Footprint"].values())
        
#         cost_Oman = list(Scenarios[Countries[0]][Electrolyser_types[0]][pv][sens]["Costs"].values())
#         cost_NL = list(Scenarios[Countries[1]][Electrolyser_types[0]][pv][sens]["Costs"].values())
#         cost_Mor = list(Scenarios[Countries[2]][Electrolyser_types[0]][pv][sens]["Costs"].values())
        
#         dist_Oman = {}
#         dist_NL = {}
#         dist_Mor = {}
        
#         for i,cf in enumerate(cf_Oman):
#                 distOm = math.sqrt(cf**2+cost_Oman[i]**2)
#                 distNL = math.sqrt(cf_NL[i]**2+cost_NL[i]**2)
#                 distMor = math.sqrt(cf_Mor[i]**2+cost_Oman[i]**2)
#                 dist_Oman[i]=distOm
#                 dist_NL[i]=distNL
#                 dist_Mor[i]=distMor
                
#         All_distances['Duqm'+pv]=min(dist_Oman,key=dist_Oman.get)
#         All_distances['Gro'+pv]=min(dist_NL,key=dist_NL.get)
#         All_distances['Mor'+pv]=min(dist_Mor,key=dist_Mor.get)


#%% RANDOM CODE - SOME USEFUL STUF
# fig, ax = plt.subplots(figsize=(20, 5))
# # Plot the generation, electrolyser and curtailment data
# ax.plot(AveragePowerYield['Time'],X[0,0]*AveragePowerYield['Solar (MW)']+X[0,1]*AveragePowerYield['Wind (MW)'], label='Generation')
# ax.plot(AveragePowerYield['Time'], X[0,-len(AveragePowerYield):], label='Electrolyser')
# ax.plot(AveragePowerYield['Time'],X[0,0]*AveragePowerYield['Solar (MW)']+X[0,1]*AveragePowerYield['Wind (MW)']- X[0,-len(AveragePowerYield):], label='Curtailment')

# # Readable time axis
# plt.xticks(rotation=90)
# x_ticks = AveragePowerYield['Time'][::6]  # Select every 6th value
# ax.set_xticks(x_ticks)

# # Labels, title and legend
# ax.set_xlabel('Time')
# ax.set_ylabel('Power (MW)')
# ax.set_title('Total Generation, Electrolyser demand and curtailment vs Time')
# ax.legend()

# plt.show()


# sol=[]
# wind=[]
# elec=[]
# CF=[]
# LC=[]
    
# Solar_upper = round(Energy_required_avg/Yield_solar_avg)
# Wind_upper = round(Energy_required_avg/Yield_wind_avg)
# Electrolyser_minimum = Energy_required/8760
# for i in range(0,Wind_upper,100):
#     xi=i    
#     yi=-(Solar_upper/Wind_upper)*xi + Solar_upper
#     eleci=max(AveragePowerYield.loc[:,'Solar (MW)']*xi+AveragePowerYield.loc[:,'Wind (MW)']*yi)
#     CFi=(xi*CarbonFootprint_solar+yi*CarbonFootprint_wind + 6*Electrolyser_minimum)/(1000*Hydrogen_required_avg)
#     LCi=(xi*CAPEX_solar+yi*CAPEX_wind + 400*Electrolyser_minimum) 
#     sol.append(xi)
#     wind.append(yi)
#     elec.append(eleci)
#     CF.append(CFi)
#     LC.append(LCi)
    
# plt.subplot(2, 1, 1)  
# plt.scatter(sol, wind)
# #plt.scatter(sol, elec)
# plt.scatter(X[:,0],X[:,1],color='#F39C12')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title("Performance")
# # Second plot: Scatter plot of CF and LC
# plt.subplot(2, 1, 2)  
# plt.scatter(CF, LC)
# plt.scatter(F[:,0],F[:,1],color='#F39C12')
# plt.xlabel('CF')
# plt.ylabel('LC')

# plt.tight_layout()

# plt.show()






# PowerGeneration = np.array(X[0,0]*RenewablePowerYield['Solar (MW)']+X[0,1]*RenewablePowerYield['Wind (MW)'])
# H2prod = []
# for i in range(len(PowerGeneration)):
#     if PowerGeneration[i] < 0.1*max(X[0,3:]):
#         H2prodi = 0
#     elif PowerGeneration[i] > max(X[0,3:]):
#         H2prodi = max(X[0,3:])
#     else:
#         H2prodi = PowerGeneration[i]
#     H2prod.append(H2prodi) 
# H2prod = np.array(H2prod)
# Energy_used= sum(H2prod)
# Energy_generated = sum(PowerGeneration)

# scale_down = Energy_required/Energy_used

# PowerGeneration_scaled = PowerGeneration*scale_down
# ELectrolyser_scaled = max(X[0,3:])*scale_down
# H2prod_scaled = []
# for i in range(len(PowerGeneration)):
#     if PowerGeneration[i] < 0.1*ELectrolyser_scaled:
#         H2prodi = 0
#     elif PowerGeneration_scaled[i] > ELectrolyser_scaled:
#         H2prodi = ELectrolyser_scaled
#     else: 
#         H2prodi = PowerGeneration_scaled[i]
#     H2prod_scaled.append(H2prodi) 
# H2prod_scaled = np.array( H2prod_scaled)

# fig, ax = plt.subplots(figsize=(40, 5))
# # Plot the generation, electrolyser and curtailment data
# ax.plot(PowerGeneration_scaled[:2000], label='Generation')
# ax.plot(H2prod_scaled[:2000], label='Electrolyser')
# ax.plot(PowerGeneration_scaled[:2000]-H2prod_scaled[:2000], label='Curtailment')
# # Labels, title and legend
# ax.set_xlabel('Time')
# ax.set_ylabel('Power (MW)')
# ax.set_title('Total Generation, Electrolyser demand and curtailment vs Time '+Country+str(run+1) )
# ax.legend()
# plt.savefig('Plots\PowerProfile'+Country+str(run + 1)+ ID_flag +'.png', format='png',bbox_inches='tight')
# plt.savefig('Plots\PowerProfile'+Country+str(run + 1)+ ID_flag +'.eps', format='eps',bbox_inches='tight')
# plt.show()
# # Interesting results
# Electrolyser_size = ELectrolyser_scaled
# Solar_size = X[0,0]*scale_down
# Wind_size = X[0,1]*scale_down
# CarbonFootprint = (CarbonFootprint_solar * Solar_size * Yield_solar_avg + CarbonFootprint_wind *Wind_size*Yield_wind_avg + 6*ELectrolyser_scaled)/(1000*Hydrogen_required)       #Carbon Footprint
# LCOH = (Solar_anual * Solar_size + Wind_anual *Wind_size + Electrolyser_anual* Electrolyser_size)/(1000*Hydrogen_required)      #Costs
# CapacityFactor_electrolyser = Energy_required/(8760*ELectrolyser_scaled)
# Curtailment = (1-(Energy_required/(Solar_size*Yield_solar+Wind_size*Yield_wind)))*100
# Results.loc[Country+' '+str(run + 1)+ID_flag]= [CarbonFootprint,LCOH,Electrolyser_size,Solar_size,Wind_size,CapacityFactor_electrolyser, Curtailment]
# key = f'{Country}{run}{ID_flag}'
# Decisions[key]=[F,X,C_ieq,C_eq]

## POWER PROFILE Wind & Solar #########################################################################

# for country in Countries:
#     style =  'fivethirtyeight'
#     mp.style.use(style)
    
#     if country == Countries[0]:
#         location = 'Duqm'
#     elif country == Countries[1]:
#         location = 'Groningen'
#     elif country == Countries[2]:
#         location = 'Dakhla'
    
   
#     A_sol = Solar_data[country]
#     B_wind = Wind_data[country]
    
    
#     #     xtick_values = [0,     744,  1417, 2160, 2880, 3624, 4344, 5088, 5832, 6552, 7296, 8016,8759 ]
#     #     xtick_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan']
     
#     xtick_values = [0        ,744]
#     xtick_labels = ["Jan '20","Feb '20"]
    
#     fig, ax = plt.subplots(nrows=2,figsize=(5, 5))
#     # Plot the generation, electrolyser and curtailment data
#     ax[0].plot(A_sol[:745], label='Solar',linewidth=0.5,color = '#e5ae38' )
#     ax[0].set_ylabel('$C_f$ Solar')
#     ax[0].set_xticks(xtick_values)
#     ax[0].set_xticklabels(xtick_labels)
#     ax[0].set_title(f' {location}')
#     ax[1].plot(B_wind[:745], label='Wind',linewidth=0.5)
#     ax[1].set_xlabel('Time')
#     ax[1].set_ylabel('$C_f$ Wind')
#     ax[1].set_xticks(xtick_values)
#     ax[1].set_xticklabels(xtick_labels)
#     plt.subplots_adjust(hspace=0.4)
    
#     plt.savefig('Plots\Weather_profile_'+country[:3]+'.png', format='png',bbox_inches='tight')
#     plt.savefig('Plots\Weather_profile_'+country[:3]+'.eps', format='eps',bbox_inches='tight')
    
#     plt.show()

## EXTRACT SENSITIVITY DATA #############################################################################
# indices = [round(i/10, 1) for i in range(11)]  # [0.0, 0.1, ..., 1.0]


# Cost_change = {}
# Carbon_change = {}
# for country in Countries_run:
#     for electrolyser in Electrolyser_types_run:
#         for pv in PV_types_run:                
#             for sens in Sens_run[1:]:
#                 if not (country == 'Duqm, Oman' and electrolyser == 'Alkaline' and pv == 'CdTe'):    
#                     C_change_list = []
#                     CF_change_list = []
#                     for key in Scenarios[country][electrolyser][pv][sens]["Costs"].keys():
#                         C_percentage = ((Scenarios[country][electrolyser][pv][sens]['Costs'][key]/Scenarios[country][electrolyser][pv]['N']['Costs'][key])-1)*100
#                         CF_percentage = ((Scenarios[country][electrolyser][pv][sens]['Carbon Footprint'][key]/Scenarios[country][electrolyser][pv]['N']['Carbon Footprint'][key])-1)*100
#                         C_change_list.append(C_percentage)
#                         CF_change_list.append(CF_percentage)
#                     entry_name = country[:3]+electrolyser[:3]+pv[:3]+sens
#                     Cost_change[entry_name]=C_change_list
#                     Carbon_change[entry_name]=CF_change_list

# COST_CHANGES = pd.DataFrame(Cost_change,index=indices)
# CF_CHANGES = pd.DataFrame(Carbon_change,index=indices)

# column_strings = ['CS+', 'CW+', 'CE+', 'CS-', 'CW-', 'CE-', 'CFS+', 'CFW+', 'CFE+', 'CFS-', 'CFW-', 'CFE-']

# filtered_data_c = {}
# filtered_data_cf = {}

# filtered_data_cncf = {}
# for col_string in column_strings:
#     filtered_df_c = COST_CHANGES.filter(like=col_string)
#     filtered_data_c[col_string] = filtered_df_c
    
#     filtered_df_cf = CF_CHANGES.filter(like=col_string)
#     filtered_data_cf[col_string] = filtered_df_cf
    
#     filtered_df_cncf = Sens_data.filter(like=col_string)
#     filtered_data_cf[col_string] = filtered_df_cncf



#%% Print runtime
end_time = time.time()

# Calculate the elapsed time
elapsed_time = (end_time - start_time)/60

# Print the runtime in seconds
print("Runtime: {} minutes".format(elapsed_time))
 