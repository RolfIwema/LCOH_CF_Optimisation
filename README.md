# LCOH_CF_Optimisation
Resulting model from the MSc Thesis of Rolf Iwema. This model can be used to optimise a theoretical hydrogen production chain for chosen locations for a low cost and a low carbon footprint. 

**How to install the model?**

1. Download the 'Hydrogen_production_optimization' python file and the 'Model input' excel file. 
2. Make sure to place both files in the same folder, otherwise the python model will not run. 
3. Install all necessary modules, these modules are listed at the top of the python file.
4. You can start working in the model.

**How does the model work?**

The python file, reads the inputs from the 'Model Input' excel file.
The python file loads all information from this file and sorts all characteristics in the necessary dataframes. 

Before running the model, make sure to go to lines 420-438 to configure the scenarios that you want to run. 
How the configuration works, is explained in the comments at those lines.

After running, some graphs should appear that display the scenario performance for different weight factors. 
To get to the actual data, open the dataframe Scenarios, and navigate through the multiple layers to the desired scenario. 
**How to add new locations**

Adding new locations is as simple as copying an example test location in the excel file, and changing the data specific for that location. 
The weather data can be obtained from www.renewables.ninja, at this location, three years worth of data must be downloaded and placed in the corresponding excel tabs. Go to the website, select the desired location, and configure the type of data you want to download.

_Solar_
1. At the renewables.ninja website, select the solar photovoltaic power (PV).
2. Choose 1000 kW as capacity. 
3. Choose the pv system configurations according to your preferences.
4. Now click run, and save hourly output as CSV.
5. Repeat these steps for the three years you want to consider.
6. Go to global solar atlas website and note the mean solar irradiance for that location.

_Wind_
1. At the renewables.ninja website, select the wind power.
2. Choose the capacity of the turbine you're considering. 
3. Choose the hub height you are considering.
4. Choose the Turbine model you are considering. (For all data I used Vestas V164 8000, as it is the biggest available turbine) 
5. Now click run, and save hourly output as CSV.
6. Repeat these steps for the three years you want to consider.
7. Go to global wind atlas website and note the mean wind speed at 100 m for that location.

_General_
1. Alter all country specific characteristics that need to be changed.
2. Check that all new entries are correct in Model_general. Should be the casem but just to be safe.  

**How to add new electrolyser or PV technology**
1. Copy an existing PV or Electrolyser type.
2. Edit the characteristics
3. Add new tech in Model_General
4. Run model
   
**How to add new energy technologies**
1. Add required characteristics of the new technology to Country inputs - General.
2. Make sure to add the new technology in the Model_General tab.
3. Make a new tab, Country inputs - New Tech, by copying wind or solar.
4. Fill the new tab with an energy yield for the new technology. Preferabbly for a 1MW system, or or a bigger sytem, but then make sure in the model to bring this down to a 1MW representation. 
5. Go to the python model
6. Follow the same logic that is used for wind&solar.
7. Nake sure to also apply the same logic in the data visualisation part.
8. Run model
9. Correct mistakes
10. Run model again
11. Iterate ;)
