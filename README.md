# DataHack @ Odiasay: Solar Energy Forecast
Forked from [mriganv/Machine-Learning-Model-for-Solar-Energy-Forecast](https://github.com/mriganv/Machine-Learning-Model-for-Solar-Energy-Forecast)

## About

This is a data science project about developing a machine learning regression model to accurately predict the rate of solar output measured as power realtive to a baseline of capacity.
The project was presented during April 2025 [DataHack](https://datahack.org.il/) @ [Odiasay](https://www.madaney.net/odyssey) hackaton.

## Data Collection

The model is trained using real data obtained from three sources 
* A dataset which measures the rate of solar output measured as a % of baseline of capacity between 2014 and 2018, collected from real-life example. 
* Weather dataset from an API call to www.worldweatheronline.com for Hanover, Massachusetts location between  2014 and 2018. 
* Solar Irradiance datasets from www.nsrdb.nrel.gov between 2014 and 2018, which includes data on solar and weather values for variables such as Global Horizontal Irradiance (GHI), Direct Horizontal Irradiance (DHI), Direct Normal Irradiance (DNI), Wind Speed, Temperature and Solar Zenith Angle downloaded from the NSRDB. 

## Relative Power Distribution
Relative power average and standard deviation for each hour of the day in the dataset.

![power_hourly](output/relative_power_by_hour.png)

Average weekly relative power distribution.

![power_weekly](output/avg_relative_power_by_week.png)

## Physical Model   

Irradiance is the power per unit area received from the sun in the form of electromagnetic radiation in the wavelength range of the measuring instrument.
Irradiance is often measured in watts per square meter (W/m²).
The irradiance data is used to calculate the power output of a solar panel.
[read more.](https://pvpmc.sandia.gov/modeling-guide/1-weather-design-inputs/irradiance-insolation/)

Key terms:
* [Global Horizontal Irradiance](https://pvpmc.sandia.gov/modeling-guide/1-weather-design-inputs/irradiance-insolation/global-horizontal-irradiance/) (GHI) - The total amount of shortwave radiation received from above by a surface horizontal to the ground.
  GHI may be calculated from direct normal irradiance (DNI) and diffuse horizontal irradiance (DHI) using the following equation:
  
  $$GHI = DNI \cdot \cos(\theta_Z) + DHI$$
  
* [Plane of Array](https://pvpmc.sandia.gov/modeling-guide/1-weather-design-inputs/plane-of-array-poa-irradiance/) (POA) Irradiance - The amount of solar radiation received by a photovoltaic panel, taking into account the angle of the sun relative to the panel and the angle of the panel relative to the ground.
  Mathematically POA irradiance, $E_{POA}$, is:
  
  $$E_{POA} = E_b + E_g + E_d$$
 
  where $E_b$ is the POA beam component, 
  $E_g$ is the POA ground-reflected component, and 
  $E_d$ is the POA sky-diffuse component.


* [Point-value models](https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/point-value-models/pvwatts/) -
These types of models are designed to predict a point (usually the maximum power point ($P_{mp}$) on the IV curve
as a function of environmental variables (e.g., irradiance, cell temperature, and spectral content, for example).
  - [PVWatts](https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/point-value-models/pvwatts/) - 
A point-value model developed by the National Renewable Energy Laboratory (NREL) that estimates the energy production of grid-connected photovoltaic systems.
